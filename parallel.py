"""Parallel self-play: CPU-only worker processes + a shared GPU inference server.

The single-process MCTS is GIL-bound on legal-move generation and tree
bookkeeping (~90% of search time is CPU, ~10% GPU).  Threads can't fix that:
python-chess move generation and the Python tree descent don't release the GIL,
so N threads round-robin on one core instead of running in parallel.

The fix is processes:

  * P self-play WORKER PROCESSES each own an MCTS + chess board and run the
    CPU-heavy parts (descent, legal-move generation, encoding, backprop) in
    parallel across cores.  They never touch CUDA.
  * ONE inference SERVER (a thread in the trainer process) owns the net on the
    GPU.  Workers submit batches of *encoded* positions; the server coalesces
    requests from ALL workers into a single large forward pass and returns the
    logits + values.  The GPU stays fed, and the CPU's only job is producing
    the leaves the GPU needs.

Each worker has its OWN request/response queue pair (no cross-worker tagging,
no pickling queues inside queues).  The server round-robins every channel,
concatenates whatever is pending, forwards once, and replies per-worker.
"""

import queue
import threading
import time

import numpy as np
import torch

from selfplay import play_game


class InferenceClient:
    """Duck-types torch.nn.Module for MCTS: ``__call__(x) -> (logits, values)``.

    Lives in a worker process.  Sends encoded positions to the server over
    ``request_queue`` and blocks for the answer on ``response_queue`` (which is
    private to this worker).  MCTS only needs ``.eval()`` and ``net(x)``, both
    of which this provides, so the MCTS code is unchanged.
    """

    def __init__(self, request_queue, response_queue):
        self._req = request_queue
        self._resp = response_queue

    def eval(self):           # MCTS calls net.eval() on construction
        pass

    def train(self, mode=True):  # never used during search; present for safety
        pass

    def __call__(self, x):
        xs = x.detach().cpu().numpy().astype(np.float32, copy=False)
        self._req.put(xs)
        logits, values = self._resp.get()
        return torch.from_numpy(logits), torch.from_numpy(values)


class InferenceServer:
    """GPU inference server (a thread in the trainer process).

    ``channels`` is a list of ``(request_queue, response_queue)``, one per
    worker.  Each cycle it drains every channel, coalesces the encoded
    positions into ONE forward pass, and replies to each worker on its own
    queue.  A short grace wait lets stragglers accumulate so the GPU gets
    fat batches even when workers drift out of lockstep.
    """

    def __init__(self, net, device, channels, max_batch=4096, min_batch=256, wait_secs=0.002):
        self.net = net
        self.device = device
        self.channels = channels
        self.max_batch = max_batch
        self.min_batch = min_batch
        self.wait_secs = wait_secs
        self._stop = threading.Event()
        # Guards self.net so the trainer can swap in fresh weights without a
        # torn read racing an in-flight forward pass.
        self.lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def update_weights(self, state_dict):
        """Atomically load fresh weights into the served net (called by trainer)."""
        with self.lock:
            self.net.load_state_dict(state_dict)

    def _drain(self):
        """Collect all currently-pending encoded batches from every channel."""
        jobs = []   # (resp_q, xs_np)
        total = 0
        for req_q, resp_q in self.channels:
            while total < self.max_batch:
                try:
                    xs = req_q.get_nowait()
                except queue.Empty:
                    break
                jobs.append((resp_q, xs))
                total += xs.shape[0]
        return jobs, total

    def _serve(self):
        while not self._stop.is_set():
            jobs, total = self._drain()
            if not jobs:
                time.sleep(self.wait_secs)
                continue

            # If the coalesced batch is thin, pause briefly so more workers can
            # catch up before we spend a forward pass on a small batch.
            if total < self.min_batch:
                time.sleep(self.wait_secs)
                more, extra = self._drain()
                jobs += more
                total += extra

            xs_cat = np.concatenate([x for _, x in jobs], axis=0)
            with self.lock:
                with torch.no_grad():
                    logits, values = self.net(torch.from_numpy(xs_cat).to(self.device))
            logits_np = logits.float().cpu().numpy()
            values_np = values.float().cpu().numpy()

            idx = 0
            for resp_q, xs in jobs:
                n = xs.shape[0]
                resp_q.put((logits_np[idx:idx + n], values_np[idx:idx + n]))
                idx += n


def worker_loop(cfg, request_queue, response_queue, result_queue, seed, stop_event):
    """Entry point for a self-play worker PROCESS (module-level for 'spawn').

    Plays full games and pushes each game's ``(state, pi, z)`` examples to
    ``result_queue`` until ``stop_event`` is set.  ``cfg`` is the trainer's
    Config, mutated here to ``device='cpu'`` so the worker never touches CUDA.
    """
    import random

    # Each worker must stay single-threaded.  torch and numpy would otherwise
    # spawn their own thread pools, and 12-16 workers x N threads oversubscribes
    # the CPU and *slows* self-play down.  One worker == one core == no
    # contention; the GPU is the shared resource, not the CPU thread pool.
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg.device = "cpu"  # workers are pure CPU; the server owns the GPU
    client = InferenceClient(request_queue, response_queue)

    while not stop_event.is_set():
        examples = play_game(client, cfg)
        result_queue.put(examples)
