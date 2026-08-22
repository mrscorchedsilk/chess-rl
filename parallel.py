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

Sprint A/D reliability contract:

  * Every reply is an explicit 3-tuple ``(payload, values, kind)``:
      - ``kind == "sparse"`` (default): payload is one ``(legal_idx, logits)``
        pair per request position -- the network logits restricted to that
        position's legal actions.  The server decodes each encoded position
        back to a board (``planes_to_board``) to compute the legal mask, so
        workers never send action lists over the wire.
      - ``kind == "dense"``: payload is the full ``(n, policy_size)`` logits.
      - ``kind == "error"``: the server failed (forward pass exception, decode
        failure, ...).  ``InferenceServer.get_error()`` carries the message and
        the server thread stops serving (``is_alive()`` -> False).
  * ``InferenceClient.__call__`` is bounded: it waits at most ``timeout``
    seconds for a reply and raises ``RuntimeError`` instead of hanging, and it
    raises ``RuntimeError`` on an ``"error"`` reply.  No unbounded
    ``Queue.get()`` anywhere on the request path.
  * ``worker_loop`` sends result envelopes ``{"kind": "game"|"error",
    "worker": <id>, "examples"|"traceback": ...}`` so the trainer can tell a
    finished game from a dead worker and attribute failures to a worker id.
    Passing ``worker_id=None`` (the legacy callers) keeps the v1 raw-examples
    protocol on the success path.
"""

import json
import os
import queue
import threading
import time

import chess
import numpy as np
import torch

import encoding
from selfplay import play_game


def _profile_event(event, **fields):
    """Append one opt-in JSONL profiling event.

    ``CHESS_PROFILE_JSONL`` is deliberately checked at call time so spawned
    workers inherit the same destination without any queue or shared Python
    object.  A single ``os.write`` to an ``O_APPEND`` descriptor keeps each
    short event line intact when several workers emit concurrently.  When the
    variable is unset the hot path performs only one environment lookup.
    """
    path = os.environ.get("CHESS_PROFILE_JSONL")
    if not path:
        return
    row = {
        "event": str(event),
        "pid": os.getpid(),
        "wall_time": time.time(),
        "monotonic_ns": time.perf_counter_ns(),
    }
    row.update(fields)
    payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def planes_to_board(planes):
    """Invert a v2-encoded position back into a ``chess.Board``.

    ``planes`` is the encoder's (12*history_steps + 8, 8, 8) float32 stack.
    Only the CURRENT position is reconstructed: piece planes ``planes[:12]``
    (white P N B R Q K then black P N B R Q K) plus the trailing 8 meta planes
    (side to move, castling K Q k q, en-passant target, halfmove clock,
    repetition).  Older history planes are ignored.  The result is rebuilt
    through FEN so python-chess' own validation applies, and the returned board
    has the same pieces, side to move, castling rights, en-passant square and
    therefore the same legal moves as the original.
    """
    planes = np.asarray(planes, dtype=np.float32)
    meta = planes.shape[0] - 8

    piece = planes[:12]
    occupied = piece.sum(axis=0) > 0.5
    plane_id = np.argmax(piece, axis=0)

    rows = []
    for rank in range(7, -1, -1):  # FEN lists rank 8 first
        row = []
        empty = 0
        for file in range(8):
            if occupied[rank, file]:
                if empty:
                    row.append(str(empty))
                    empty = 0
                pid = int(plane_id[rank, file])
                color = chess.WHITE if pid < 6 else chess.BLACK
                piece_type = (pid % 6) + 1
                row.append(chess.Piece(piece_type, color).symbol())
            else:
                empty += 1
        if empty:
            row.append(str(empty))
        rows.append("".join(row))

    turn = "w" if planes[meta].max() > 0.5 else "b"
    castling = ""
    if planes[meta + 1].max() > 0.5:
        castling += "K"
    if planes[meta + 2].max() > 0.5:
        castling += "Q"
    if planes[meta + 3].max() > 0.5:
        castling += "k"
    if planes[meta + 4].max() > 0.5:
        castling += "q"
    castling = castling or "-"

    ep_plane = planes[meta + 5]
    if ep_plane.max() > 0.5:
        r, f = np.unravel_index(np.argmax(ep_plane), ep_plane.shape)
        ep = chess.square_name(chess.square(int(f), int(r)))
    else:
        ep = "-"

    halfmove = max(0, int(round(float(planes[meta + 6].max()) * 100.0)))
    fen = f"{'/'.join(rows)} {turn} {castling} {ep} {halfmove} 1"
    return chess.Board(fen)


class InferenceClient:
    """Duck-types torch.nn.Module for MCTS: ``__call__(x) -> (logits, values)``.

    Lives in a worker process.  Sends encoded positions to the server over
    ``request_queue`` and waits (bounded by ``timeout``) for the answer on
    ``response_queue`` (which is private to this worker).  MCTS only needs
    ``.eval()`` and ``net(x)``, both of which this provides, so the MCTS code
    is unchanged.

    Replies follow the 3-tuple protocol ``(payload, values, kind)``: sparse
    replies are reconstructed into a dense ``(n, policy_size)`` logits tensor
    with zeros on every illegal action, dense replies pass through, and error
    replies raise ``RuntimeError`` (never a silent garbage tensor).
    """

    def __init__(self, request_queue, response_queue, policy_size=None, timeout=60.0):
        self._req = request_queue
        self._resp = response_queue
        if policy_size is None:
            from config import Config

            policy_size = Config().policy_size
        self.policy_size = policy_size
        self.timeout = timeout

    def eval(self):           # MCTS calls net.eval() on construction
        pass

    def train(self, mode=True):  # never used during search; present for safety
        pass

    def __call__(self, x):
        xs = x.detach().cpu().numpy().astype(np.float32, copy=False)
        try:
            self._req.put(xs, timeout=self.timeout)
        except queue.Full:
            raise RuntimeError(
                f"inference request queue stayed full for {self.timeout}s"
            ) from None
        _profile_event("inference_request_sent", positions=int(xs.shape[0]))
        wait_started_ns = time.perf_counter_ns()
        try:
            payload, values, kind = self._resp.get(timeout=self.timeout)
        except queue.Empty:
            raise RuntimeError(
                f"inference server did not respond within {self.timeout}s"
            ) from None
        _profile_event(
            "inference_reply_received",
            positions=int(xs.shape[0]),
            kind=str(kind),
            wait_ms=(time.perf_counter_ns() - wait_started_ns) / 1_000_000.0,
        )

        if kind == "error":
            msg = payload if isinstance(payload, str) and payload else "inference server error"
            raise RuntimeError(msg)
        if kind == "sparse":
            logits = np.zeros((len(payload), self.policy_size), dtype=np.float32)
            for i, (idx, vals) in enumerate(payload):
                logits[i, idx] = vals
        elif kind == "dense":
            logits = payload
        else:
            raise RuntimeError(f"unknown inference reply kind {kind!r}")
        # Mirror nn.Module semantics: outputs live on the input's device (the
        # caller moved x onto cfg.device before the call).  Workers are CPU-only,
        # so in production this is a no-op; it keeps MCTS working when a client
        # is used from a CUDA process too.
        return (torch.from_numpy(logits).to(x.device),
                torch.from_numpy(values).to(x.device))


class InferenceServer:
    """GPU inference server (a thread in the trainer process).

    ``channels`` is a list of ``(request_queue, response_queue)``, one per
    worker.  Each cycle it drains every channel, coalesces the encoded
    positions into ONE forward pass, and replies to each worker on its own
    queue.  A short grace wait lets stragglers accumulate so the GPU gets
    fat batches even when workers drift out of lockstep.

    With ``sparse_response=True`` (default) the reply for each position is the
    network logits restricted to that position's legal actions (computed by
    decoding the planes back to a board); with ``sparse_response=False`` the
    full dense logits are returned.  Any exception inside the serve loop
    (forward pass, decode, ...) is captured: every pending requester gets an
    ``"error"`` reply, ``get_error()`` exposes the message, and the serve
    thread stops (``is_alive()`` -> False) so the trainer can fail the
    iteration instead of hanging.
    """

    def __init__(self, net, device, channels, max_batch=4096, min_batch=256,
                 wait_secs=0.002, sparse_response=True):
        self.net = net
        self.device = device
        self.channels = channels
        self.max_batch = max_batch
        self.min_batch = min_batch
        self.wait_secs = wait_secs
        self.sparse_response = sparse_response
        self._error = None
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

    def get_error(self):
        """Error message if the serve loop died, else None."""
        return self._error

    def is_alive(self):
        """False once the server has failed (or been stopped)."""
        return self._error is None and self._thread.is_alive()

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
            jobs = []
            try:
                jobs, total = self._drain()
                if not jobs:
                    time.sleep(self.wait_secs)
                    continue

                # If the coalesced batch is thin, pause briefly so more workers
                # can catch up before we spend a forward pass on a small batch.
                if total < self.min_batch:
                    time.sleep(self.wait_secs)
                    more, extra = self._drain()
                    jobs += more
                    total += extra

                self._process(jobs)
            except Exception as exc:  # noqa: BLE001 - bounded failure, never hang
                self._error = f"{type(exc).__name__}: {exc}"
                for resp_q, _ in jobs:
                    try:
                        resp_q.put((self._error, None, "error"))
                    except Exception:  # noqa: BLE001 - queue may be closed
                        pass
                break

    def _process(self, jobs):
        """Coalesce `jobs`, run one forward pass, and reply per requester."""
        xs_cat = np.concatenate([x for _, x in jobs], axis=0)
        profiling = bool(os.environ.get("CHESS_PROFILE_JSONL"))

        def mark_time():
            if profiling and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            return time.perf_counter_ns() if profiling else 0

        _profile_event("batch_formed", jobs=len(jobs), positions=int(xs_cat.shape[0]))
        total_started = mark_time()
        x_device = torch.from_numpy(xs_cat).to(self.device)
        h2d_done = mark_time()
        with self.lock:
            with torch.no_grad():
                logits, values = self.net(x_device)
        forward_done = mark_time()
        logits_np = logits.float().cpu().numpy()
        values_np = values.float().cpu().numpy()
        d2h_done = mark_time()

        if self.sparse_response:
            boards = [planes_to_board(x) for x in xs_cat]
            legal_idx = [
                np.array([encoding.move_to_index(m) for m in b.legal_moves],
                         dtype=np.int32)
                for b in boards
            ]
        else:
            legal_idx = None
        legal_done = mark_time()
        if profiling:
            _profile_event(
                "inference_batch_completed",
                jobs=len(jobs),
                positions=int(xs_cat.shape[0]),
                h2d_ms=(h2d_done - total_started) / 1_000_000.0,
                forward_ms=(forward_done - h2d_done) / 1_000_000.0,
                d2h_ms=(d2h_done - forward_done) / 1_000_000.0,
                legal_ms=(legal_done - d2h_done) / 1_000_000.0,
                total_ms=(legal_done - total_started) / 1_000_000.0,
            )

        idx = 0
        for resp_q, xs in jobs:
            n = xs.shape[0]
            if self.sparse_response:
                payload = [
                    (legal_idx[idx + k], logits_np[idx + k, legal_idx[idx + k]])
                    for k in range(n)
                ]
                kind = "sparse"
            else:
                payload = logits_np[idx:idx + n]
                kind = "dense"
            resp_q.put((payload, values_np[idx:idx + n], kind))
            _profile_event("inference_reply_sent", positions=int(n), kind=kind)
            idx += n


def _put_result(result_queue, envelope, stop_event):
    """Queue.put that treats a closed/shut-down queue as a stop signal."""
    try:
        result_queue.put(envelope)
        return True
    except (ValueError, OSError):
        stop_event.set()
        return False


def worker_loop(cfg, request_queue, response_queue, result_queue, seed, stop_event,
                worker_id=None, generation_value=None):
    """Entry point for a self-play worker PROCESS (module-level for 'spawn').

    Plays full games and pushes one result envelope per game to
    ``result_queue`` until ``stop_event`` is set:

      * success: ``{"kind": "game", "worker": worker_id, "examples": [...]}``
      * failure: ``{"kind": "error", "worker": worker_id, "traceback": "..."}``
        followed by a clean exit, so a broken worker is a bounded, attributable
        error instead of a silently dead process.

    ``cfg`` is the trainer's Config, mutated here to ``device='cpu'`` so the
    worker never touches CUDA.  For backward compatibility, ``worker_id=None``
    keeps the v1 raw-examples protocol on the success path (used by train.py /
    bench_parallel.py, which pass no worker id).
    """
    import os
    import random
    import traceback

    # Each worker must stay single-threaded.  torch and numpy would otherwise
    # spawn their own thread pools, and 12-16 workers x N threads oversubscribes
    # the CPU and *slows* self-play down.  One worker == one core == no
    # contention; the GPU is the shared resource, not the CPU thread pool.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg.device = "cpu"  # workers are pure CPU; the server owns the GPU
    timeout = getattr(cfg, "inference_timeout_seconds", None)
    if timeout is None:
        timeout = getattr(cfg, "result_timeout_seconds", 60.0)
    client = InferenceClient(
        request_queue, response_queue,
        policy_size=getattr(cfg, "policy_size", None),
        timeout=timeout,
    )
    _profile_event("worker_started", worker=worker_id, seed=int(seed))

    try:
        while not stop_event.is_set():
            game_generation = (
                int(generation_value.value) if generation_value is not None else None
            )
            game_started_ns = time.perf_counter_ns()
            _profile_event(
                "game_started", worker=worker_id, generation=game_generation,
            )
            profile_path = os.environ.get("CHESS_PROFILE_JSONL")
            if profile_path:
                examples = play_game(
                    client,
                    cfg,
                    on_ply=lambda ply: _profile_event(
                        "ply_completed", worker=worker_id,
                        generation=game_generation, ply=int(ply),
                    ),
                )
            else:
                # Preserve the legacy two-argument callback contract for tests
                # and downstream callers when profiling is disabled.
                examples = play_game(client, cfg)
            _profile_event(
                "game_completed",
                worker=worker_id,
                generation=game_generation,
                plies=len(examples),
                examples=len(examples),
                game_ms=(time.perf_counter_ns() - game_started_ns) / 1_000_000.0,
            )
            envelope = examples if worker_id is None else {
                "kind": "game", "worker": worker_id, "examples": examples,
                "generation": game_generation,
            }
            if not _put_result(result_queue, envelope, stop_event):
                return
    except Exception:  # noqa: BLE001 - must never die silently in a spawn child
        tb = traceback.format_exc()
        _put_result(result_queue, {"kind": "error", "worker": worker_id,
                                   "traceback": tb}, stop_event)
