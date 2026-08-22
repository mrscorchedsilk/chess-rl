"""Compressed replay buffer for the AlphaZero-style chess trainer (Sprint A).

Each training example ``(state, pi, z)`` is stored compactly:

* ``state``  (C, H, W) float32 planes -> binary ``packbits`` plus sparse
  exact float32 overrides for non-binary rule-state cells (for example the
  normalized halfmove-clock plane). This preserves state values losslessly
  while keeping the overwhelmingly-binary history planes compact.
* ``pi``     (P,) float32 dense policy  -> stored sparsely as legal-action
  indices + probabilities (only the nonzero entries; self-play policies are
  zero outside the legal moves, so the round-trip is exact).
* ``z``      float scalar.

Storage is a ring buffer (``collections.deque`` with ``maxlen=capacity``):
oldest examples are evicted first, exactly like the old ``deque``-backed
buffer.

``state_dict()`` / ``load_state_dict()`` give a DETERMINISTIC round-trip:
positions are a stacked uint8 array, sparse policies are flat concatenated
arrays + an offsets vector, in insertion order.  This is what the trainer
persists inside ``latest.pt`` and what resume restores.

``sample_indices(rows)`` reconstructs a DENSE minibatch ``(states, pis, zs)``
on demand (torch tensors), so training sees the same shapes as before while
memory stays ~10x smaller.
"""

from collections import deque
import threading

import numpy as np
import torch


class ReplayBuffer:
    """Ring-buffer of compressed ``(state, pi, z)`` training examples."""

    def __init__(self, capacity, policy_size, num_input_planes=104, board_size=8):
        self.capacity = int(capacity)
        self.policy_size = int(policy_size)
        self.num_input_planes = int(num_input_planes)
        self.board_size = int(board_size)
        self._plane_bytes = (
            self.num_input_planes * self.board_size * self.board_size // 8
        )
        self._positions = deque(maxlen=self.capacity)  # packed uint8 arrays
        self._state_extra_idx = deque(maxlen=self.capacity)  # non-binary flat indices
        self._state_extra_values = deque(maxlen=self.capacity)  # exact float values
        self._legal = deque(maxlen=self.capacity)      # int32 index arrays
        self._probs = deque(maxlen=self.capacity)      # float32 prob arrays
        self._zs = deque(maxlen=self.capacity)         # floats

    # ------------------------------------------------------------------ API

    def __len__(self):
        return len(self._positions)

    def add(self, state, pi, z):
        """Append one example: (C,H,W) binary planes, (P,) dense policy, z."""
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.num_input_planes, self.board_size, self.board_size):
            raise ValueError(
                f"state shape {state.shape} != "
                f"({self.num_input_planes}, {self.board_size}, {self.board_size})"
            )
        pi = np.asarray(pi, dtype=np.float32).reshape(-1)
        if pi.shape[0] != self.policy_size:
            raise ValueError(
                f"policy length {pi.shape[0]} != policy_size {self.policy_size}"
            )
        flat_state = state.reshape(-1)
        packed = np.packbits((flat_state > 0.5).astype(np.uint8), bitorder="big")
        binary = (flat_state == 0.0) | (flat_state == 1.0)
        extra_idx = np.flatnonzero(~binary).astype(np.int32)
        extra_values = flat_state[extra_idx].astype(np.float32)
        idx = np.nonzero(pi)[0].astype(np.int32)
        probs = pi[idx].astype(np.float32)
        self._positions.append(packed)
        self._state_extra_idx.append(extra_idx)
        self._state_extra_values.append(extra_values)
        self._legal.append(idx)
        self._probs.append(probs)
        self._zs.append(float(z))

    def extend(self, examples):
        """Append a list of ``(state, pi, z)`` tuples (self-play output)."""
        for state, pi, z in examples:
            self.add(state, pi, z)

    def sample(self, batch_size, device=None):
        """Sample (without replacement) uniformly and return a dense batch."""
        n = len(self)
        if n == 0:
            return self.sample_indices([], device)
        size = min(int(batch_size), n)
        rows = np.random.choice(n, size=size, replace=False)
        return self.sample_indices(rows, device)

    def sample_indices(self, rows, device=None):
        """Reconstruct a dense minibatch for the given example row indices.

        Returns ``(states, pis, zs)`` torch tensors of shape
        ``(B, C, H, W)``, ``(B, policy_size)``, ``(B, 1)`` on ``device``.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        b = rows.shape[0]
        if b == 0:
            return (
                torch.zeros(0, self.num_input_planes, self.board_size,
                            self.board_size),
                torch.zeros(0, self.policy_size),
                torch.zeros(0, 1),
            )
        packed = np.stack([self._positions[i] for i in rows])
        states = np.unpackbits(packed, axis=1, bitorder="big").astype(
            np.float32
        ).reshape(b, self.num_input_planes, self.board_size, self.board_size)
        flat_states = states.reshape(b, -1)
        for out_row, source_row in enumerate(rows):
            extra_idx = self._state_extra_idx[int(source_row)]
            if extra_idx.size:
                flat_states[out_row, extra_idx] = self._state_extra_values[int(source_row)]

        counts = np.array([len(self._legal[i]) for i in rows], dtype=np.int64)
        flat_idx = np.concatenate([self._legal[i] for i in rows])
        flat_probs = np.concatenate([self._probs[i] for i in rows])
        pis = np.zeros((b, self.policy_size), dtype=np.float32)
        if flat_idx.size:
            row_ids = np.repeat(np.arange(b), counts)
            pis[row_ids, flat_idx] = flat_probs

        zs = np.array([self._zs[i] for i in rows], dtype=np.float32).reshape(b, 1)

        states_t = torch.from_numpy(states)
        pis_t = torch.from_numpy(pis)
        zs_t = torch.from_numpy(zs)
        if device is not None:
            states_t = states_t.to(device)
            pis_t = pis_t.to(device)
            zs_t = zs_t.to(device)
        return states_t, pis_t, zs_t

    # ------------------------------------------------------ deterministic I/O

    def state_dict(self):
        """Deterministic, torch.save-able snapshot of the buffer contents."""
        n = len(self)
        if n == 0:
            positions = np.zeros((0, self._plane_bytes), dtype=np.uint8)
            legal = np.zeros(0, dtype=np.int32)
            probs = np.zeros(0, dtype=np.float32)
            offsets = np.zeros(1, dtype=np.int64)
            state_extra_idx = np.zeros(0, dtype=np.int32)
            state_extra_values = np.zeros(0, dtype=np.float32)
            state_extra_offsets = np.zeros(1, dtype=np.int64)
            zs = np.zeros(0, dtype=np.float32)
        else:
            positions = np.stack(list(self._positions))
            counts = np.array([len(li) for li in self._legal], dtype=np.int64)
            offsets = np.zeros(n + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(counts)
            legal = np.concatenate(list(self._legal))
            probs = np.concatenate(list(self._probs))
            extra_counts = np.array(
                [len(idx) for idx in self._state_extra_idx], dtype=np.int64
            )
            state_extra_offsets = np.zeros(n + 1, dtype=np.int64)
            state_extra_offsets[1:] = np.cumsum(extra_counts)
            state_extra_idx = np.concatenate(list(self._state_extra_idx))
            state_extra_values = np.concatenate(list(self._state_extra_values))
            zs = np.array(list(self._zs), dtype=np.float32)
        return {
            "capacity": self.capacity,
            "policy_size": self.policy_size,
            "num_input_planes": self.num_input_planes,
            "board_size": self.board_size,
            "positions": positions,
            "state_extra_idx": state_extra_idx,
            "state_extra_values": state_extra_values,
            "state_extra_offsets": state_extra_offsets,
            "legal_idx": legal,
            "probs": probs,
            "offsets": offsets,
            "z": zs,
        }

    def load_state_dict(self, sd):
        """Restore from a state_dict produced by ``state_dict()``."""
        required = ("capacity", "policy_size", "num_input_planes", "board_size",
                    "positions", "state_extra_idx", "state_extra_values",
                    "state_extra_offsets", "legal_idx", "probs", "offsets", "z")
        missing = [k for k in required if k not in sd]
        if missing:
            raise ValueError(f"invalid replay state_dict: missing {missing}")
        self.capacity = int(sd["capacity"])
        self.policy_size = int(sd["policy_size"])
        self.num_input_planes = int(sd["num_input_planes"])
        self.board_size = int(sd["board_size"])
        self._plane_bytes = (
            self.num_input_planes * self.board_size * self.board_size // 8
        )
        self._positions = deque(maxlen=self.capacity)
        self._state_extra_idx = deque(maxlen=self.capacity)
        self._state_extra_values = deque(maxlen=self.capacity)
        self._legal = deque(maxlen=self.capacity)
        self._probs = deque(maxlen=self.capacity)
        self._zs = deque(maxlen=self.capacity)
        positions = np.asarray(sd["positions"], dtype=np.uint8)
        state_extra_idx = np.asarray(sd["state_extra_idx"], dtype=np.int32)
        state_extra_values = np.asarray(sd["state_extra_values"], dtype=np.float32)
        state_extra_offsets = np.asarray(sd["state_extra_offsets"], dtype=np.int64)
        legal = np.asarray(sd["legal_idx"], dtype=np.int32)
        probs = np.asarray(sd["probs"], dtype=np.float32)
        offsets = np.asarray(sd["offsets"], dtype=np.int64)
        zs = np.asarray(sd["z"], dtype=np.float32)
        n = positions.shape[0]
        if offsets.shape[0] != n + 1:
            raise ValueError("invalid replay state_dict: offsets/positions mismatch")
        if state_extra_offsets.shape[0] != n + 1:
            raise ValueError("invalid replay state_dict: state extras/positions mismatch")
        for i in range(n):
            a, b = int(offsets[i]), int(offsets[i + 1])
            ea, eb = int(state_extra_offsets[i]), int(state_extra_offsets[i + 1])
            self._positions.append(positions[i])
            self._state_extra_idx.append(state_extra_idx[ea:eb])
            self._state_extra_values.append(state_extra_values[ea:eb])
            self._legal.append(legal[a:b])
            self._probs.append(probs[a:b])
            self._zs.append(float(zs[i]))


# --------------------------------------------------------------------------- #
#  Pinned minibatch staging (Task 8)                                          #
# --------------------------------------------------------------------------- #
# The replay buffer stays COMPRESSED in RAM.  PinnedReplayLoader stages dense
# minibatches into reusable page-locked host buffers with one async prefetcher
# thread, so the expensive decompression of the NEXT minibatch overlaps the
# optimizer step of the CURRENT one, and the H2D transfer is a non-blocking
# copy.  Row selection is always the CALLER's job (np.random.choice in
# train._epoch_train), so the loader never perturbs the training RNG sequence
# and resume determinism is preserved.

class PinnedReplayLoader:
    """Stage dense minibatches from a ReplayBuffer with pinned buffers +
    async prefetch of the next minibatch (replay stays in RAM, not VRAM).

    ``batches(rows)`` yields ``(states, pis, zs)`` on ``device`` for
    consecutive ``batch_size`` slices of ``rows``.  On CUDA a background
    thread decompresses slice i+1 into a reusable pinned CPU buffer and
    issues a non-blocking H2D copy while the caller trains on slice i; a
    CUDA event guarantees a buffer is never overwritten before its previous
    copy completed.  On CPU (or without CUDA) it is a synchronous passthrough
    identical to ``buffer.sample_indices(rows, device)``.

    The pinned buffers are allocated lazily and REUSED across ``batches()``
    calls (bounded pool of ``max(2, num_prefetch + 1)`` sets), so a long
    training run does not allocate per-minibatch host memory.
    """

    def __init__(self, buffer, batch_size, device, num_prefetch=1):
        self.buffer = buffer
        self.batch_size = int(batch_size)
        self.device = device
        self.num_prefetch = max(1, int(num_prefetch))
        device_type = str(device).split(":")[0]
        self._async = bool(device_type == "cuda" and torch.cuda.is_available())
        self._num_buffers = max(2, self.num_prefetch + 1)
        self._pin_sets = []          # reusable (states, pis, zs) pinned tensors
        self._pin_events = []        # last CUDA event per pin set (copy-done)
        self._free = list(range(self._num_buffers))  # pin sets ready to refill
        self._ready = deque()        # (states, pis, zs) device tensors
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = False
        self._worker_error = None
        self._thread = None
        self._prefetch_calls = 0
        self._batch_count = 0

    # ---- diagnostics (used by tests + benchmarks) ----

    @property
    def pin_sets_allocated(self):
        """Number of pinned buffer sets actually allocated (<= _num_buffers)."""
        return len(self._pin_sets)

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def prefetch_calls(self):
        return self._prefetch_calls

    @property
    def batch_count(self):
        return self._batch_count

    # ---- public iteration ----

    def batches(self, rows):
        """Yield (states, pis, zs) on self.device for slices of ``rows``,
        prefetching the next slice on a background thread (CUDA only)."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        n = rows.shape[0]
        if n == 0:
            return
        slices = [rows[s:s + self.batch_size]
                  for s in range(0, n, self.batch_size)]
        if not self._async:
            for rb in slices:
                self._batch_count += 1
                yield self.buffer.sample_indices(rb, self.device)
            return
        self._start_worker(slices)
        try:
            for _ in range(len(slices)):
                with self._cond:
                    while not self._ready and not self._stop:
                        self._cond.wait()
                    if not self._ready:
                        break
                    dev = self._ready.popleft()
                self._batch_count += 1
                yield dev
        finally:
            self._stop_worker()
        if self._worker_error is not None:
            raise self._worker_error

    # ---- async machinery ----

    def _start_worker(self, slices):
        self._stop = False
        self._worker_error = None
        self._thread = threading.Thread(
            target=self._worker, args=(slices,), daemon=True
        )
        self._thread.start()

    def _stop_worker(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _worker(self, slices):
        try:
            for rb in slices:
                with self._cond:
                    while len(self._ready) >= self.num_prefetch and not self._stop:
                        self._cond.wait()
                    if self._stop:
                        break
                    buf_idx = self._free.pop(0)
                states, pis, zs = self.buffer.sample_indices(rb, "cpu")
                pins = self._stage(buf_idx, states, pis, zs)
                dev_states = pins[0].to(self.device, non_blocking=True)
                dev_pis = pins[1].to(self.device, non_blocking=True)
                dev_zs = pins[2].to(self.device, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                with self._cond:
                    self._ready.append((dev_states, dev_pis, dev_zs))
                    self._free.append(buf_idx)
                    self._pin_events[buf_idx] = ev
                    self._prefetch_calls += 1
                    self._cond.notify_all()
        except Exception as e:  # noqa: BLE001 - surfaced to the consumer
            with self._cond:
                self._worker_error = e
                self._stop = True
                self._cond.notify_all()

    def _stage(self, buf_idx, states, pis, zs):
        """Copy into the reusable pinned buffer set ``buf_idx``; wait for its
        previous H2D copy (CUDA event) so it is safe to overwrite."""
        if buf_idx >= len(self._pin_sets):
            self._pin_sets.append((
                torch.empty_like(states, pin_memory=True),
                torch.empty_like(pis, pin_memory=True),
                torch.empty_like(zs, pin_memory=True),
            ))
            self._pin_events.append(None)
        prev = self._pin_events[buf_idx]
        if prev is not None:
            prev.synchronize()   # previous copy from this buffer has finished
        states_p, pis_p, zs_p = self._pin_sets[buf_idx]
        states_p.copy_(states, non_blocking=False)  # host -> pinned host
        pis_p.copy_(pis, non_blocking=False)
        zs_p.copy_(zs, non_blocking=False)
        return states_p, pis_p, zs_p
