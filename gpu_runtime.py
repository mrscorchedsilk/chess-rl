"""Task 7: persistent GPU inference runtime for native-MCTS sparse leaf batches.

Consumes the sparse leaf batches produced by ``chess_rl_native.MCTS.gather_leaves``
-- ``inputs[B,104,8,8] f32``, ``legal_offsets[B+1] i32`` (CSR row pointers),
``legal_indices[K] i32`` (sorted per row) -- and returns ONLY the legal-action
logits ``legal_logits[K]`` (in CSR order) plus side-to-move values ``values[B,1]``.
The dense 4672-wide policy rows NEVER leave the GPU; the legal logits are
gathered on-device with a CSR row-id/offset gather.

Pinned contract (module level)::

    from gpu_runtime import evaluate
    legal_logits, values = evaluate(inputs, legal_offsets, legal_indices)

Architecture of the runtime (``InferenceRuntime``):

1. A ``ChessNet`` resident on GPU in ``channels_last`` memory format, eval mode,
   FP32 master weights, with all gradients disabled.
2. Fixed batch buckets (32/64/128/256): per-bucket preallocated pinned host
   buffers (inputs, CSR offsets, CSR indices, output logits, values) and device
   buffers.  Padded rows are explicitly masked: their CSR ranges are empty
   (offset[b] == offset[b+1]) and their input planes are zeroed, so padded rows
   can never leak into the returned tensors.
3. FP16 autocast for the body + policy head; the value head and the CSR gather
   run in FP32.  Under amp, inputs are staged/host-copied as FP16 (halving H2D
   bus bytes; the network casts to FP16 anyway); eager mode stages FP32.
4. Non-blocking H2D/D2H on dedicated copy streams, fenced by CUDA events; the
   CPU only reads host results after the D2H-completion event synchronizes.
5. GPU-side legal-logit gather from the dense policy rows using the CSR offsets
   (``repeat_interleave`` row ids + ``index_select`` over the flattened rows) --
   the dense rows are never copied back to the host.
6. ``channels_last`` model plus ``torch.compile(mode="reduce-overhead")`` with a
   clean eager fallback if compilation fails.
7. Determinism: ``torch.backends.cudnn.deterministic`` is enabled for the
   runtime's lifetime (restored on ``close()``).  ``amp``/``eager`` modes are
   bitwise deterministic for repeated identical inputs; ``compile`` mode replays
   a captured CUDA graph, which is deterministic for repeated identical inputs
   on a given process but may pick different kernels across processes.

The runtime is NOT thread-safe: callers must serialize access.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from model import ChessNet

__all__ = [
    "BUCKETS",
    "MAX_LEGAL_PER_ROW",
    "InferenceRuntime",
    "evaluate",
    "get_default_runtime",
    "set_default_runtime",
]

# Fixed batch buckets: a call with B leaves is padded to the smallest bucket
# that fits.  B > max(BUCKETS) is rejected.
BUCKETS = (32, 64, 128, 256)
# Chess has at most 218 legal moves; 256 per row gives headroom and bounds the
# CSR index buffers at 256 * B.
MAX_LEGAL_PER_ROW = 256


@dataclass
class _Call:
    """One staged inference call: device-resident inputs + outcome tensors."""

    bucket: int
    B: int
    K: int
    dev_in: torch.Tensor
    dev_off: torch.Tensor
    dev_idx: torch.Tensor
    legal: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    ev_fwd: Optional["torch.cuda.Event"] = None


class InferenceRuntime:
    """Persistent GPU inference service for native-MCTS leaf batches.

    Parameters
    ----------
    cfg : Config, optional
        Architecture config for ``ChessNet``.  Only used when ``model`` is not
        given (a fresh, randomly initialised net is constructed).
    model : ChessNet, optional
        Pre-built net to serve (weights are used as-is; the runtime moves it to
        the device in eval mode, ``channels_last``, no-grad).  Defaults to a
        fresh ``ChessNet(cfg)``.
    device : str | torch.device, optional
        CUDA device (default: current CUDA device).
    amp : bool
        FP16 autocast for body + policy head (default True).  The value head and
        the CSR gather always run in FP32.
    compile : bool
        ``torch.compile(mode="reduce-overhead")`` with eager fallback
        (default True).
    buckets : tuple[int, ...]
        Fixed batch buckets (default ``(32, 64, 128, 256)``).
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        model: Optional[ChessNet] = None,
        device: Optional[str] = None,
        amp: bool = True,
        compile: bool = True,
        buckets: Tuple[int, ...] = BUCKETS,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("InferenceRuntime requires CUDA; none available")
        self.cfg = cfg or Config()
        self.device = torch.device(device or "cuda")
        self.amp = bool(amp)
        self.buckets = tuple(sorted(int(b) for b in buckets))
        self.policy_size = int(self.cfg.policy_size)
        self.num_planes = int(self.cfg.num_input_planes)
        self.board_size = int(self.cfg.board_size)

        # ---- model residency: FP32 master weights, eval, channels_last ---- #
        if model is None:
            model = ChessNet(self.cfg)
        self.model = model.to(self.device).eval()
        self.model.to(memory_format=torch.channels_last)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for b in self.model.buffers():
            b.requires_grad_(False)

        # ---- deterministic convs for the runtime's lifetime (restored in
        #      close()) ---- #
        self._prev_cudnn_deterministic = torch.backends.cudnn.deterministic
        self._prev_cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # ---- forward: torch.compile with a clean eager fallback ---- #
        self._compiled = None
        self.compiled = False
        if compile:
            try:
                self._compiled = torch.compile(
                    self._forward_impl, mode="reduce-overhead"
                )
                self.compiled = True
            except Exception as exc:  # pragma: no cover - environment dependent
                warnings.warn(
                    f"torch.compile unavailable ({exc!r}); falling back to "
                    "eager forward"
                )
        if self._compiled is None:
            self._compiled = self._forward_impl

        # ---- dedicated copy streams (H2D in, D2H out) ---- #
        self.stream_h2d = torch.cuda.Stream(device=self.device)
        self.stream_d2h = torch.cuda.Stream(device=self.device)

        # ---- telemetry counters (Ticket A; per-instance, semantic-free) ----
        # ``call_count`` / ``batch_b`` / ``total_forward_s`` are purely
        # additive counters around ``evaluate`` — no tensor/RNG/order change.
        # ``stats()`` computes the batch distribution from the deque at read
        # time (cheap at 10k elements).
        self.call_count = 0
        self.batch_b: deque = deque(maxlen=10_000)
        self.total_forward_s = 0.0

        # ---- per-bucket preallocated pinned host + device buffers ---- #
        self._pinned: dict[int, dict[str, torch.Tensor]] = {}
        self._dev: dict[int, dict[str, torch.Tensor]] = {}
        self._events: dict[int, dict[str, "torch.cuda.Event"]] = {}
        # Under amp the network runs in FP16 anyway, so stage inputs as FP16
        # and halve the H2D bus bytes (autocast was casting on-device before).
        self.staging_dtype = torch.float16 if self.amp else torch.float32
        for b in self.buckets:
            cap = b * MAX_LEGAL_PER_ROW
            in_shape = (b, self.num_planes, self.board_size, self.board_size)
            self._pinned[b] = {
                "in": torch.empty(
                    in_shape, dtype=self.staging_dtype, pin_memory=True
                ),
                "off": torch.empty((b + 1,), dtype=torch.int32, pin_memory=True),
                "idx": torch.empty((cap,), dtype=torch.int32, pin_memory=True),
                "logits": torch.empty((cap,), dtype=torch.float32, pin_memory=True),
                "val": torch.empty((b, 1), dtype=torch.float32, pin_memory=True),
            }
            self._dev[b] = {
                "in": torch.empty(
                    in_shape,
                    dtype=self.staging_dtype,
                    device=self.device,
                    memory_format=torch.channels_last,
                ),
                "off": torch.empty((b + 1,), dtype=torch.int32, device=self.device),
                "idx": torch.empty((cap,), dtype=torch.int32, device=self.device),
            }
            self._events[b] = {
                "h2d": torch.cuda.Event(),
                "fwd": torch.cuda.Event(),
                "d2h": torch.cuda.Event(),
            }

    # ------------------------------------------------------------------ #
    #  forward                                                            #
    # ------------------------------------------------------------------ #
    def _forward_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Body + policy head under FP16 autocast; value head in FP32.

        Matches ``ChessNet.forward`` exactly, except the value head is computed
        in FP32 even under autocast (per the Task 7 contract).
        """
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.amp
        ):
            h = self.model.body(x)
            p = self.model.policy_conv(h)
        # policy head: NHWC flatten -> flat index = from_square * 73 + plane
        p = p.permute(0, 2, 3, 1).contiguous().view(p.size(0), -1).float()
        # value head, FP32 (reshape: value_conv output may be channels_last)
        v = F.relu(self.model.value_conv(h.float()))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.model.value_fc1(v))
        v = torch.tanh(self.model.value_fc2(v))
        return p, v

    # ------------------------------------------------------------------ #
    #  validation + staging                                              #
    # ------------------------------------------------------------------ #
    def _validate(
        self,
        inputs: np.ndarray,
        offsets: np.ndarray,
        indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)
        if inputs.dtype != np.float32:
            raise ValueError(
                f"inputs must be float32, got {inputs.dtype}; the native "
                "gather_leaves contract is np.float32[B,104,8,8]"
            )
        if inputs.ndim != 4 or inputs.shape[1:] != (
            self.num_planes,
            self.board_size,
            self.board_size,
        ):
            raise ValueError(
                f"inputs must be [B,{self.num_planes},{self.board_size},"
                f"{self.board_size}], got {inputs.shape}"
            )
        B = int(inputs.shape[0])
        if B == 0:
            raise ValueError("empty leaf batch (B=0)")
        if B > self.buckets[-1]:
            raise ValueError(
                f"B={B} exceeds the largest fixed bucket {self.buckets[-1]}; "
                f"buckets are {self.buckets}"
            )

        off = np.asarray(offsets)
        idx = np.asarray(indices)
        if not np.issubdtype(off.dtype, np.integer) or off.ndim != 1:
            raise ValueError(
                f"legal_offsets must be a 1-D integer array, got dtype="
                f"{off.dtype} ndim={off.ndim}"
            )
        if not np.issubdtype(idx.dtype, np.integer) or idx.ndim != 1:
            raise ValueError(
                f"legal_indices must be a 1-D integer array, got dtype="
                f"{idx.dtype} ndim={idx.ndim}"
            )
        if off.shape[0] != B + 1:
            raise ValueError(
                f"legal_offsets must have length B+1={B + 1}, got "
                f"{off.shape[0]}"
            )
        off = off.astype(np.int32, copy=False)
        idx = idx.astype(np.int32, copy=False)
        K = int(idx.shape[0])
        if int(off[0]) != 0 or int(off[-1]) != K:
            raise ValueError(
                f"CSR boundary mismatch: offsets[0]={off[0]}, "
                f"offsets[-1]={off[-1]} but len(legal_indices)={K}"
            )
        if np.any(off[1:] < off[:-1]):
            raise ValueError("legal_offsets must be non-decreasing (CSR)")
        if np.any((off[1:] - off[:-1]) > MAX_LEGAL_PER_ROW):
            raise ValueError(
                f"a CSR row exceeds MAX_LEGAL_PER_ROW={MAX_LEGAL_PER_ROW} "
                "(chess has at most 218 legal moves)"
            )
        if K > 0 and (int(idx.min()) < 0 or int(idx.max()) >= self.policy_size):
            raise ValueError(
                f"legal_indices must lie in [0, {self.policy_size}), got "
                f"min={idx.min()} max={idx.max()}"
            )
        return inputs, off, idx, B, K

    def prepare(
        self, inputs: np.ndarray, offsets: np.ndarray, indices: np.ndarray
    ) -> _Call:
        """Validate + stage the batch: pinned host copy, then non-blocking H2D.

        The caller's current CUDA stream waits on the H2D-completion event
        before ``prepare`` returns, so the device buffers are safe to consume.
        """
        inputs, off, idx, B, K = self._validate(inputs, offsets, indices)
        bucket = next(b for b in self.buckets if b >= B)
        pin, dev = self._pinned[bucket], self._dev[bucket]

        # ---- stage on pinned host (synchronous CPU copies) ---- #
        staged_in = torch.from_numpy(np.ascontiguousarray(inputs))
        if self.staging_dtype == torch.float16:
            staged_in = staged_in.half()
        pin["in"][:B].copy_(staged_in)
        pin["off"][: B + 1].copy_(torch.from_numpy(off))
        if K:
            pin["idx"][:K].copy_(torch.from_numpy(idx))
        if B < bucket:
            # Explicitly mask padded rows: empty CSR ranges (offset[b]==offset[b+1])
            pin["off"][B + 1 :].fill_(int(off[-1]))

        # ---- non-blocking H2D on the dedicated copy-in stream ---- #
        ev_h2d = self._events[bucket]["h2d"]
        with torch.cuda.stream(self.stream_h2d):
            dev["in"].copy_(pin["in"], non_blocking=True)
            dev["off"].copy_(pin["off"], non_blocking=True)
            if K:
                dev["idx"][:K].copy_(pin["idx"][:K], non_blocking=True)
            if B < bucket:
                dev["in"][B:].zero_()  # masked rows: zero planes (deterministic)
            ev_h2d.record(self.stream_h2d)
        torch.cuda.current_stream().wait_event(ev_h2d)
        return _Call(
            bucket=bucket,
            B=B,
            K=K,
            dev_in=dev["in"],
            dev_off=dev["off"],
            dev_idx=dev["idx"],
        )

    # ------------------------------------------------------------------ #
    #  forward + CSR gather (device side)                                #
    # ------------------------------------------------------------------ #
    def forward_device(self, call: _Call) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compiled forward + GPU-side CSR legal-logit gather, no copies.

        Runs on the caller's current CUDA stream; records an event (stored on
        ``call.ev_fwd``) that the D2H stage must wait on.  Returns
        ``(legal_logits[K] fp32, values[bucket,1] fp32)``.
        """
        cur = torch.cuda.current_stream()
        logits, values = self._compiled(call.dev_in)  # [bucket,4672] [bucket,1]
        K = call.K
        if K == 0:
            legal = torch.empty((0,), dtype=torch.float32, device=self.device)
        else:
            # CSR gather: row_ids = repeat_interleave(arange(bucket), lengths)
            # then index_select over the flattened dense rows.  Padded rows have
            # length 0 and contribute nothing.  Never copies dense rows to CPU.
            lengths = (call.dev_off[1:] - call.dev_off[:-1]).to(torch.int64)
            row_ids = torch.repeat_interleave(
                torch.arange(call.dev_off.shape[0] - 1, device=self.device),
                lengths,
            )
            flat = logits.reshape(-1)
            flat_idx = row_ids * self.policy_size + call.dev_idx[:K].to(
                torch.int64
            )
            legal = torch.index_select(flat, 0, flat_idx)
        ev_fwd = self._events[call.bucket]["fwd"]
        ev_fwd.record(cur)
        call.legal, call.values, call.ev_fwd = legal, values, ev_fwd
        return legal, values

    def copy_back(
        self, call: _Call, legal: torch.Tensor, values: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Non-blocking D2H on the copy-out stream; returns numpy results.

        The CPU blocks only on the D2H-completion event, then slices the pinned
        buffers down to the real rows (padded rows are never returned).
        """
        B, K, bucket = call.B, call.K, call.bucket
        pin = self._pinned[bucket]
        ev_d2h = self._events[bucket]["d2h"]
        with torch.cuda.stream(self.stream_d2h):
            self.stream_d2h.wait_event(call.ev_fwd)
            if K:
                pin["logits"][:K].copy_(legal, non_blocking=True)
            pin["val"][:B].copy_(values[:B], non_blocking=True)
            ev_d2h.record(self.stream_d2h)
        ev_d2h.synchronize()
        logits_np = (
            pin["logits"][:K].numpy().copy()
            if K
            else np.zeros((0,), dtype=np.float32)
        )
        values_np = pin["val"][:B].numpy().copy()
        return logits_np, values_np

    def evaluate(
        self, inputs: np.ndarray, offsets: np.ndarray, indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full pipeline: pin -> H2D -> forward+gather -> D2H -> numpy.

        Returns ``(legal_logits[K] f32, values[B,1] f32)``.
        """
        t0 = time.perf_counter()
        call = self.prepare(inputs, offsets, indices)
        legal, values = self.forward_device(call)
        out = self.copy_back(call, legal, values)
        self.total_forward_s += time.perf_counter() - t0
        self.call_count += 1
        self.batch_b.append(call.B)
        return out

    def stats(self) -> dict:
        """Canonical inference counters (Ticket A, design §3.3).

        Returns ``{"calls", "batch_min", "batch_mean", "batch_p50",
        "batch_p90", "batch_max", "total_forward_s"}`` with the batch
        distribution computed from the ``batch_b`` deque at read time (the
        batch percentiles are ``None`` before the first successful call).
        """
        batch = list(self.batch_b)
        if batch:
            arr = np.asarray(batch, dtype=np.float64)
            return {
                "calls": int(self.call_count),
                "batch_min": float(arr.min()),
                "batch_mean": float(arr.mean()),
                "batch_p50": float(np.percentile(arr, 50)),
                "batch_p90": float(np.percentile(arr, 90)),
                "batch_max": float(arr.max()),
                "total_forward_s": float(self.total_forward_s),
            }
        return {
            "calls": int(self.call_count),
            "batch_min": None,
            "batch_mean": None,
            "batch_p50": None,
            "batch_p90": None,
            "batch_max": None,
            "total_forward_s": float(self.total_forward_s),
        }

    def close(self) -> None:
        """Restore global cudnn determinism flags (buffers stay valid)."""
        torch.backends.cudnn.deterministic = self._prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = self._prev_cudnn_benchmark


# --------------------------------------------------------------------------- #
#  Module-level pinned contract                                                #
# --------------------------------------------------------------------------- #
_DEFAULT_RUNTIME: Optional[InferenceRuntime] = None


def get_default_runtime(cfg: Optional[Config] = None, **kwargs) -> InferenceRuntime:
    """Lazily constructed process-wide runtime (default config)."""
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = InferenceRuntime(cfg=cfg, **kwargs)
    return _DEFAULT_RUNTIME


def set_default_runtime(rt: InferenceRuntime) -> None:
    """Install a runtime for the module-level ``evaluate`` (mainly tests)."""
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = rt


def evaluate(
    inputs: np.ndarray, legal_offsets: np.ndarray, legal_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pinned contract: consume a native-MCTS sparse leaf batch.

    Parameters
    ----------
    inputs : np.float32 [B,104,8,8]
    legal_offsets : np.int32 [B+1]  (CSR row pointers; offsets[0]==0)
    legal_indices : np.int32 [K]    (sorted per row; offsets[-1]==K)

    Returns
    -------
    legal_logits : np.float32 [K]  logit per legal_indices[k], CSR order
    values : np.float32 [B,1]      side-to-move values in [-1, 1]
    """
    return get_default_runtime().evaluate(inputs, legal_offsets, legal_indices)
