"""Permanent phase & resource telemetry for the native training loop (Ticket A).

This is the swallow-guarded, semantic-free telemetry layer specified in
``docs/telemetry-design.md`` §4.  Two hard guarantees:

* **Telemetry must never kill training.**  ``emit`` / ``sample_resources`` /
  ``replay_diversity`` / ``game_trajectory_hash`` never raise, and call sites
  use ``safe_emit`` (the ``try/except Exception: pass`` pattern already used by
  ``train._log_metrics``) so even a monkeypatched/broken ``emit`` cannot escape
  into training control flow.
* **Telemetry must not change training or game semantics.**  Everything here is
  ``time.perf_counter()`` deltas, pure BLAKE2 hashing, or read-only psutil /
  torch.cuda snapshots.  There is no extra forward pass, no RNG draw, no data
  mutation, no reseeding and no reordering — a run's replay examples, move
  choices, checkpoints and scores are bit-identical with telemetry on vs off.

Record envelope: every line is one JSON object with a top-level
``"schema": "telemetry/v1"``, a ``"t"`` Unix-epoch timestamp and a ``"type"``
discriminator; ``emit`` injects those (plus ``backend`` / ``run_id`` /
``iteration`` / ``generation`` fallbacks) and appends atomically per call.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

_SCHEMA = "telemetry/v1"
# The diversity record's trajectory-hash sample is bounded (design §2.3).
_TRAJECTORY_SAMPLE_LIMIT = 32


class PhaseTimer:
    """Wall-clock phase timer built on ``time.perf_counter()``.

    ``__exit__`` computes ``duration_s`` and never raises; it deliberately
    returns ``None`` so a body exception propagates unchanged (telemetry must
    not change control flow).
    """

    def __init__(self, name: str):
        self.name = name
        self._start = None
        self.duration_s = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._start is not None:
                self.duration_s = time.perf_counter() - self._start
        except Exception:  # noqa: BLE001 - never raise from a timer
            self.duration_s = 0.0
        return None


def _telemetry_path(cfg):
    """Resolve the JSONL output path; ``None`` when the cfg cannot name one
    (so degenerate cfgs never write into the process CWD)."""
    path = getattr(cfg, "telemetry_path", None)
    if path:
        return str(path)
    checkpoint_dir = getattr(cfg, "checkpoint_dir", None)
    if not checkpoint_dir:
        return None
    return os.path.join(str(checkpoint_dir), "telemetry.jsonl")


def emit(cfg, record: dict) -> None:
    """Append one JSON line to ``cfg.telemetry_path``.  NEVER raises.

    The file handle is opened/closed per call (append mode), so a full-disk or
    permission failure is caught and training continues; no module-level
    buffering state exists to corrupt.
    """
    try:
        if not getattr(cfg, "telemetry_enabled", True):
            return
        path = _telemetry_path(cfg)
        if not path:
            return
        rec = {
            "schema": _SCHEMA,
            "t": time.time(),
            "run_id": getattr(cfg, "run_id", None),
            "iteration": getattr(cfg, "iteration", None),
            "generation": getattr(cfg, "generation", None),
            "backend": getattr(cfg, "backend", "python"),
        }
        rec.update(record)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 - observability must never kill training
        pass


def safe_emit(cfg, record: dict) -> None:
    """Call-site swallow-guard around ``emit`` (the exact
    ``try/except Exception: pass`` pattern mandated by the design at every
    emit/sample site): even a broken or monkeypatched ``emit`` cannot escape
    into training control flow."""
    try:
        emit(cfg, record)
    except Exception:  # noqa: BLE001
        pass


def _percentiles(values):
    """Batch distribution ``min/mean/p50/p90/max``; all ``None`` when empty."""
    vals = [float(v) for v in values]
    if not vals:
        return {
            "batch_min": None,
            "batch_mean": None,
            "batch_p50": None,
            "batch_p90": None,
            "batch_max": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "batch_min": float(arr.min()),
        "batch_mean": float(arr.mean()),
        "batch_p50": float(np.percentile(arr, 50)),
        "batch_p90": float(np.percentile(arr, 90)),
        "batch_max": float(arr.max()),
    }


def sample_resources() -> dict:
    """GPU / CPU / RAM / swap snapshot; every field degrades to ``None``.

    VRAM/GPU come from ``torch.cuda`` (``mem_get_info``, ``utilization``,
    ``memory_allocated``, ``memory_reserved``); CPU/RAM/swap come from
    ``psutil``.  Both are imported lazily and every field degrades to ``None``
    (or is omitted) if the source is unavailable — NEVER raises.
    """
    rec = {
        "cpu_percent": None,
        "cpu_count": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "ram_percent": None,
        "swap_used_mb": None,
        "swap_total_mb": None,
        "gpu_util_percent": None,
        "vram_used_mb": None,
        "vram_total_mb": None,
        "torch_alloc_mb": None,
        "torch_reserved_mb": None,
    }
    try:
        import psutil

        rec["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        cores = psutil.cpu_count(logical=True)
        rec["cpu_count"] = int(cores) if cores else None
        vm = psutil.virtual_memory()
        rec["ram_used_mb"] = float(vm.used) / (1024.0 * 1024.0)
        rec["ram_total_mb"] = float(vm.total) / (1024.0 * 1024.0)
        rec["ram_percent"] = float(vm.percent)
        sw = psutil.swap_memory()
        rec["swap_used_mb"] = float(sw.used) / (1024.0 * 1024.0)
        rec["swap_total_mb"] = float(sw.total) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 - resource sampling must never raise
        pass
    try:
        import torch

        if torch.cuda.is_available():
            rec["gpu_util_percent"] = float(torch.cuda.utilization())
            free, total = torch.cuda.mem_get_info()
            rec["vram_used_mb"] = float(total - free) / (1024.0 * 1024.0)
            rec["vram_total_mb"] = float(total) / (1024.0 * 1024.0)
            rec["torch_alloc_mb"] = (
                float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            )
            rec["torch_reserved_mb"] = (
                float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
            )
    except Exception:  # noqa: BLE001 - resource sampling must never raise
        pass
    return rec


def _digest(b):
    # ``b`` may be any bytes-like object (numpy arrays implement the buffer
    # protocol), so hashing is copy-free yet byte-identical to
    # ``audit_replay._digest(b.tobytes())``.
    return hashlib.blake2b(b, digest_size=16).digest()


def _example_digest(positions, i, offsets, legal, probs, zs):
    """Stable digest of example i's (state, policy, z) content — byte-for-byte
    the same digest as ``scripts/audit_replay._example_digest``."""
    a, b = int(offsets[i]), int(offsets[i + 1])
    h = hashlib.blake2b(digest_size=16)
    h.update(positions[i])
    h.update(legal[a:b])
    h.update(probs[a:b])
    h.update(zs[i])
    return h.digest()


def replay_diversity(buffer) -> dict:
    """Port of ``scripts/audit_replay.audit_replay`` hashing onto the LIVE
    replay buffer (no ``torch.load`` round-trip).

    Uses the same BLAKE2 (digest_size=16) digests — packed state + state extras
    + sparse policy indices/probs + z — and the same inferred game-start
    detection (all-zero future-history planes at ply 0), so the ``unique_*``
    counts match ``audit_replay`` on the same contents.  NEVER raises.

    Overhead: the packed-state layout is plane-major (one byte per board row of
    8 bits), so "all future-history planes zero" is checked with one vectorised
    ``count_nonzero`` over the packed future byte range instead of per-example
    ``unpackbits`` — the SAME zero-detection, ~2x cheaper, keeping the audit
    within its 0.5 s budget at 50k examples.
    """
    try:
        sd = buffer.state_dict()
        positions = np.asarray(sd["positions"], dtype=np.uint8)
        zs = np.asarray(sd["z"], dtype=np.float32)
        offsets = np.asarray(sd["offsets"], dtype=np.int64)
        legal = np.asarray(sd["legal_idx"], dtype=np.int32)
        probs = np.asarray(sd["probs"], dtype=np.float32)
        extra_idx = np.asarray(sd["state_extra_idx"], dtype=np.int32)
        extra_values = np.asarray(sd["state_extra_values"], dtype=np.float32)
        extra_offsets = np.asarray(sd["state_extra_offsets"], dtype=np.int64)
        num_input_planes = int(sd.get("num_input_planes", 104))
        board_size = int(sd.get("board_size", 8))
        n = int(positions.shape[0])
    except Exception:  # noqa: BLE001
        return _empty_diversity()

    state_hashes = set()
    exact_hashes = set()

    # Future-history planes (steps 1..history_steps-1) are planes
    # [12 : num_input_planes - 8]; all-zero ONLY at ply 0 (game start).
    # Packed layout: plane p occupies bytes [p*bpp : (p+1)*bpp], bpp =
    # board_size*board_size//8 (one byte per board row of 8 packed bits) —
    # identical to audit_replay's unpackbits zero-detection.
    bpp = max(1, board_size * board_size // 8)
    start_byte = 12 * bpp
    end_byte = (num_input_planes - 8) * bpp
    if n:
        zero_future = (
            np.count_nonzero(positions[:, start_byte:end_byte], axis=1) == 0
        )
        game_starts = np.flatnonzero(zero_future).tolist()
    else:
        game_starts = []

    for i in range(n):
        state_hashes.add(_digest(positions[i]))

        a, b = int(offsets[i]), int(offsets[i + 1])
        ea, eb = int(extra_offsets[i]), int(extra_offsets[i + 1])
        h = hashlib.blake2b(digest_size=16)
        h.update(positions[i])
        h.update(extra_idx[ea:eb])
        h.update(extra_values[ea:eb])
        h.update(legal[a:b])
        h.update(probs[a:b])
        h.update(zs[i])
        exact_hashes.add(h.digest())

    # Trajectories: one game runs from a detected start to the next start (or
    # end of buffer), exactly like the audit tool.
    traj_counts = {}
    boundaries = game_starts + [n]
    for gi in range(len(game_starts)):
        s = game_starts[gi]
        e = boundaries[gi + 1]
        th = hashlib.blake2b(digest_size=16)
        for i in range(s, e):
            th.update(_example_digest(positions, i, offsets, legal, probs, zs))
        d = th.digest()
        traj_counts[d] = traj_counts.get(d, 0) + 1

    return {
        "source": "replay_buffer",
        "replay_size": n,
        "unique_packed_states": len(state_hashes),
        "unique_exact_examples": len(exact_hashes),
        "unique_state_fraction": (len(state_hashes) / n) if n else 0.0,
        "unique_example_fraction": (len(exact_hashes) / n) if n else 0.0,
        "unique_trajectory_hashes": len(traj_counts),
        "most_repeated_trajectory_count": (
            max(traj_counts.values()) if traj_counts else 0
        ),
        "trajectory_hashes": [
            d.hex() for d in list(traj_counts.keys())[:_TRAJECTORY_SAMPLE_LIMIT]
        ],
    }


def _empty_diversity() -> dict:
    return {
        "source": "replay_buffer",
        "replay_size": 0,
        "unique_packed_states": 0,
        "unique_exact_examples": 0,
        "unique_state_fraction": 0.0,
        "unique_example_fraction": 0.0,
        "unique_trajectory_hashes": 0,
        "most_repeated_trajectory_count": 0,
        "trajectory_hashes": [],
    }


def game_trajectory_hash(examples) -> str:
    """BLAKE2(digest_size=16) over one finished game's ordered ``(state, pi,
    z)`` examples — the generation-time trajectory identity emitted by
    ``NativeSelfPlay.run`` (design §2.3, source="selfplay_round")."""
    h = hashlib.blake2b(digest_size=16)
    for state, pi, z in examples:
        h.update(np.asarray(state, dtype=np.float32).tobytes())
        h.update(np.asarray(pi, dtype=np.float32).tobytes())
        h.update(np.float32(z).tobytes())
    return h.hexdigest()
