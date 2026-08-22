"""Task 7 benchmark: gpu_runtime.py throughput, transfer overhead, VRAM.

Measures, per (mode, batch): wall ms per ``evaluate()`` (pinned staging + H2D +
FP16 forward + CSR gather + D2H + sync), positions/s, the H2D+D2H+sync transfer
overhead as a fraction of wall time (target < 15%), and peak VRAM per mode.

Modes:
    eager   : FP32 forward, no torch.compile (pinned/streams architecture kept)
    amp     : FP16 autocast forward, no torch.compile
    compile : FP16 autocast + torch.compile(mode="reduce-overhead")

CLI::

    .venv/bin/python benchmarks/gpu_runtime.py --batches 1,32,128,256 \\
        --modes eager,amp,compile
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from config import Config  # noqa: E402
from gpu_runtime import BUCKETS, InferenceRuntime  # noqa: E402
from model import ChessNet  # noqa: E402

PLANES, SIZE = 104, 8
POLICY_SIZE = 4672

MODE_KWARGS = {
    "eager": dict(amp=False, compile=False),
    "amp": dict(amp=True, compile=False),
    "compile": dict(amp=True, compile=True),
}


def make_batch(B, seed):
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((B, PLANES, SIZE, SIZE)).astype(np.float32)
    lengths = rng.integers(1, 41, size=B)
    offsets = np.zeros(B + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths).astype(np.int32)
    idx = np.concatenate(
        [np.sort(rng.choice(POLICY_SIZE, int(l), replace=False)) for l in lengths]
    ).astype(np.int32)
    return inputs, offsets, idx


def auto_repeats(B):
    return int(max(20, min(300, 3000 // max(B, 1))))


def bench_mode(mode, cfg, base_state, batches, repeats_override, seed):
    model = ChessNet(cfg).to("cuda").eval()
    model.load_state_dict(base_state)
    if mode == "eager":
        rt = InferenceRuntime(cfg=cfg, model=model, amp=False, compile=False)
    elif mode == "amp":
        rt = InferenceRuntime(cfg=cfg, model=model, amp=True, compile=False)
    else:  # compile
        rt = InferenceRuntime(cfg=cfg, model=model, amp=True, compile=True)
    torch.cuda.reset_peak_memory_stats()
    mode_label = mode if (mode != "compile" or rt.compiled) else "compile(fallback)"
    rows = []
    for B in batches:
        inputs, offsets, indices = make_batch(B, seed=seed)
        repeats = repeats_override or auto_repeats(B)
        call = rt.prepare(inputs, offsets, indices)  # staged device buffers

        # warmup (compiles graphs / picks cudnn algos)
        for _ in range(3):
            rt.evaluate(inputs, offsets, indices)
        torch.cuda.synchronize()

        # wall time: full pipeline (pin -> H2D -> fwd+gather -> D2H -> sync)
        t0 = time.perf_counter()
        for _ in range(repeats):
            rt.evaluate(inputs, offsets, indices)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) / repeats * 1000.0

        # compute-only: device-resident forward + gather, no transfers
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            rt.forward_device(call)
        torch.cuda.synchronize()
        compute_ms = (time.perf_counter() - t0) / repeats * 1000.0

        transfer_frac = max(0.0, 1.0 - compute_ms / wall_ms)
        pos_s = B * 1000.0 / wall_ms
        rows.append((B, wall_ms, pos_s, transfer_frac))
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    rt.close()
    return mode_label, rows, peak_mb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batches",
        default=",".join(str(b) for b in BUCKETS),
        help="comma-separated batch sizes (default all buckets)",
    )
    ap.add_argument(
        "--modes",
        default=",".join(MODE_KWARGS),
        help="comma-separated modes: eager,amp,compile (default all)",
    )
    ap.add_argument("--repeats", type=int, default=0, help="forwards per cell (0=auto)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for this benchmark")
        sys.exit(1)

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODE_KWARGS:
            ap.error(f"unknown mode {m!r}; choose from {sorted(MODE_KWARGS)}")

    cfg = Config()  # default v2-6x128 body
    base = ChessNet(cfg).to("cuda").eval()
    n_params = sum(p.numel() for p in base.parameters())
    gpu = torch.cuda.get_device_name(0)

    print(f"== gpu_runtime benchmark == {gpu} | torch {torch.__version__}")
    print(f"net: {cfg.architecture_id} ({n_params:,} params) | "
          f"buckets {BUCKETS} | amp FP16 body, FP32 value/gather")
    print()

    header = f"{'mode':<18}{'batch':>7}{'ms/fwd':>10}{'pos/s':>12}{'transfer%':>11}  <15%"
    print(header)
    print("-" * len(header))

    transfer_cells = 0
    transfer_ok = 0
    peaks = {}
    for mode in modes:
        label, rows, peak_mb = bench_mode(
            mode, cfg, base.state_dict(), batches, args.repeats, args.seed
        )
        peaks[mode] = peak_mb
        for B, wall_ms, pos_s, tf in rows:
            ok = tf < 0.15
            transfer_cells += 1
            transfer_ok += int(ok)
            print(
                f"{label:<18}{B:>7}{wall_ms:>10.3f}{pos_s:>12.1f}"
                f"{100.0 * tf:>10.1f}%   {'PASS' if ok else 'FAIL'}"
            )
        print()

    print("peak VRAM (max_memory_allocated):")
    for mode in modes:
        print(f"  {mode:<18}{peaks[mode]:>8.1f} MB")
    print()
    print(
        f"transfer-overhead target (<15%): {transfer_ok}/{transfer_cells} cells PASS"
    )
    if transfer_ok == transfer_cells:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL (see cells above)")
        sys.exit(2)


if __name__ == "__main__":
    main()
