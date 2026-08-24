"""Training-throughput benchmark: optimizer steps and sample presentations per
iteration and per hour, for a given architecture and epoch bound.

Measures the LEARNER side only (replay -> gradient steps); self-play throughput
is measured separately by `benchmarks/selfplay_bench.py`.  Reports:

  * steps/iteration        - what `train_epoch_size` actually buys
  * ms/step, samples/s     - raw GPU training throughput
  * peak VRAM              - fit check against the 11 GB budget
  * steps/hour             - steps/iteration scaled by a measured or supplied
                             seconds-per-iteration figure

Run:
  .venv/bin/python benchmarks/train_bench.py --arch v2-6x128 --epoch-size 768
  .venv/bin/python benchmarks/train_bench.py --arch v4-20x256 --epoch-size 8192
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import ARCHITECTURES, Config   # noqa: E402
from model import ChessNet                 # noqa: E402
from replay import ReplayBuffer            # noqa: E402
import train                               # noqa: E402


def build_cfg(arch: str, epoch_size: int, batch: int, epochs: int,
              replay: int, device: str) -> Config:
    cfg = Config()
    if arch not in ARCHITECTURES:
        raise SystemExit(f"unknown architecture {arch!r}; "
                         f"known: {sorted(ARCHITECTURES)}")
    cfg.architecture_id = arch
    cfg.num_res_blocks, cfg.num_filters = ARCHITECTURES[arch]
    cfg.train_epoch_size = epoch_size
    cfg.train_batch_size = batch
    cfg.training_epochs = epochs
    cfg.epochs_per_iteration = epochs
    cfg.replay_buffer_size = replay
    cfg.device = device
    cfg.amp = device.startswith("cuda")
    cfg.telemetry_enabled = False
    return cfg


def fill(cfg: Config, n: int, seed: int = 0) -> ReplayBuffer:
    """Synthetic replay with realistic shapes and sparsity."""
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(cfg.replay_buffer_size, cfg.policy_size,
                       cfg.num_input_planes, cfg.board_size)
    chunk = []
    for i in range(n):
        state = (rng.random((cfg.num_input_planes, cfg.board_size,
                             cfg.board_size)) < 0.2).astype(np.float32)
        k = int(rng.integers(4, 40))
        idx = rng.choice(cfg.policy_size, size=k, replace=False)
        p = rng.random(k)
        pi = np.zeros(cfg.policy_size, dtype=np.float32)
        pi[idx] = p / p.sum()
        chunk.append((state, pi, float(rng.choice([-1.0, 0.0, 1.0]))))
        if len(chunk) == 2048:
            buf.extend(chunk); chunk = []
    if chunk:
        buf.extend(chunk)
    return buf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="v2-6x128")
    ap.add_argument("--epoch-size", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--replay", type=int, default=50_000)
    ap.add_argument("--fill", type=int, default=20_000,
                    help="rows of synthetic replay to generate")
    ap.add_argument("--iters", type=int, default=3,
                    help="timed _epoch_train iterations (median reported)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--sec-per-iter", type=float, default=None,
                    help="measured wall seconds per training iteration, for "
                         "the steps/hour projection")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--json", default=None, help="write results as JSON here")
    args = ap.parse_args()

    cfg = build_cfg(args.arch, args.epoch_size, args.batch, args.epochs,
                    args.replay, args.device)
    if args.no_channels_last:
        cfg.train_channels_last = False
    if args.no_prefetch:
        cfg.train_prefetch = 0
    buf = fill(cfg, min(args.fill, args.replay))
    net = ChessNet(cfg).to(args.device)
    params = sum(p.numel() for p in net.parameters())
    opt = train._new_optimizer(cfg, net)
    scaler = train._new_scaler(cfg, args.device)

    for _ in range(args.warmup):
        train._epoch_train(cfg, net, opt, buf, args.device, scaler,
                           positions_generated=2400)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    durations, out = [], None
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = train._epoch_train(cfg, net, opt, buf, args.device, scaler,
                                 positions_generated=2400)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)

    dur = float(np.median(durations))
    peak = (torch.cuda.max_memory_allocated() / 1e9
            if args.device.startswith("cuda") else None)
    steps = out["steps"]
    res = {
        "arch": args.arch,
        "params": params,
        "device": args.device,
        "train_epoch_size": args.epoch_size,
        "channels_last": bool(getattr(cfg, "train_channels_last", False)),
        "prefetch": int(getattr(cfg, "train_prefetch", 0)),
        "train_batch_size": args.batch,
        "epochs": args.epochs,
        "replay_rows": len(buf),
        "steps_per_iteration": steps,
        "sample_size": out["sample_size"],
        "positions_trained": out["positions_trained"],
        "sample_reuse": out["sample_reuse"],
        "train_phase_s": dur,
        "ms_per_step": 1000.0 * dur / steps if steps else None,
        "samples_per_s": out["positions_trained"] / dur if dur else None,
        "peak_vram_gb": peak,
    }
    if args.sec_per_iter:
        res["sec_per_iteration"] = args.sec_per_iter
        res["steps_per_hour"] = steps * 3600.0 / args.sec_per_iter
        res["train_fraction_of_iteration"] = dur / args.sec_per_iter

    w = max(len(k) for k in res)
    print(f"--- train_bench: {args.arch} epoch_size={args.epoch_size} ---")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:<{w}} : {v:,.4f}")
        else:
            print(f"  {k:<{w}} : {v:,}" if isinstance(v, int) else f"  {k:<{w}} : {v}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
