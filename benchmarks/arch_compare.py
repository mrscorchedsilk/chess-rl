"""Architecture comparison: what a bigger body actually costs.

For each architecture, on the SAME pipeline and settings:

  * parameters
  * training throughput   (ms/step, samples/s, peak VRAM at the training batch)
  * inference throughput  (positions/s at the self-play batch size)
  * self-play throughput  (real games/hour through ShardedSelfPlay)
  * GPU occupancy         (in-driver busy fraction and sampled nvidia-smi)

Strength is NOT measured here and cannot be: these are randomly initialised
nets.  Which body is strongest per wall-clock hour is a training experiment,
not a benchmark.

Run:
  .venv/bin/python benchmarks/arch_compare.py --archs v2-6x128,v3-10x192,v4-20x256
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import ARCHITECTURES, Config      # noqa: E402
from model import ChessNet                    # noqa: E402
import native_selfplay                        # noqa: E402
from benchmarks.selfplay_bench import GpuSampler   # noqa: E402
from benchmarks.train_bench import fill            # noqa: E402
import train                                   # noqa: E402


def train_throughput(cfg, batch, rows=6000, iters=3):
    buf = fill(cfg, rows)
    net = ChessNet(cfg).to(cfg.device)
    opt = train._new_optimizer(cfg, net)
    scaler = train._new_scaler(cfg, cfg.device)
    for _ in range(1):
        train._epoch_train(cfg, net, opt, buf, cfg.device, scaler,
                           positions_generated=2400)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durs, out = [], None
    for _ in range(iters):
        t0 = time.perf_counter()
        out = train._epoch_train(cfg, net, opt, buf, cfg.device, scaler,
                                 positions_generated=2400)
        torch.cuda.synchronize()
        durs.append(time.perf_counter() - t0)
    dur = float(np.median(durs))
    peak = torch.cuda.max_memory_allocated() / 1e9
    del net, opt, buf
    torch.cuda.empty_cache()
    return {
        "steps_per_iteration": out["steps"],
        "ms_per_step": 1000.0 * dur / out["steps"],
        "train_samples_per_s": out["positions_trained"] / dur,
        "train_peak_vram_gb": peak,
    }


def selfplay_throughput(cfg, games, shards, max_ply):
    cfg.max_game_length = max_ply
    fn = native_selfplay.make_gpu_inference_fn(cfg)
    warm = native_selfplay.ShardedSelfPlay(cfg, fn, games=4, shards=1, seed=1)
    warm.run()
    sp = native_selfplay.ShardedSelfPlay(cfg, fn, games=games, shards=shards,
                                         seed=20260824)
    with GpuSampler(interval=0.2) as sampler:
        t0 = time.perf_counter()
        sp.run()
        wall = time.perf_counter() - t0
    out = {
        "selfplay_wall_s": wall,
        "games_per_hour": games * 3600.0 / wall,
        "gpu_busy_fraction": sp.gpu_busy_fraction,
        "selfplay_batch_mean": sp.batch_stats.get("batch_mean"),
        "inference_calls": sp.inference_calls,
        "positions_per_s": (sp.batch_stats.get("batch_mean") or 0)
        * sp.inference_calls / wall,
    }
    out.update(sampler.summary())
    try:
        fn.close()
    except Exception:  # noqa: BLE001
        pass
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", default="v2-6x128,v3-10x192,v4-20x256")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--max-ply", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epoch-size", type=int, default=8192)
    ap.add_argument("--json", default="benchmarks/results/arch-compare.json")
    args = ap.parse_args()

    rows = []
    for arch in args.archs.split(","):
        arch = arch.strip()
        cfg = Config()
        cfg.architecture_id = arch
        cfg.num_res_blocks, cfg.num_filters = ARCHITECTURES[arch]
        cfg.train_batch_size = args.batch
        cfg.train_epoch_size = args.epoch_size
        cfg.telemetry_enabled = False
        cfg.device = "cuda"
        params = ChessNet(cfg).parameter_count()

        row = {"arch": arch, "params": params}
        row.update(train_throughput(cfg, args.batch))
        row.update(selfplay_throughput(cfg, args.games, args.shards,
                                       args.max_ply))
        rows.append(row)
        print(f"{arch:12s} params={params/1e6:6.2f}M  "
              f"train {row['ms_per_step']:6.2f} ms/step "
              f"({row['train_samples_per_s']:8.0f} samp/s, "
              f"{row['train_peak_vram_gb']:.2f} GB)  "
              f"selfplay {row['games_per_hour']:7.1f} games/h  "
              f"gpu_busy {100*row['gpu_busy_fraction']:5.1f}%  "
              f"nvsmi {row['gpu_util_mean'] or 0:5.1f}%  "
              f"{row['gpu_power_mean_w'] or 0:5.0f}W", flush=True)

    if rows:
        base = rows[0]
        print(f"\nrelative to {base['arch']}:")
        for r in rows[1:]:
            print(f"  {r['arch']:12s} {r['params']/base['params']:5.2f}x params, "
                  f"{r['games_per_hour']/base['games_per_hour']:5.2f}x games/hour, "
                  f"{r['train_samples_per_s']/base['train_samples_per_s']:5.2f}x train samples/s, "
                  f"gpu_busy {100*base['gpu_busy_fraction']:.1f}% -> "
                  f"{100*r['gpu_busy_fraction']:.1f}%")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"games": args.games, "shards": args.shards,
                       "max_ply": args.max_ply, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
