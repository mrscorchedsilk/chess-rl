"""CPU/GPU overlap benchmark: shards vs throughput and GPU occupancy.

Runs a real self-play round through ShardedSelfPlay with the production GPU
InferenceRuntime and reports, per shard count:

  * wall seconds and games/hour for a full round
  * gpu_busy_fraction  - share of the round during which a batch was on the GPU
                         (measured inside the driver, from the GPU lock)
  * sampled GPU utilisation and power from nvidia-smi
  * the phase split summed across shards (gather / infer / apply / advance)

shards=1 is the old serial loop, so the shards=1 row IS the baseline.

Run:
  .venv/bin/python benchmarks/overlap_bench.py --games 48 --shards 1,2,4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import ARCHITECTURES, Config      # noqa: E402
import native_selfplay                        # noqa: E402
from benchmarks.selfplay_bench import GpuSampler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="v2-6x128")
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--shards", default="1,2,4")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--max-game-length", type=int, default=None)
    ap.add_argument("--leaves-per-game", type=int, default=12)
    ap.add_argument("--max-batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg.architecture_id = args.arch
    cfg.num_res_blocks, cfg.num_filters = ARCHITECTURES[args.arch]
    cfg.telemetry_enabled = False
    cfg.selfplay_leaves_per_game = args.leaves_per_game
    cfg.selfplay_max_batch = args.max_batch
    if args.sims is not None:
        cfg.num_simulations = args.sims
    if args.max_game_length is not None:
        cfg.max_game_length = args.max_game_length

    inference_fn = native_selfplay.make_gpu_inference_fn(cfg)

    # Warm the compiled graphs and bucket buffers so the first timed row is
    # not paying for them.
    warm = native_selfplay.ShardedSelfPlay(
        cfg, inference_fn, games=4, shards=1, seed=args.seed - 1)
    warm.cfg.num_simulations = cfg.num_simulations
    warm.max_batch = args.max_batch
    for _ in range(1):
        w = native_selfplay.ShardedSelfPlay(
            cfg, inference_fn, games=2, shards=1, seed=args.seed - 2)
        w.cfg = cfg
        w.run()

    rows = []
    for shards in [int(x) for x in args.shards.split(",")]:
        sp = native_selfplay.ShardedSelfPlay(
            cfg, inference_fn, games=args.games, shards=shards, seed=args.seed)
        with GpuSampler(interval=0.2) as sampler:
            t0 = time.perf_counter()
            examples = sp.run()
            wall = time.perf_counter() - t0
        row = {
            "shards": shards,
            "shard_games": list(sp._shard_games),
            "games": sp.games,
            "wall_s": wall,
            "games_per_hour": sp.games * 3600.0 / wall,
            "examples": len(examples),
            "gpu_busy_s": sp.gpu_busy_s,
            "gpu_busy_fraction": sp.gpu_busy_fraction,
            "gather_s": sp.gather_s,
            "infer_s": sp.infer_s,
            "apply_s": sp.apply_s,
            "advance_s": sp.advance_s,
            "inference_calls": sp.inference_calls,
            "batch_mean": sp.batch_stats.get("batch_mean"),
            "batch_p50": sp.batch_stats.get("batch_p50"),
        }
        row.update(sampler.summary())
        rows.append(row)
        print(f"shards={shards:2d} games={sp.games:3d} "
              f"wall={wall:7.2f}s  games/h={row['games_per_hour']:8.1f}  "
              f"gpu_busy={100 * row['gpu_busy_fraction']:5.1f}%  "
              f"nvsmi_util={row['gpu_util_mean'] or 0:5.1f}%  "
              f"power={row['gpu_power_mean_w'] or 0:5.1f}W  "
              f"batch={row['batch_mean'] or 0:7.1f}", flush=True)

    if len(rows) > 1:
        base = rows[0]
        print("\nrelative to shards=%d:" % base["shards"])
        for r in rows[1:]:
            print(f"  shards={r['shards']:2d}: "
                  f"{r['games_per_hour'] / base['games_per_hour']:.2f}x games/hour, "
                  f"gpu_busy {100 * base['gpu_busy_fraction']:.1f}% -> "
                  f"{100 * r['gpu_busy_fraction']:.1f}%")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"arch": args.arch, "games": args.games,
                       "sims": cfg.num_simulations, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
