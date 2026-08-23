#!/usr/bin/env python3
"""Controlled diversity canary for the corrected native self-play trainer.

Warm-starts a NEW lineage from the selected champion weights (empty replay,
fresh optimizer, new run id) into a SEPARATE checkpoint directory, runs a
bounded number of iterations, then reports measured replay diversity and the
per-iteration self-play round seeds.  Never touches checkpoints/v2.

Run:
  .venv/bin/python scripts/diversity_canary.py \
      --warm-start checkpoints/v2/latest.pt \
      --checkpoint-dir checkpoints/v2-canary \
      --iterations 12 --arena-every 6
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from config import Config  # noqa: E402
import train  # noqa: E402
import audit_replay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", required=True)
    ap.add_argument("--checkpoint-dir", default=os.path.join(HERE, "checkpoints", "v2-canary"))
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--arena-every", type=int, default=6)
    ap.add_argument("--arena-games", type=int, default=20)
    args = ap.parse_args()

    cfg = Config()
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.metrics_path = os.path.join(args.checkpoint_dir, "training.jsonl")
    cfg.num_simulations = 25
    cfg.games_per_iteration = 12
    cfg.num_iterations = args.iterations
    cfg.arena_every = args.arena_every
    cfg.arena_games = args.arena_games
    cfg.arena_simulations = 40
    cfg.checkpoint_every_iterations = 1

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    t0 = time.monotonic()
    train.run_native(cfg, resume=False, warm_start_checkpoint=args.warm_start)
    wall = time.monotonic() - t0

    # ---- report round seeds + arena suite hashes from metrics ----
    recs = []
    if os.path.exists(cfg.metrics_path):
        with open(cfg.metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    iters = [r for r in recs if r.get("event") != "arena"]
    arenas = [r for r in recs if r.get("event") == "arena"]
    seeds = [r["round_seed"] for r in iters if "round_seed" in r]
    print(f"\n[canary] {len(iters)} iterations in {wall:.1f}s "
          f"({wall / max(1, len(iters)):.1f}s/iter)")
    print(f"[canary] round seeds ({len(seeds)}): {seeds[:12]}{'...' if len(seeds) > 12 else ''}")
    print(f"[canary] distinct round seeds: {len(set(seeds))}/{len(seeds)}")
    for a in arenas:
        print(f"[canary] arena iter {a['iteration']}: "
              f"seed={a.get('opening_seed')} pairs={a.get('opening_pairs')} "
              f"suite_hash={str(a.get('opening_suite_hash'))[:16]} "
              f"score={a.get('score')} accepted={a.get('accepted')}")

    # ---- replay diversity audit ----
    latest = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if os.path.exists(latest):
        rep = audit_replay.audit_replay(latest)
        print("\n[canary] post-fix replay audit:")
        for k in ("replay_example_count", "unique_packed_states",
                  "unique_exact_examples", "unique_state_fraction",
                  "unique_example_fraction", "inferred_game_start_count",
                  "unique_full_game_trajectory_hashes",
                  "most_repeated_trajectory_count",
                  "unique_12_game_blocks", "most_repeated_12_game_block_count"):
            print(f"  {k:38s} {rep[k]}")

        n = rep["replay_example_count"]
        g = rep["inferred_game_start_count"]
        print(f"\n[canary] positions/hour = {n / wall * 3600:.0f}")
        print(f"[canary] games/hour    = {g / wall * 3600:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
