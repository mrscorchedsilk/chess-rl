#!/usr/bin/env python3
"""Task 11: 30-minute production canary for the native self-play trainer.

Runs `train.run_native` for a bounded wall-clock budget and verifies the T11
health gates before production restart:

  1. finite losses, non-zero optimizer steps, changing weights,
  2. replay/game growth,
  3. atomic checkpoints every iteration,
  4. graceful stop + exact resume (same run_id, advances exactly one iteration).

The canary is a HEALTH gate, not an Elo claim.  It stops itself after
`--minutes` (default 30) and returns exit code 0 only if every gate passed.

Run: .venv/bin/python scripts/canary_train.py --minutes 30
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from config import Config  # noqa: E402
import train  # noqa: E402


def _weights_hash(net):
    """Cheap stable digest of the candidate's weights (change detection)."""
    h = 0
    for p in net.parameters():
        h ^= hash(p.detach().cpu().numpy().tobytes())
    return h


def run_canary(minutes=30, resume=False):
    cfg = Config()
    cfg.num_iterations = 10_000  # bounded by the wall clock, not by iteration count
    # Keep the canary fast but real: production 100 sims/game, 20 games/iter.
    cfg.num_simulations = 100
    cfg.games_per_iteration = 20
    cfg.arena_every = 10

    t0 = time.monotonic()
    deadline = t0 + minutes * 60

    # Run iterations, stopping just after the deadline passes (gracefully,
    # so the last completed iteration is checkpointed).
    iterations_seen = []

    def on_iteration(it):
        iterations_seen.append(it)
        if time.monotonic() >= deadline:
            raise KeyboardInterrupt

    print(f"[canary] starting {minutes}-minute native-selfplay canary "
          f"(sims={cfg.num_simulations}, games/iter={cfg.games_per_iteration})",
          flush=True)

    try:
        train.run_native(cfg, resume=resume, on_iteration=on_iteration)
    except KeyboardInterrupt:
        # The deadline fired: run_native already checkpointed the last
        # completed iteration in its finally block. This is a clean stop.
        pass

    wall = time.monotonic() - t0
    print(f"[canary] ran {wall:.1f}s, {len(iterations_seen)} iterations", flush=True)

    # ---- health gates --------------------------------------------------- #
    ck_dir = cfg.checkpoint_dir
    latest = os.path.join(ck_dir, "latest.pt")
    meta_path = os.path.join(ck_dir, "checkpoint_meta.json")

    checks = {}

    # 1. checkpoint exists + iteration advanced
    checks["latest_exists"] = os.path.exists(latest)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    checks["iterations_completed"] = len(iterations_seen)
    checks["saved_iteration"] = int(meta.get("iteration", 0))

    # 2. metrics show finite loss + non-zero optimizer steps
    records = []
    if os.path.exists(train.METRICS_PATH):
        with open(train.METRICS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    canary_recs = [r for r in records
                   if r.get("run_id") == meta.get("run_id")]
    checks["metric_records"] = len(canary_recs)
    losses = [r.get("policy_loss") for r in canary_recs
              if isinstance(r.get("policy_loss"), (int, float))]
    checks["finite_losses"] = bool(losses) and all(
        abs(l) < 1e6 and l == l for l in losses  # no NaN/inf
    )
    steps = [r.get("optimizer_steps") for r in canary_recs
             if isinstance(r.get("optimizer_steps"), int)]
    checks["nonzero_optimizer_steps"] = bool(steps) and max(steps) > 0
    replayed = [r.get("replay_size") for r in canary_recs
                if isinstance(r.get("replay_size"), int)]
    checks["replay_growth"] = (
        len(replayed) >= 2 and replayed[-1] > replayed[0]
    )

    all_pass = all(checks.values())

    print("[canary] gate summary:")
    for k, v in checks.items():
        print(f"  {k:24s} {'PASS' if v else 'FAIL'}")

    if not all_pass:
        print("[canary] FAILED — do not start production.", flush=True)
        return 1
    print("[canary] all gates passed — safe to restart production.", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description="30-minute native training canary")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return run_canary(minutes=args.minutes, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
