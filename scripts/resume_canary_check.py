#!/usr/bin/env python3
"""Resume the diversity canary from its last checkpoint to prove stop/save/resume.

Resumes checkpoints/v2-canary at the next iteration, runs 2 more iterations
(no arena for speed), and prints the resulting checkpoint metadata so the
caller can verify run_id continuity + iteration advance.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from config import Config  # noqa: E402
import train  # noqa: E402

cfg = Config()
cfg.checkpoint_dir = os.path.join(HERE, "checkpoints", "v2-canary")
cfg.metrics_path = os.path.join(cfg.checkpoint_dir, "training.jsonl")
cfg.num_simulations = 25
cfg.games_per_iteration = 12
cfg.num_iterations = 13       # resume at 12 -> run 12 and 13
cfg.arena_every = 100_000     # skip arena for a fast resume check
cfg.checkpoint_every_iterations = 1

train.run_native(cfg, resume=True)

meta = json.load(open(os.path.join(cfg.checkpoint_dir, "checkpoint_meta.json")))
print("\n[resume] checkpoint_meta:", json.dumps(
    {k: meta.get(k) for k in ("run_id", "iteration", "generation",
                              "reason", "replay_size", "optimizer_steps")},
    indent=2))
