"""End-to-end smoke test of the parallel training loop (run_parallel).

Uses a tiny config (few games, few sims, short arena, /tmp checkpoints) so it
finishes in seconds, then prints the checkpoints it produced.  Run:

    .venv/bin/python test_parallel_train.py
"""

import os
import shutil

from config import Config
from train import run_parallel


def main():
    cfg = Config()
    cfg.num_simulations = 40
    cfg.batch_size = 32
    cfg.games_per_iteration = 4
    cfg.num_iterations = 2
    cfg.arena_every = 2          # trigger the arena gate on iteration 2
    cfg.arena_games = 2
    cfg.arena_simulations = 10
    cfg.replay_buffer_size = 5000
    cfg.checkpoint_interval_minutes = 60  # avoid a mid-run time checkpoint
    cfg.checkpoint_dir = "/tmp/chess_parallel_test"

    shutil.rmtree(cfg.checkpoint_dir, ignore_errors=True)

    run_parallel(cfg=cfg, resume=False, num_workers=4)

    print("\n=== checkpoint dir ===")
    for f in sorted(os.listdir(cfg.checkpoint_dir)):
        print(f"  {f}  {os.path.getsize(os.path.join(cfg.checkpoint_dir, f))} bytes")


if __name__ == "__main__":
    main()
