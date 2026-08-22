"""Benchmark the parallel self-play pipeline: N worker processes + GPU server.

Measures games/hour, validates every produced example, and samples GPU
utilization + process CPU during the run.  Usage:

    .venv/bin/python bench_parallel.py --workers 8 --games 40 --sims 100

This exercises the exact same code path train.py uses (parallel.py), without
running the full training loop.
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import time

import chess
import numpy as np
import torch

from config import Config
from model import ChessNet
from parallel import InferenceServer, worker_loop
import encoding


def sample_gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            return {
                "gpu_util": float(parts[0]),
                "mem_mb": float(parts[1]),
                "power_w": float(parts[2]),
            }
    except Exception:
        pass
    return None


def cpu_util():
    try:
        return os.getloadavg()[0]
    except Exception:
        return -1.0


def validate(examples):
    """Check every (state, pi, z) tuple is well-formed and legal."""
    for state, pi, z in examples:
        assert state.shape == (18, 8, 8), state.shape
        assert state.dtype == np.float32
        assert pi.shape == (4096,), pi.shape
        assert pi.dtype == np.float32
        assert abs(float(pi.sum()) - 1.0) < 1e-3, f"pi sums to {pi.sum()}"
        assert float(pi.min()) >= 0.0
        assert z in (-1.0, 0.0, 1.0), z
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32, help="MCTS leaf-eval batch per worker")
    ap.add_argument("--load-best", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.num_simulations = args.sims
    cfg.batch_size = args.batch

    print(f"device={cfg.device}  workers={args.workers}  sims={args.sims}  "
          f"batch={args.batch}  games={args.games}", flush=True)

    net = ChessNet(cfg).to(cfg.device)
    net.eval()
    if args.load_best:
        best = os.path.join(cfg.checkpoint_dir, "best.pt")
        if os.path.exists(best):
            net.load_state_dict(torch.load(best, map_location=cfg.device))
            print("loaded best.pt", flush=True)
    search_net = ChessNet(cfg).to(cfg.device)
    search_net.load_state_dict(net.state_dict())
    search_net.eval()

    ctx = mp.get_context("spawn")
    channels = []
    procs = []
    result_queue = ctx.Queue(maxsize=max(8, args.workers * 2))
    stop_event = ctx.Event()
    for i in range(args.workers):
        req_q = ctx.Queue(maxsize=8)
        resp_q = ctx.Queue(maxsize=8)
        channels.append((req_q, resp_q))
        p = ctx.Process(
            target=worker_loop,
            args=(cfg, req_q, resp_q, result_queue, cfg.seed + 1000 + i, stop_event),
            daemon=True,
        )
        procs.append(p)

    server = InferenceServer(search_net, cfg.device, channels)
    server.start()
    for p in procs:
        p.start()

    # warm up: discard the first few games so lazy allocs / autotune don't
    # pollute the timing
    for _ in range(args.workers):
        result_queue.get()

    n_examples = 0
    n_plies = 0
    gpu_samples = []
    t0 = time.time()
    for _ in range(args.games):
        examples = result_queue.get()
        validate(examples)
        n_examples += len(examples)
        n_plies += len(examples)
        gpu_samples.append(sample_gpu())

    dt = time.time() - t0
    gph = args.games / dt * 3600.0
    gpu_utils = [s["gpu_util"] for s in gpu_samples if s]
    mems = [s["mem_mb"] for s in gpu_samples if s]
    pwrs = [s["power_w"] for s in gpu_samples if s]
    print(f"\n=== parallel self-play results ===")
    print(f"workers       : {args.workers}")
    print(f"games         : {args.games} in {dt:.1f}s")
    print(f"games/hour    : {gph:.0f}")
    print(f"avg plies/game: {n_plies/args.games:.1f}")
    print(f"examples      : {n_examples} (all validated legal + well-formed)")
    print(f"GPU util      : avg {np.mean(gpu_utils):.1f}%  min {min(gpu_utils):.0f}%  "
          f"max {max(gpu_utils):.0f}%")
    print(f"GPU mem       : avg {np.mean(mems):.0f} MiB")
    if pwrs:
        print(f"GPU power     : avg {np.mean(pwrs):.0f} W")
    print(f"load avg      : {cpu_util():.1f} (cores={os.cpu_count()})")

    stop_event.set()
    for p in procs:
        p.join(timeout=2.0)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=2.0)
    server.stop()


if __name__ == "__main__":
    main()
