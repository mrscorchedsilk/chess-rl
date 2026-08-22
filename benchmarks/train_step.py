"""Benchmark training-step cost and peak VRAM across model sizes (Task 8/9).

Runs one full `_epoch_train` (a bounded number of real minibatches) for each
architecture and reports: step time, positions/s, and peak VRAM.  Uses AMP
(GradScaler + autocast) when CUDA is available.  Model sizes beyond the GPU's
memory report a clean OOM instead of crashing.

Run: .venv/bin/python benchmarks/train_step.py --models 6x128,10x128,10x192,10x256
"""

import argparse
import time
import warnings

import numpy as np
import torch

warnings.simplefilter("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ARCHITECTURES, Config  # noqa: E402
from model import ChessNet  # noqa: E402
from replay import ReplayBuffer  # noqa: E402
import train  # noqa: E402


def _fill_buffer(capacity, cfg, n=256, seed=0):
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(capacity, cfg.policy_size)
    for _ in range(n):
        state = rng.random((cfg.num_input_planes, 8, 8), dtype=np.float32)
        pi = np.zeros(cfg.policy_size, dtype=np.float32)
        idx = rng.choice(cfg.policy_size, 24, replace=False)
        pi[idx] = 1.0
        pi /= pi.sum()
        buf.add(state, pi, float(rng.choice([-1.0, 0.0, 1.0])))
    return buf


def bench_one(arch_id, device, iters=3):
    num_res, num_f = ARCHITECTURES[arch_id]
    cfg = Config()
    cfg.architecture_id = arch_id
    cfg.num_res_blocks = num_res
    cfg.num_filters = num_f
    cfg.device = device
    cfg.train_batch_size = 256
    cfg.training_epochs = 1
    cfg.train_epoch_size = 256 * 2  # two minibatches per measurement

    net = ChessNet(cfg).to(device)
    optimizer = train._new_optimizer(cfg, net)
    scaler = train._new_scaler(cfg, device)
    buffer = _fill_buffer(2000, cfg)

    # warmup
    train._epoch_train(cfg, net, optimizer, buffer, device, scaler)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = train._epoch_train(cfg, net, optimizer, buffer, device, scaler)
    dt = time.perf_counter() - t0
    peak_mib = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                if device == "cuda" else 0.0)
    params = sum(p.numel() for p in net.parameters())
    return {
        "arch": arch_id,
        "params": params,
        "steps": out["steps"],
        "step_time_ms": dt / max(1, out["steps"]) * 1000,
        "positions_s": out["steps"] * cfg.train_batch_size / dt,
        "peak_vram_mib": peak_mib,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v2-6x128,v3-10x128,v3-10x192,v3-10x256")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}", flush=True)
    print(f"{'arch':<10} {'params':>10} {'step_ms':>9} {'pos/s':>9} {'vram_mib':>9}")
    for aid in args.models.split(","):
        aid = aid.strip()
        # Accept bare "6x128" or registered "v2-6x128".
        if aid not in ARCHITECTURES and f"v2-{aid}" in ARCHITECTURES:
            aid = f"v2-{aid}"
        try:
            r = bench_one(aid, device)
            print(f"{r['arch']:<10} {r['params']:>10,} {r['step_time_ms']:>9.2f} "
                  f"{r['positions_s']:>9.0f} {r['peak_vram_mib']:>9.1f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{aid:<10} {'OOM':>10} (too large for this GPU)", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
