"""Self-play pipeline benchmark: phase split, batch sizes, throughput.

Two modes:

  --fake   CPU-only.  A deterministic in-process evaluator replaces the GPU so
           the numbers isolate ACTOR cost (tree descent, thread scheduling,
           encode, backprop).  Use this to compare native builds A/B.

  --gpu    Production path: native Actor + persistent InferenceRuntime.
           Reports the gather/inference/apply/advance split, leaf batch
           distribution, and sampled GPU utilisation.

Both modes report rounds/s and, with --full-games, real games/hour.

Run:
  .venv/bin/python benchmarks/selfplay_bench.py --fake --games 20 --rounds 300
  .venv/bin/python benchmarks/selfplay_bench.py --gpu  --games 20 --full-games
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess_rl_native as native   # noqa: E402
from config import ARCHITECTURES, Config   # noqa: E402


def fake_evaluator(inputs, legal_offsets, legal_indices):
    """Cheap deterministic stand-in for the network (no GPU involved)."""
    inputs = np.asarray(inputs, dtype=np.float32)
    indices = np.asarray(legal_indices, dtype=np.int32)
    offsets = np.asarray(legal_offsets, dtype=np.int32)
    # A hash of the first plane row keeps this position-dependent but cheap.
    seed = (inputs[:, 0, 0, :].sum(axis=1) * 1000.0).astype(np.int64)
    logits = np.zeros(indices.shape[0], dtype=np.float32)
    for i in range(inputs.shape[0]):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e > s:
            logits[s:e] = ((np.arange(s, e) + int(seed[i])) % 17).astype(np.float32)
    values = np.zeros((inputs.shape[0], 1), dtype=np.float32)
    return logits, values


class GpuSampler:
    """Background nvidia-smi sampler; mean/max utilisation over the window."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                parts = out.stdout.strip().split(",")
                if len(parts) >= 2:
                    self.samples.append((float(parts[0]), float(parts[1])))
            except Exception:  # noqa: BLE001 - sampling must never fail a bench
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)

    def summary(self):
        if not self.samples:
            return {"gpu_util_mean": None, "gpu_util_max": None,
                    "gpu_power_mean_w": None, "gpu_samples": 0}
        util = [s[0] for s in self.samples]
        power = [s[1] for s in self.samples]
        return {
            "gpu_util_mean": float(np.mean(util)),
            "gpu_util_max": float(np.max(util)),
            "gpu_power_mean_w": float(np.mean(power)),
            "gpu_samples": len(util),
        }


def build_actor(cfg, games, threads, seed):
    return native.Actor(
        games=games, c_puct=float(cfg.c_puct),
        virtual_loss=float(cfg.virtual_loss),
        num_simulations=int(cfg.num_simulations),
        temperature=float(cfg.temperature),
        temperature_threshold=int(cfg.temperature_threshold),
        max_game_length=int(cfg.max_game_length),
        seed=seed, num_threads=threads,
    )


def run(cfg, evaluate, games, threads, max_batch, max_rounds, full_games, seed):
    actor = build_actor(cfg, games, threads, seed)
    actor.set_teacher(0, 0)
    t_gather = t_infer = t_apply = t_adv = 0.0
    rounds = 0
    batches: list[int] = []
    t0 = time.perf_counter()
    while not actor.is_done():
        if not full_games and rounds >= max_rounds:
            break
        a = time.perf_counter()
        tokens, inputs, offsets, indices = actor.gather_leaves(max_batch)
        b = time.perf_counter()
        t_gather += b - a
        if len(tokens) == 0:
            actor.advance()
            continue
        batches.append(len(tokens))
        logits, values = evaluate(inputs, offsets, indices)
        c = time.perf_counter()
        t_infer += c - b
        actor.apply_evaluations(tokens, offsets, logits, values)
        d = time.perf_counter()
        t_apply += d - c
        actor.advance()
        t_adv += time.perf_counter() - d
        rounds += 1
    wall = time.perf_counter() - t0
    finished = actor.finished_games()
    bs = np.array(batches) if batches else np.array([0])
    return {
        "wall_s": wall,
        "rounds": rounds,
        "rounds_per_s": rounds / wall if wall else None,
        "games": games,
        "threads": int(getattr(actor, "num_threads", threads or games)),
        "games_finished": len(finished),
        "examples": sum(len(g["examples"]) for g in finished),
        "gather_s": t_gather, "gather_pct": 100 * t_gather / wall if wall else None,
        "infer_s": t_infer, "infer_pct": 100 * t_infer / wall if wall else None,
        "apply_s": t_apply, "apply_pct": 100 * t_apply / wall if wall else None,
        "advance_s": t_adv, "advance_pct": 100 * t_adv / wall if wall else None,
        "batch_mean": float(bs.mean()), "batch_p50": float(np.percentile(bs, 50)),
        "batch_max": int(bs.max()), "batch_min": int(bs.min()),
        "leaves_total": int(bs.sum()),
        "leaves_per_s": float(bs.sum()) / wall if wall else None,
        "games_per_hour": (len(finished) * 3600.0 / wall) if (full_games and wall) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true",
                      help="CPU-only deterministic evaluator (isolates actor cost)")
    mode.add_argument("--gpu", action="store_true",
                      help="production GPU InferenceRuntime")
    ap.add_argument("--arch", default="v2-6x128")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="0 -> actor default")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--full-games", action="store_true",
                    help="run to completion and report real games/hour")
    ap.add_argument("--max-game-length", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.arch not in ARCHITECTURES:
        raise SystemExit(f"unknown architecture {args.arch!r}")
    cfg.architecture_id = args.arch
    cfg.num_res_blocks, cfg.num_filters = ARCHITECTURES[args.arch]
    if args.sims is not None:
        cfg.num_simulations = args.sims
    if args.max_game_length is not None:
        cfg.max_game_length = args.max_game_length

    sampler = None
    if args.gpu:
        from gpu_runtime import InferenceRuntime
        from model import ChessNet
        net = ChessNet(cfg)
        buckets = tuple(b for b in (32, 64, 128, 256, 512, 1024, 2048, 4096)
                        if b <= max(256, args.max_batch))
        rt = InferenceRuntime(cfg=cfg, model=net, buckets=buckets)
        evaluate = rt.evaluate
        # warm the compiled graphs / bucket buffers before timing
        run(cfg, evaluate, min(args.games, 4), args.threads or 0,
            args.max_batch, 20, False, args.seed + 1)
        sampler = GpuSampler()
    else:
        evaluate = fake_evaluator

    if sampler is not None:
        with sampler:
            res = run(cfg, evaluate, args.games, args.threads,
                      args.max_batch, args.rounds, args.full_games, args.seed)
        res.update(sampler.summary())
    else:
        res = run(cfg, evaluate, args.games, args.threads,
                  args.max_batch, args.rounds, args.full_games, args.seed)

    res["mode"] = "gpu" if args.gpu else "fake"
    res["arch"] = args.arch
    res["sims"] = cfg.num_simulations
    res["max_batch"] = args.max_batch
    res["native_module"] = native.__file__
    res["label"] = args.label or res["mode"]

    w = max(len(k) for k in res)
    print(f"--- selfplay_bench [{res['label']}] "
          f"{res['mode']} arch={args.arch} games={args.games} ---")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:<{w}} : {v:,.3f}")
        elif isinstance(v, int):
            print(f"  {k:<{w}} : {v:,}")
        else:
            print(f"  {k:<{w}} : {v}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
