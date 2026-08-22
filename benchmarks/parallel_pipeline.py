#!/usr/bin/env python3
"""Bounded four-worker self-play benchmark with opt-in JSONL diagnostics.

Unlike ``test_parallel_train.py``, this measures only the self-play pipeline. It
excludes optimizer, arena and checkpoint time so the four-game/30-second gate is
well-defined and attributable.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config import Config
from model import ChessNet
from parallel import InferenceServer, worker_loop


def summarize_profile(rows):
    """Reduce raw diagnostic events into stable machine-readable metrics."""
    batches = [int(r["positions"]) for r in rows if r.get("event") == "batch_formed"]
    completed = [r for r in rows if r.get("event") == "inference_batch_completed"]
    games = [r for r in rows if r.get("event") == "game_completed"]

    timing = {
        "h2d": sum(float(r.get("h2d_ms", 0.0)) for r in completed),
        "forward": sum(float(r.get("forward_ms", 0.0)) for r in completed),
        "d2h": sum(float(r.get("d2h_ms", 0.0)) for r in completed),
        "legal": sum(float(r.get("legal_ms", 0.0)) for r in completed),
        "total": sum(float(r.get("total_ms", 0.0)) for r in completed),
    }
    measured_total = timing["total"]
    transfer = timing["h2d"] + timing["d2h"]
    batch_summary = {
        "count": len(batches),
        "min": min(batches) if batches else None,
        "median": statistics.median(batches) if batches else None,
        "max": max(batches) if batches else None,
    }
    return {
        "event_counts": {
            event: sum(1 for r in rows if r.get("event") == event)
            for event in sorted({str(r.get("event")) for r in rows})
        },
        "games_completed": len(games),
        "plies_completed": sum(int(r.get("plies", 0)) for r in games),
        "batch_positions": batch_summary,
        "timing_ms": timing,
        "transfer_fraction_of_measured_batch": (
            transfer / measured_total if measured_total > 0.0 else None
        ),
    }


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _validate_game(examples, cfg):
    if not examples:
        raise RuntimeError("self-play returned an empty game")
    for state, policy, value in examples:
        if state.shape != (cfg.num_input_planes, cfg.board_size, cfg.board_size):
            raise RuntimeError(f"invalid state shape: {state.shape}")
        if policy.shape != (cfg.policy_size,):
            raise RuntimeError(f"invalid policy shape: {policy.shape}")
        if not np.isfinite(state).all() or not np.isfinite(policy).all():
            raise RuntimeError("non-finite replay example")
        if abs(float(policy.sum()) - 1.0) >= 1e-3:
            raise RuntimeError(f"policy mass is {policy.sum()}")
        if value not in (-1.0, 0.0, 1.0):
            raise RuntimeError(f"invalid game value: {value}")


def _close_queues(queues):
    for q in queues:
        try:
            q.close()
            q.join_thread()
        except (AttributeError, OSError, ValueError):
            pass


def run_benchmark(args):
    cfg = Config()
    cfg.device = args.device or cfg.device
    cfg.num_simulations = args.sims
    cfg.batch_size = args.batch
    cfg.max_game_length = args.max_plies
    cfg.result_timeout_seconds = args.timeout
    setattr(cfg, "inference_timeout_seconds", args.timeout)

    profile_path = Path(args.profile_jsonl).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.unlink(missing_ok=True)
    os.environ["CHESS_PROFILE_JSONL"] = str(profile_path)

    net = ChessNet(cfg).to(cfg.device)
    net.eval()

    ctx = mp.get_context("spawn")
    channels = []
    all_queues = []
    procs = []
    result_queue = ctx.Queue(maxsize=max(8, args.workers * 2))
    all_queues.append(result_queue)
    stop_event = ctx.Event()
    generation = ctx.Value("i", 0)
    for worker in range(args.workers):
        request_queue = ctx.Queue(maxsize=8)
        response_queue = ctx.Queue(maxsize=8)
        channels.append((request_queue, response_queue))
        all_queues.extend((request_queue, response_queue))
        procs.append(ctx.Process(
            target=worker_loop,
            args=(cfg, request_queue, response_queue, result_queue,
                  cfg.seed + 1000 + worker, stop_event, worker, generation),
            daemon=True,
        ))

    server = InferenceServer(
        net, cfg.device, channels,
        max_batch=cfg.inference_max_batch,
        min_batch=cfg.inference_min_batch,
    )
    games = []
    started = time.perf_counter()
    try:
        server.start()
        for proc in procs:
            proc.start()
        deadline = started + args.timeout
        while len(games) < args.games:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise RuntimeError(
                    f"benchmark timed out after {args.timeout}s with "
                    f"{len(games)}/{args.games} games"
                )
            try:
                item = result_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                dead = [p.pid for p in procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"workers died before completing games: {dead}")
                if not server.is_alive():
                    raise RuntimeError(
                        f"inference server stopped: {server.get_error() or 'unknown'}"
                    )
                continue
            if isinstance(item, dict) and item.get("kind") == "error":
                raise RuntimeError(
                    f"worker {item.get('worker')} failed:\n{item.get('traceback')}"
                )
            examples = item.get("examples") if isinstance(item, dict) else item
            games.append(examples)
        wall_seconds = time.perf_counter() - started
    finally:
        stop_event.set()
        for proc in procs:
            proc.join(timeout=2.0)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=2.0)
        server.stop()
        _close_queues(all_queues)

    for game in games:
        _validate_game(game, cfg)
    rows = _read_jsonl(profile_path)
    profile = summarize_profile(rows)
    result = {
        "schema": 1,
        "backend": "python-processes",
        "device": str(cfg.device),
        "workers": args.workers,
        "games": len(games),
        "simulations": args.sims,
        "mcts_batch_size": args.batch,
        "max_game_length": args.max_plies,
        "wall_seconds": wall_seconds,
        "games_per_hour": len(games) / wall_seconds * 3600.0,
        "examples": sum(len(game) for game in games),
        "profile_jsonl": str(profile_path),
        "profile": profile,
    }
    if len(games) == 4:
        result["four_game_under_30s"] = wall_seconds < 30.0
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--sims", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--profile-jsonl", default="/tmp/chess-parallel-profile.jsonl")
    parser.add_argument("--json", dest="json_path", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.json_path:
        out = Path(args.json_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
