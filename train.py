"""Training loop for the AlphaZero-style chess learner.

run() orchestrates: self-play -> replay buffer -> optimize -> arena gate ->
checkpoint. CPU-friendly: everything runs on cfg.device (auto CPU if no CUDA).

Checkpointing:
  - `best.pt`    : incumbent best net (arena-accepted). Refreshed whenever a new
                   net passes the arena gate, and again at the end of training.
                   On a fresh (non-resume) start any existing best.pt is first
                   archived to best-<timestamp>.pt so a reset can't destroy it.
  - `latest.pt`  : full resumable training state (net + best_net + optimizer +
                   iteration + RNG), written every `checkpoint_interval_minutes`
                   so a crashed run can be resumed without losing much work.
  - `ckpt-iterNNNN-<timestamp>.pt` : versioned, never-overwritten snapshot of
                   latest.pt taken at each periodic checkpoint (hardlink), so
                   checkpoint history is countable and recoverable.
"""

import json
import os
import random
import shutil
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from model import ChessNet
from selfplay import play_game
from arena import play_match
from parallel import InferenceServer, worker_loop


METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training.jsonl")


def _log_metrics(cfg, iteration, avg_loss, games, arena_win_rate, accepted):
    """Append one structured record per iteration to training.jsonl (machine-readable)."""
    try:
        rec = {
            "t": time.time(),
            "iteration": iteration,
            "loss": round(float(avg_loss), 4),
            "games": int(games),
            "arena_win_rate": None if arena_win_rate is None else round(float(arena_win_rate), 3),
            "accepted": accepted,
        }
        with open(METRICS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _sample_batch(buffer, cfg, device):
    """Sample a training batch of (states, pis, zs) tensors on device."""
    batch = random.sample(buffer, min(cfg.train_batch_size, len(buffer)))
    states = torch.tensor(np.stack([e[0] for e in batch])).to(device)
    pis = torch.tensor(np.stack([e[1] for e in batch])).to(device)
    zs = torch.tensor([e[2] for e in batch], dtype=torch.float32).unsqueeze(1).to(device)
    return states, pis, zs


def _checkpoint_paths(cfg):
    return {
        "best": os.path.join(cfg.checkpoint_dir, "best.pt"),
        "latest": os.path.join(cfg.checkpoint_dir, "latest.pt"),
    }


def _save_latest(cfg, net, best_net, optimizer, iteration):
    """Persist a full resumable training snapshot to `latest.pt`."""
    payload = {
        "iteration": iteration,
        "net": net.state_dict(),
        "best_net": best_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "random_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }
    path = _checkpoint_paths(cfg)["latest"]
    # Atomic-ish write: save to a temp name then rename so a crash mid-write
    # can't corrupt the last good snapshot.
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def _load_latest(cfg):
    """Return the latest resumable snapshot, or None if it doesn't exist."""
    path = _checkpoint_paths(cfg)["latest"]
    if not os.path.exists(path):
        return None
    # weights_only=False: the snapshot is self-created and trusted, and it
    # carries non-tensor state (numpy RNG) that weights_only=True rejects.
    return torch.load(path, map_location=cfg.device, weights_only=False)


def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _archive_best(cfg):
    """Move an existing best.pt out of the way before it's overwritten, so a
    fresh (non-resume) start can never silently destroy the incumbent best net."""
    best_path = _checkpoint_paths(cfg)["best"]
    if os.path.exists(best_path):
        archived = os.path.join(cfg.checkpoint_dir, f"best-{_timestamp()}.pt")
        os.replace(best_path, archived)
        print(f"Archived previous best -> {os.path.basename(archived)}", flush=True)


def _snapshot_checkpoint(cfg, iteration):
    """Hardlink the just-written latest.pt to a versioned, never-overwritten
    filename so checkpoints are countable and recoverable after a reset."""
    latest_path = _checkpoint_paths(cfg)["latest"]
    if not os.path.exists(latest_path):
        return None
    versioned = os.path.join(
        cfg.checkpoint_dir, f"ckpt-iter{iteration:04d}-{_timestamp()}.pt"
    )
    try:
        os.link(latest_path, versioned)
    except OSError:
        shutil.copy2(latest_path, versioned)
    return versioned


def run(cfg=None, resume=False):
    """The full self-play / train / arena loop."""
    if cfg is None:
        cfg = Config()

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    net = ChessNet(cfg).to(cfg.device)
    best_net = ChessNet(cfg).to(cfg.device)
    best_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    buffer = deque(maxlen=cfg.replay_buffer_size)
    start_iteration = 1

    if resume:
        state = _load_latest(cfg)
        if state is not None:
            net.load_state_dict(state["net"])
            best_net.load_state_dict(state["best_net"])
            optimizer.load_state_dict(state["optimizer"])
            # RNG states must be CPU ByteTensors for set_rng_state; loading with
            # map_location=device may have moved them to CUDA.
            rng = state["torch_rng"]
            torch.set_rng_state(rng.cpu() if rng.is_cuda else rng)
            if state.get("cuda_rng") and torch.cuda.is_available():
                cuda_rng = state["cuda_rng"]
                if isinstance(cuda_rng, (list, tuple)) and all(
                    hasattr(r, "cpu") for r in cuda_rng
                ):
                    torch.cuda.set_rng_state_all(
                        [r.cpu() if r.is_cuda else r for r in cuda_rng]
                    )
            random.setstate(state["random_rng"])
            np.random.set_state(state["numpy_rng"])
            start_iteration = state["iteration"] + 1
            print(
                f"Resumed from latest.pt (next iter {start_iteration})", flush=True
            )

    best_path = _checkpoint_paths(cfg)["best"]
    if not resume:
        _archive_best(cfg)
    torch.save(best_net.state_dict(), best_path)

    interval_s = getattr(cfg, "checkpoint_interval_minutes", 10) * 60.0
    last_ckpt = time.time()
    last_arena_win_rate = None
    last_accepted = None

    for iteration in range(start_iteration, cfg.num_iterations + 1):
        # ---- 1. self-play ----
        net.eval()
        for _ in range(cfg.games_per_iteration):
            buffer.extend(play_game(net, cfg))

        # ---- 2. optimize on a batch from the replay buffer ----
        net.train()
        total_loss = 0.0
        for _ in range(cfg.epochs_per_iteration):
            states, pis, zs = _sample_batch(buffer, cfg, cfg.device)
            logits, value = net(states)
            loss = F.mse_loss(value, zs) + F.cross_entropy(logits, pis)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / cfg.epochs_per_iteration

        # ---- 3. arena gate against the incumbent best ----
        msg = f"iter {iteration:4d}  loss {avg_loss:.4f}  buffer {len(buffer)}"
        if iteration % cfg.arena_every == 0:
            net.eval()
            best_net.eval()
            result = play_match(net, best_net, cfg, num_games=cfg.arena_games)
            win_rate = result["a"] / cfg.arena_games  # net is player 'a'
            last_arena_win_rate = win_rate
            last_accepted = False
            msg += (
                f"  arena: a {result['a']} / b {result['b']} / draw {result['draws']}"
                f"  (win-rate {win_rate:.2f})"
            )
            if win_rate >= cfg.arena_accept_threshold:
                last_accepted = True
                best_net.load_state_dict(net.state_dict())
                best_net.eval()
                torch.save(best_net.state_dict(), best_path)
                msg += "  -> accepted as new best"

        # ---- 4. periodic time-based checkpoint (crash recovery) ----
        if time.time() - last_ckpt >= interval_s:
            path = _save_latest(cfg, net, best_net, optimizer, iteration)
            snap = _snapshot_checkpoint(cfg, iteration)
            last_ckpt = time.time()
            msg += f"  [ckpt {os.path.basename(path)}]"
            if snap:
                msg += f"  [snap {os.path.basename(snap)}]"

        _log_metrics(cfg, iteration, avg_loss, iteration * cfg.games_per_iteration,
                     last_arena_win_rate, last_accepted)
        print(msg, flush=True)

    # Final checkpoint of whatever ended up as best.
    torch.save(best_net.state_dict(), best_path)
    print("Training complete.")


def run_parallel(cfg=None, resume=False, num_workers=None):
    """Self-play / train / arena loop with parallel self-play worker processes.

    CPU-heavy MCTS work runs in `num_workers` processes (one game each, in
    parallel); a shared GPU inference server (a thread here) coalesces their
    leaf evaluations into fat forward passes.  The trainer process owns the
    training net, the arena incumbent, and the served `search_net` copy.
    """
    import multiprocessing as mp

    if cfg is None:
        cfg = Config()
    if num_workers is None or num_workers < 1:
        num_workers = max(1, getattr(cfg, "selfplay_workers", 12))

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ---- model / optimizer / resume (same as serial run) ----
    net = ChessNet(cfg).to(cfg.device)
    best_net = ChessNet(cfg).to(cfg.device)
    best_net.load_state_dict(net.state_dict())
    search_net = ChessNet(cfg).to(cfg.device)
    search_net.load_state_dict(net.state_dict())
    search_net.eval()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    start_iteration = 1

    if resume:
        state = _load_latest(cfg)
        if state is not None:
            net.load_state_dict(state["net"])
            best_net.load_state_dict(state["best_net"])
            optimizer.load_state_dict(state["optimizer"])
            rng = state["torch_rng"]
            torch.set_rng_state(rng.cpu() if rng.is_cuda else rng)
            if state.get("cuda_rng") and torch.cuda.is_available():
                cuda_rng = state["cuda_rng"]
                if isinstance(cuda_rng, (list, tuple)) and all(
                    hasattr(r, "cpu") for r in cuda_rng
                ):
                    torch.cuda.set_rng_state_all(
                        [r.cpu() if r.is_cuda else r for r in cuda_rng]
                    )
            random.setstate(state["random_rng"])
            np.random.set_state(state["numpy_rng"])
            start_iteration = state["iteration"] + 1
            search_net.load_state_dict(net.state_dict())
            print(
                f"Resumed from latest.pt (next iter {start_iteration})", flush=True
            )

    best_path = _checkpoint_paths(cfg)["best"]
    if not resume:
        _archive_best(cfg)
    torch.save(best_net.state_dict(), best_path)

    buffer = deque(maxlen=cfg.replay_buffer_size)

    # ---- spawn worker processes + inference server ----
    ctx = mp.get_context("spawn")
    channels = []
    procs = []
    result_queue = ctx.Queue(maxsize=max(8, num_workers * 2))
    stop_event = ctx.Event()
    for i in range(num_workers):
        req_q = ctx.Queue(maxsize=8)
        resp_q = ctx.Queue(maxsize=8)
        channels.append((req_q, resp_q))
        p = ctx.Process(
            target=worker_loop,
            args=(cfg, req_q, resp_q, result_queue, cfg.seed + 1000 + i, stop_event),
            daemon=True,
        )
        procs.append(p)

    server = InferenceServer(
        search_net, cfg.device, channels,
        max_batch=getattr(cfg, "inference_max_batch", 4096),
        min_batch=getattr(cfg, "inference_min_batch", 256),
    )
    server.start()
    for p in procs:
        p.start()
    print(
        f"[parallel] {num_workers} self-play workers + GPU inference server",
        flush=True,
    )

    interval_s = getattr(cfg, "checkpoint_interval_minutes", 10) * 60.0
    last_ckpt = time.time()
    last_arena_win_rate = None
    last_accepted = None

    try:
        for iteration in range(start_iteration, cfg.num_iterations + 1):
            # ---- 1. self-play: collect games_per_iteration games from workers ----
            # Serve the freshly-updated net for this round of games.
            server.update_weights(net.state_dict())
            for _ in range(cfg.games_per_iteration):
                buffer.extend(result_queue.get())

            # ---- 2. optimize ----
            net.train()
            total_loss = 0.0
            for _ in range(cfg.epochs_per_iteration):
                states, pis, zs = _sample_batch(buffer, cfg, cfg.device)
                logits, value = net(states)
                loss = F.mse_loss(value, zs) + F.cross_entropy(logits, pis)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            avg_loss = total_loss / cfg.epochs_per_iteration

            # ---- 3. arena gate against the incumbent best ----
            msg = f"iter {iteration:4d}  loss {avg_loss:.4f}  buffer {len(buffer)}"
            if iteration % cfg.arena_every == 0:
                net.eval()
                best_net.eval()
                result = play_match(net, best_net, cfg, num_games=cfg.arena_games)
                win_rate = result["a"] / cfg.arena_games
                last_arena_win_rate = win_rate
                last_accepted = False
                msg += (
                    f"  arena: a {result['a']} / b {result['b']} / draw {result['draws']}"
                    f"  (win-rate {win_rate:.2f})"
                )
                if win_rate >= cfg.arena_accept_threshold:
                    last_accepted = True
                    best_net.load_state_dict(net.state_dict())
                    best_net.eval()
                    torch.save(best_net.state_dict(), best_path)
                    msg += "  -> accepted as new best"

            # ---- 4. periodic time-based checkpoint (crash recovery) ----
            if time.time() - last_ckpt >= interval_s:
                path = _save_latest(cfg, net, best_net, optimizer, iteration)
                snap = _snapshot_checkpoint(cfg, iteration)
                last_ckpt = time.time()
                msg += f"  [ckpt {os.path.basename(path)}]"
                if snap:
                    msg += f"  [snap {os.path.basename(snap)}]"

            _log_metrics(cfg, iteration, avg_loss,
                         iteration * cfg.games_per_iteration,
                         last_arena_win_rate, last_accepted)
            print(msg, flush=True)
    finally:
        # Shutdown: signal workers, give them a beat to finish the current
        # game, then hard-stop anything left and stop the server.
        stop_event.set()
        for p in procs:
            p.join(timeout=2.0)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2.0)
        server.stop()

    torch.save(best_net.state_dict(), best_path)
    print("Training complete.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="AlphaZero-style chess self-play trainer")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/latest.pt if present",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel self-play worker processes (0 = serial loop; >=2 = process pool)",
    )
    args = p.parse_args()
    if args.workers >= 2:
        run_parallel(resume=args.resume, num_workers=args.workers)
    else:
        run(resume=args.resume)
