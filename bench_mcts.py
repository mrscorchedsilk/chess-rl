"""Benchmark MCTS search speed. Times search() at N sims from a mid-game position,
and breaks down where the time goes (network forward vs Python overhead)."""
import time
import os
import sys

import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model import ChessNet
from mcts import MCTS
import encoding


def sync_device(device):
    """Synchronize only CUDA benchmarks; CPU runs need no device barrier."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def main():
    cfg = Config()
    print(f"device={cfg.device}  num_simulations={cfg.num_simulations}")
    net = ChessNet(cfg).to(cfg.device)
    net.eval()
    best = os.path.join(cfg.checkpoint_dir, "best.pt")
    if os.path.exists(best):
        net.load_state_dict(torch.load(best, map_location=cfg.device))
        print("loaded best.pt")

    # a middlegame position (both sides have pieces, many legal moves)
    fen = "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8"
    board = chess.Board(fen)

    mcts = MCTS(net, cfg)

    # warm up (lazy allocs, cudnn autotune)
    mcts.search(board, temperature=1.0, num_sims=10)

    runs = 5
    total = 0.0
    for _ in range(runs):
        t0 = time.time()
        mcts.search(board, temperature=1.0, num_sims=cfg.num_simulations)
        total += time.time() - t0
    avg = total / runs
    print(f"\nsearch() @ {cfg.num_simulations} sims: {avg*1000:.1f} ms/move  ({runs} runs)")
    print(f"  => games/hour (at ~117 plies/game, search only): {3600/(avg*117):.1f}")

    # ---- breakdown: time the network forward calls alone ----
    print("\n--- network forward cost ---")
    x = torch.from_numpy(encoding.encode_board(board)).unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        for _ in range(10):
            net(x)
        sync_device(cfg.device)
        t0 = time.time()
        for _ in range(200):
            net(x)
        sync_device(cfg.device)
        dt = (time.time() - t0) / 200
    print(f"batch=1 forward: {dt*1000:.2f} ms/call")
    # batch of 32
    x32 = torch.stack([torch.from_numpy(encoding.encode_board(board))] * 32).to(cfg.device)
    with torch.no_grad():
        for _ in range(10):
            net(x32)
        sync_device(cfg.device)
        t0 = time.time()
        for _ in range(200):
            net(x32)
        sync_device(cfg.device)
        dt32 = (time.time() - t0) / 200
    print(f"batch=32 forward: {dt32*1000:.2f} ms/call  (per-pos: {dt32/32*1000:.3f} ms)")

if __name__ == "__main__":
    main()
