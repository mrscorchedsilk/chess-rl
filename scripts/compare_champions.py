#!/usr/bin/env python3
"""Compare accepted champion snapshots with the CORRECTED paired-opening arena.

The old arena was invalid (2 effective trajectories), so generation numbers are
unreliable.  This script loads the distinct accepted champions and plays them
head-to-head using the new deterministic paired-opening arena (10 openings,
colors swapped, temperature 0, no root noise).

Reports raw win/draw/loss + standard score only — no Elo improvement claim.
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from config import Config  # noqa: E402
from model import ChessNet  # noqa: E402
from arena import play_match, generate_arena_openings  # noqa: E402


def _load(path, cfg, device):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    net = ChessNet(cfg).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default=os.path.join(HERE, "checkpoints", "v2"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--simulations", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = Config()
    cfg.device = device
    cfg.arena_simulations = args.simulations
    cfg.arena_games = args.games
    cfg.arena_seed = 424242
    cfg.arena_opening_plies = 8
    cfg.arena_root_noise = False

    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    archived = sorted(glob.glob(os.path.join(args.checkpoint_dir, "best-*.pt")))
    if not archived:
        print("no archived champions found")
        return 1

    # Only ONE distinct archived champion (all byte-identical gen-0), so the
    # comparison is: latest accepted best (gen 4) vs the archived gen-0.
    gen0 = archived[0]
    print(f"champion A (latest best, gen 4): {best_path}")
    print(f"champion B (archived gen 0)   : {gen0}")

    net_a = _load(best_path, cfg, device)
    net_b = _load(gen0, cfg, device)

    openings = generate_arena_openings(
        args.games // 2, cfg.arena_opening_plies, cfg.arena_seed)
    print(f"openings ({len(openings)} pairs):")
    for i, o in enumerate(openings):
        print(f"  {i + 1:2d}: {' '.join(o)}")

    result = play_match(net_a, net_b, cfg, num_games=args.games)
    wins, losses, draws = result["a"], result["b"], result["draws"]
    games = wins + losses + draws
    score = (wins + 0.5 * draws) / games if games else 0.0
    print(f"\nlatest-best (gen4) vs archived (gen0):")
    print(f"  wins={wins} losses={losses} draws={draws} games={games}")
    print(f"  standard score (candidate=latest best) = {score:.3f}")


if __name__ == "__main__":
    main()
