#!/usr/bin/env python3
"""Fixed, reproducible evaluation for the ChessNet learner (Sprint C).

Three fixed baselines, played head-to-head in paired-colour matches with
standard chess scoring (win = 1, draw = 0.5, loss = 0), plus a small
tactical competence suite (mate-in-one and avoid-immediate-mate positions).

  - RandomPlayer      : uniform-random legal move (seeded)
  - GreedyPlayer      : one-ply material-greedy (maximises captured material
                        difference from the mover's perspective; deterministic
                        tie-break = first move in python-chess generation order)
  - NetPlayer         : the ChessNet + MCTS (temperature 0, configurable sims)

Everything is seed-reproducible: the master seed drives numpy/torch and, via
:func:`seeded_context`, also the (otherwise unseeded) Dirichlet root noise in
MCTS, so two runs with the same seed produce byte-identical JSON.

CLI:

    .venv/bin/python evaluate.py --seed 42 --games 4 --sims 100 --out eval.json

Output is one JSON document (also mirrored to stdout):
  {seed, config, results: [{white, black, games, score_white, score_black,
    ci_white, ci_black, games_detail}], tactics: {mate_in_one, avoid_mate},
   summary}
Every pairing appears twice in ``results`` — once per colour assignment
(paired-colour records) — and play_match alternates colours within each match.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import chess
import numpy as np
import torch

from config import Config
from mcts import MCTS
from model import ChessNet

HERE = os.path.dirname(os.path.abspath(__file__))

# Piece values for the one-ply material-greedy baseline.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# --------------------------------------------------------------------------- #
#  Tactical competence suite                                                  #
# --------------------------------------------------------------------------- #
#  Mate-in-one: the side to move has (at least) one legal move that is
#  checkmate.  Passing = the baseline actually plays a mating move.
MATE_IN_ONE = [
    ("scholar_qxf7",
     "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4"),
    ("back_rank_ra8",
     "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("kq_corner_qg7",
     "7k/5K2/8/8/8/8/6Q1/8 w - - 0 1"),
    ("kq_edge_qb7",
     "k7/8/2K5/8/8/8/1Q6/8 w - - 0 1"),
]

#  Avoid-immediate-mate: the side to move is NOT in check, but at least one
#  legal move leaves the opponent with a mate-in-one (a blunder), and at least
#  one move is safe.  Passing = the chosen move does not allow mate-in-one.
AVOID_MATE = [
    ("fools_mate_setup",           # after 1.f3 e5: only 2.g4?? allows Qh4#
     "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2"),
    ("back_rank_trap",             # black to move: only Kh8?? allows Ra8#
     "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"),
    ("scholar_defense",            # black to move: must stop Qxf7#
     "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
]


# --------------------------------------------------------------------------- #
#  Statistics                                                                 #
# --------------------------------------------------------------------------- #

def wilson_interval(k, n, z=1.96):
    """Wilson score interval for k successes in n trials.

    `k` may be fractional (draws contribute 0.5).  Returns (low, high).
    Degenerate n <= 0 yields (0.0, 1.0).  The bounds are clamped to the
    [0, 1] unit interval and pinned exactly for the degenerate outcomes
    (k == 0 -> low == 0.0, k == n -> high == 1.0), which floating-point
    cancellation would otherwise push a hair outside.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    if k == 0:
        lo = 0.0
    if k >= n:
        hi = 1.0
    return (lo, hi)


# --------------------------------------------------------------------------- #
#  Determinism                                                                #
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def seeded_context(seed):
    """Temporarily seed NumPy and Torch, restoring caller RNG state after."""
    if seed is None:
        yield
        return
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


# --------------------------------------------------------------------------- #
#  Baselines                                                                  #
# --------------------------------------------------------------------------- #

class RandomPlayer:
    """Uniform-random legal move, driven by a seeded RNG."""

    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def move(self, board):
        moves = list(board.legal_moves)
        if not moves:
            return None
        return moves[int(self.rng.integers(len(moves)))]


class GreedyPlayer:
    """One-ply material-greedy: play the legal move maximising the material
    difference from the mover's perspective (captured piece value + promotion
    gain).  Ties resolve to the first move in python-chess generation order,
    so the baseline is fully deterministic."""

    name = "greedy"

    def move(self, board):
        moves = list(board.legal_moves)
        if not moves:
            return None
        before = self._material(board)
        sign = 1 if board.turn == chess.WHITE else -1
        best, best_score = moves[0], float("-inf")
        for m in moves:
            bb = board.copy()
            bb.push(m)
            delta = sign * (self._material(bb) - before)
            if delta > best_score:
                best, best_score = m, delta
        return best

    @staticmethod
    def _material(board):
        return sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values())


class NetPlayer:
    """ChessNet + MCTS (temperature 0).  Search is wrapped in a seeded context
    so runs are reproducible for a fixed seed."""

    name = "net"

    def __init__(self, net, cfg, sims=None, seed=0):
        self.net = net.to(cfg.device)  # MCTS moves inputs to cfg.device; net must follow
        self.cfg = cfg
        self.sims = sims if sims is not None else cfg.num_simulations
        self.seed = seed
        self.mcts = MCTS(net, cfg)

    def move(self, board):
        with seeded_context(self.seed):
            pi = self.mcts.search(board, temperature=0.0, num_sims=self.sims)
        if not pi:
            return None
        return max(pi, key=pi.get)


# --------------------------------------------------------------------------- #
#  Match play                                                                 #
# --------------------------------------------------------------------------- #

def _terminal_result(board):
    """(is_terminal, white_result) — same rules as arena.py/selfplay."""
    if board.is_checkmate():
        return True, -1.0 if board.turn == chess.WHITE else 1.0
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_repetition(3) or board.is_fifty_moves()):
        return True, 0.0
    return False, None


def play_game(white, black, cfg, start_fen=None, seed=None, max_moves=None):
    """Play one game; return (score_from_white_perspective, info).

    score: 1.0 white win, 0.5 draw, 0.0 black win.  `info` carries
    {result: 'white'|'black'|'draw', plies, fen, reason?}.
    """
    board = chess.Board(start_fen) if start_fen else chess.Board()
    cap = max_moves if max_moves is not None else cfg.max_game_length
    plies = 0
    with seeded_context(seed):
        while True:
            if len(board.move_stack) >= cap:
                return 0.5, {"result": "draw", "plies": plies,
                             "fen": board.fen(), "reason": "length"}
            terminal, white_result = _terminal_result(board)
            if terminal:
                if white_result > 0.0:
                    return 1.0, {"result": "white", "plies": plies, "fen": board.fen()}
                if white_result < 0.0:
                    return 0.0, {"result": "black", "plies": plies, "fen": board.fen()}
                return 0.5, {"result": "draw", "plies": plies, "fen": board.fen()}
            player = white if board.turn == chess.WHITE else black
            mv = player.move(board)
            if mv is None or mv not in board.legal_moves:
                return 0.5, {"result": "draw", "plies": plies,
                             "fen": board.fen(), "reason": "no_move"}
            board.push(mv)
            plies += 1


def play_match(a, b, cfg, num_games=2, seed=None):
    """Paired-colour match: `a` plays White on even games, `b` on odd games.

    Returns aggregate scores (win = 1, draw = 0.5) plus a Wilson confidence
    interval per side over the `num_games` games.
    """
    wins_a = wins_b = draws = 0
    games = []
    for i in range(num_games):
        white, black = (a, b) if i % 2 == 0 else (b, a)
        gseed = None if seed is None else seed + i
        score, info = play_game(white, black, cfg, seed=gseed)
        if score > 0.5:
            if i % 2 == 0:
                wins_a += 1
            else:
                wins_b += 1
        elif score < 0.5:
            if i % 2 == 0:
                wins_b += 1
            else:
                wins_a += 1
        else:
            draws += 1
        games.append({
            "index": i, "white": white.name, "black": black.name,
            "result": info["result"], "plies": info["plies"],
        })
    score_a = wins_a + 0.5 * draws
    score_b = wins_b + 0.5 * draws
    ci_a = wilson_interval(score_a, num_games)
    ci_b = wilson_interval(score_b, num_games)
    return {
        "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "score_a": score_a, "score_b": score_b,
        "ci_a": {"low": round(ci_a[0], 4), "high": round(ci_a[1], 4), "z": 1.96},
        "ci_b": {"low": round(ci_b[0], 4), "high": round(ci_b[1], 4), "z": 1.96},
        "games": games,
    }


# --------------------------------------------------------------------------- #
#  Tactics                                                                    #
# --------------------------------------------------------------------------- #

def run_tactics(player, cfg, seed=None):
    """Run the tactical suite with `player` (a move(board)->Move object).

    Returns {"mate_in_one": [...], "avoid_mate": [...]} where each entry is
    {name, fen, move (uci or None), pass (bool)}.  Pass for mate-in-one means
    the played move is checkmate; pass for avoid-mate means the played move
    leaves the opponent with no mate-in-one.  Deterministic when the player is
    freshly constructed with a fixed seed.
    """
    mate = []
    for name, fen in MATE_IN_ONE:
        b = chess.Board(fen)
        mv = player.move(b)
        ok = False
        if mv is not None:
            bb = b.copy()
            bb.push(mv)
            ok = bb.is_checkmate()
        mate.append({"name": name, "fen": fen,
                     "move": mv.uci() if mv else None, "pass": ok})

    avoid = []
    for name, fen in AVOID_MATE:
        b = chess.Board(fen)
        mv = player.move(b)
        ok = False
        if mv is not None:
            bb = b.copy()
            bb.push(mv)
            ok = not any(_is_mate_in_one(bb, x) for x in bb.legal_moves)
        avoid.append({"name": name, "fen": fen,
                      "move": mv.uci() if mv else None, "pass": ok})
    return {"mate_in_one": mate, "avoid_mate": avoid}


def _is_mate_in_one(board, move):
    bb = board.copy()
    bb.push(move)
    return bb.is_checkmate()


# --------------------------------------------------------------------------- #
#  Full evaluation run                                                        #
# --------------------------------------------------------------------------- #

def _cfg_dict(cfg):
    return {
        "device": str(cfg.device),
        "num_input_planes": int(cfg.num_input_planes),
        "policy_size": int(cfg.policy_size),
        "num_res_blocks": int(cfg.num_res_blocks),
        "num_filters": int(cfg.num_filters),
        "num_simulations": int(cfg.num_simulations),
        "c_puct": float(cfg.c_puct),
        "max_game_length": int(cfg.max_game_length),
    }


def _build_net(cfg, seed, load_best):
    """Fresh deterministic net, optionally initialised from best.pt."""
    with seeded_context(seed):
        net = ChessNet(cfg).eval()
    source = "untrained (random init)"
    if load_best:
        best = os.path.join(cfg.checkpoint_dir, "best.pt")
        if os.path.exists(best):
            try:
                state = torch.load(best, map_location=cfg.device, weights_only=True)
                net.load_state_dict(state)
                source = "best.pt"
            except Exception as e:  # noqa: BLE001  (shape mismatch etc.)
                print(f"[evaluate] best.pt ignored ({e}); using random init",
                      file=sys.stderr)
    return net, source


def evaluate(cfg, seed=42, num_games=2, tactics_sims=40,
             players=("random", "greedy", "net"), load_best=True):
    """Run the full fixed evaluation; returns the JSON-serialisable dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    factory = {
        "random": lambda: RandomPlayer(seed=seed),
        "greedy": lambda: GreedyPlayer(),
    }
    net, net_source = None, None
    if "net" in players:
        net, net_source = _build_net(cfg, seed, load_best)
        factory["net"] = lambda: NetPlayer(net, cfg, sims=tactics_sims, seed=seed)

    results = []
    unique_matches = []
    for i, wa in enumerate(players):
        for j, ba in enumerate(players):
            if j <= i:
                continue  # unordered pairs, both colours
            match = play_match(factory[wa](), factory[ba](),
                               cfg, num_games=num_games, seed=seed)
            unique_matches.append((wa, ba, match, num_games))
            # Paired-colour result records: every pairing appears TWICE, once
            # per colour assignment (white=A/black=B and the mirror).  Each
            # record carries the match's aggregate scores from the listed
            # player's perspective; play_match alternates colours internally,
            # so both orientations share the same games_detail.
            results.append({
                "white": wa, "black": ba,
                "games": num_games,
                "score_white": match["score_a"],
                "score_black": match["score_b"],
                "ci_white": match["ci_a"],
                "ci_black": match["ci_b"],
                "games_detail": match["games"],
            })
            results.append({
                "white": ba, "black": wa,
                "games": num_games,
                "score_white": match["score_b"],
                "score_black": match["score_a"],
                "ci_white": match["ci_b"],
                "ci_black": match["ci_a"],
                "games_detail": match["games"],
            })

    tactics = {}
    for pname in players:
        with seeded_context(seed):
            tactics[pname] = run_tactics(factory[pname](), cfg, seed=seed)

    total_games = sum(games for _, _, _, games in unique_matches)
    summary_scores = {p: 0.0 for p in players}
    games_by_player = {p: 0 for p in players}
    for player_a, player_b, match, games in unique_matches:
        summary_scores[player_a] += match["score_a"]
        summary_scores[player_b] += match["score_b"]
        games_by_player[player_a] += games
        games_by_player[player_b] += games
    summary = {
        "total_games": total_games,
        "score": {p: summary_scores[p] for p in players},
        "games_by_player": games_by_player,
        "score_rate": {
            p: round(summary_scores[p] / games_by_player[p], 4)
            if games_by_player[p] else 0.0
            for p in players
        },
        "net_source": net_source,
    }

    return {
        "seed": seed,
        "config": _cfg_dict(cfg),
        "results": results,
        "tactics": tactics,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--games", type=int, default=4,
                    help="games per colour pairing (2 = one each colour)")
    ap.add_argument("--sims", type=int, default=100,
                    help="MCTS simulations for the net baseline")
    ap.add_argument("--players", default="random,greedy,net",
                    help="comma-separated baseline names to include")
    ap.add_argument("--no-best", action="store_true",
                    help="do not initialise the net baseline from best.pt")
    ap.add_argument("--out", default=None, help="write JSON to this file")
    args = ap.parse_args(argv)

    cfg = Config()
    players = tuple(p.strip() for p in args.players.split(",") if p.strip())
    t0 = time.time()
    out = evaluate(cfg, seed=args.seed, num_games=args.games,
                   tactics_sims=args.sims, players=players,
                   load_best=not args.no_best)
    out["elapsed_seconds"] = round(time.time() - t0, 2)
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"[evaluate] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
