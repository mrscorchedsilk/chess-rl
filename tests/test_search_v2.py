"""Sprint B search tests — MCTS / arena / self-play behaviour over the 4672 action space.

Strict-TDD RED set written BEFORE the implementation: root-noise gating (arena
determinism), claimable-draw terminals (search == self-play == arena), policy
probability mass, mate-in-one, and arena accounting.

Run with:  .venv/bin/python -m pytest tests/test_search_v2.py -q
"""
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
import encoding
from model import ChessNet
from mcts import MCTS
import selfplay
import arena


def _net(seed=0):
    torch.manual_seed(seed)
    cfg = Config()
    cfg.device = "cpu"
    net = ChessNet(cfg).to("cpu").eval()
    return net, cfg


def _threefold_claimable_board():
    """Four knight shuffles: starting position has occurred three times."""
    b = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"):
        b.push_uci(uci)
    assert b.is_repetition(3)
    assert b.can_claim_threefold_repetition()
    assert not b.is_game_over()            # claimable, not an automatic draw
    assert b.is_game_over(claim_draw=True)
    return b


MATE_IN_ONE_FEN = "7k/6pp/8/8/8/8/8/R6K w - - 0 1"   # Ra1-a8#
FOOLS_MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


# ------------------------------------------------------------------ root noise

def test_mcts_root_noise_flag_gates_dirichlet():
    net, cfg = _net()
    mcts = MCTS(net, cfg)
    calls = []
    mcts._apply_dirichlet_noise = lambda node: calls.append(node)
    mcts.search(chess.Board(), temperature=0.0, num_sims=8, add_root_noise=False)
    assert calls == [], "add_root_noise=False must not apply Dirichlet noise"
    mcts.search(chess.Board(), temperature=0.0, num_sims=8, add_root_noise=True)
    assert len(calls) == 1, "add_root_noise=True must apply Dirichlet noise"


def test_mcts_root_noise_off_is_deterministic():
    net, cfg = _net()
    mcts = MCTS(net, cfg)
    b = chess.Board()
    pi1 = mcts.search(b, temperature=0.0, num_sims=24, add_root_noise=False)
    pi2 = mcts.search(b.copy(), temperature=0.0, num_sims=24, add_root_noise=False)
    assert pi1 == pi2
    move = next(iter(pi1))
    assert move in b.legal_moves


def test_arena_no_root_noise():
    """Arena matches must never apply Dirichlet root exploration."""
    net, cfg = _net()
    cfg.arena_simulations = 8
    cfg.max_game_length = 60  # bound the arena games for CPU speed
    orig = MCTS._apply_dirichlet_noise
    calls = []

    def spy(self, node):
        calls.append(node)
        return orig(self, node)

    MCTS._apply_dirichlet_noise = spy
    try:
        result = arena.play_match(net, net, cfg, num_games=2)
    finally:
        MCTS._apply_dirichlet_noise = orig
    assert calls == [], "arena search applied Dirichlet root noise"
    assert set(result) == {"a", "b", "draws"}
    assert result["a"] + result["b"] + result["draws"] == 2


def test_arena_deterministic_same_net():
    """No-noise + temperature 0 => repeated arena matches are identical."""
    net, cfg = _net()
    cfg.arena_simulations = 8
    cfg.max_game_length = 60  # bound the arena games for CPU speed
    r1 = arena.play_match(net, net, cfg, num_games=2)
    r2 = arena.play_match(net, net, cfg, num_games=2)
    assert r1 == r2


# ------------------------------------------------------------- draw handling

def test_mcts_claimable_draw_root_is_terminal():
    """A claimable threefold-repetition root ends the search with an empty policy."""
    net, cfg = _net()
    b = _threefold_claimable_board()
    mcts = MCTS(net, cfg)
    pi = mcts.search(b, temperature=0.0, num_sims=16, add_root_noise=False)
    assert pi == {}, "claimable draw root must be terminal (empty policy)"
    assert MCTS._terminal_value(b) == 0.0


def test_terminal_rules_agree_across_search_selfplay_arena():
    # claimable threefold repetition -> draw everywhere
    b = _threefold_claimable_board()
    assert selfplay._terminal_result(b) == (True, 0.0)
    assert arena._terminal_result(b) == (True, 0.0)
    assert MCTS._terminal_value(b) == 0.0
    # checkmate -> the mated side loses; here WHITE is mated, so white_result = -1.0
    mate = chess.Board(FOOLS_MATE_FEN)
    assert mate.is_checkmate()
    assert MCTS._terminal_value(mate) == -1.0
    assert selfplay._terminal_result(mate) == (True, -1.0)
    assert arena._terminal_result(mate) == (True, -1.0)
    # stalemate -> draw
    stale = chess.Board(STALEMATE_FEN)
    assert stale.is_stalemate()
    assert MCTS._terminal_value(stale) == 0.0
    assert selfplay._terminal_result(stale) == (True, 0.0)
    assert arena._terminal_result(stale) == (True, 0.0)


def test_mcts_terminal_leaf_value_backprop():
    """A mate-in-one child must be worth +1 at the root (visit the mate branch)."""
    net, cfg = _net()
    mcts = MCTS(net, cfg)
    b = chess.Board(MATE_IN_ONE_FEN)
    pi = mcts.search(b, temperature=0.0, num_sims=64, add_root_noise=False)
    move = max(pi, key=pi.get)
    assert move.uci() == "a1a8", move.uci()


# ----------------------------------------------------------- mass / legality

def test_mcts_policy_probability_mass_and_legality():
    net, cfg = _net()
    mcts = MCTS(net, cfg)
    b = chess.Board("8/PPP5/8/8/8/8/8/k6K w - - 0 1")  # promotions + king moves
    pi = mcts.search(b, temperature=1.0, num_sims=32, add_root_noise=False)
    legal = set(b.legal_moves)
    assert set(pi) == legal
    assert abs(sum(pi.values()) - 1.0) < 1e-6
    for m, pr in pi.items():
        assert pr >= 0.0 and m in legal
    vec = encoding.policy_to_vector(pi)
    assert vec.shape == (4672,)
    assert abs(float(vec.sum()) - 1.0) < 1e-5
    assert int((vec > 0).sum()) == len(legal)  # unique plane per move, no shared mass


# ------------------------------------------------------------------ mate in 1

def test_mcts_mate_in_one():
    net, cfg = _net()
    mcts = MCTS(net, cfg)
    b = chess.Board(MATE_IN_ONE_FEN)
    pi = mcts.search(b, temperature=0.0, num_sims=800, add_root_noise=False)
    move = max(pi, key=pi.get)
    assert move.uci() == "a1a8", move.uci()
