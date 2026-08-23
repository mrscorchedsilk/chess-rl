"""TDD tests for Phase 2: deterministic paired-opening arena diversity.

Written BEFORE the implementation. They pin the required behaviour:

  A. deterministic suite: same seed + config -> identical openings
  B. seed sensitivity: different seeds -> different suites
  C. legal sequences: every generated UCI move is legal when replayed
  D. suite diversity: N pairs -> N distinct openings
  E. paired colors: each opening played exactly twice, colors swapped
  F. history preservation: opening moves pushed onto move_stack before MCTS
  G. arena game count: wins + losses + draws == arena_games
  H. even-game validation: odd arena_games raises ValueError
  I. no root noise: arena search still passes add_root_noise=False
  J. aggregate scoring: standard (wins + 0.5*draws)/games preserved

Run:  .venv/bin/python -m pytest tests/test_arena_openings.py -q
"""

import os
import sys

import chess
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model import ChessNet
import arena


def _net(seed=0):
    torch.manual_seed(seed)
    cfg = Config()
    cfg.device = "cpu"
    cfg.num_simulations = 4
    net = ChessNet(cfg).to("cpu").eval()
    return net, cfg


# --------------------------------------------------------------------------- #
#  A/B/C/D. generate_arena_openings                                            #
# --------------------------------------------------------------------------- #

def test_arena_openings_deterministic_same_seed():
    a = arena.generate_arena_openings(10, 8, 424242)
    b = arena.generate_arena_openings(10, 8, 424242)
    assert a == b


def test_arena_openings_seed_sensitive():
    a = arena.generate_arena_openings(10, 8, 424242)
    b = arena.generate_arena_openings(10, 8, 424243)
    assert a != b


def test_arena_openings_legal_when_replayed():
    openings = arena.generate_arena_openings(10, 8, 424242)
    for seq in openings:
        board = chess.Board()
        for uci in seq:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, f"illegal opening move {uci}"
            board.push(move)


def test_arena_openings_distinct():
    openings = arena.generate_arena_openings(10, 8, 424242)
    assert len(openings) == 10
    assert len(set(tuple(s) for s in openings)) == 10, \
        "opening pairs must be distinct"


# --------------------------------------------------------------------------- #
#  E. paired colors                                                           #
# --------------------------------------------------------------------------- #

def test_arena_plays_each_opening_twice_with_swapped_colors(monkeypatch):
    net_a, cfg = _net(0)
    net_b, _ = _net(1)
    cfg.arena_games = 20
    cfg.arena_simulations = 4
    cfg.max_game_length = 60
    cfg.arena_seed = 424242
    cfg.arena_opening_plies = 8

    calls = []

    def spy(mcts_white, mcts_black, cfg_, num_sims, opening_moves):
        white_is_a = mcts_white.net is net_a
        calls.append((white_is_a, tuple(opening_moves)))
        return 0.0  # draw

    monkeypatch.setattr(arena, "_play_arena_game", spy)
    result = arena.play_match(net_a, net_b, cfg, num_games=20)

    assert result == {"a": 0, "b": 0, "draws": 20}
    assert len(calls) == 20
    by_opening = {}
    for white_is_a, opening in calls:
        by_opening.setdefault(opening, []).append(white_is_a)
    assert len(by_opening) == 10, "expected 10 distinct openings"
    for opening, whites in by_opening.items():
        assert len(whites) == 2, f"opening {opening} not played exactly twice"
        assert sorted(whites) == [False, True], \
            f"opening {opening} colors not swapped"


# --------------------------------------------------------------------------- #
#  F. history preservation                                                   #
# --------------------------------------------------------------------------- #

class _FakeMCTS:
    def __init__(self):
        self.seen = []

    def search(self, board, temperature, num_sims, add_root_noise):
        self.seen.append(([m.uci() for m in board.move_stack], add_root_noise))
        moves = list(board.legal_moves)
        if not moves:
            return {}
        return {moves[0]: 1.0}


def test_arena_opening_moves_pushed_before_search():
    cfg = Config()
    cfg.arena_root_noise = False
    cfg.max_game_length = 400
    w = _FakeMCTS()
    b = _FakeMCTS()
    opening = ["e2e4", "e7e5", "g1f3"]
    arena._play_arena_game(w, b, cfg, num_sims=4, opening_moves=opening)
    # the first search must already see the replayed opening on the stack
    assert w.seen and w.seen[0][0][:len(opening)] == opening, \
        "opening moves must be pushed before MCTS begins"


# --------------------------------------------------------------------------- #
#  G. arena game count                                                       #
# --------------------------------------------------------------------------- #

def test_arena_game_count_sum_matches_arena_games():
    net, cfg = _net(0)
    cfg.arena_games = 4
    cfg.arena_simulations = 2
    cfg.max_game_length = 20
    cfg.arena_seed = 424242
    cfg.arena_opening_plies = 6
    result = arena.play_match(net, net, cfg, num_games=4)
    assert result["a"] + result["b"] + result["draws"] == 4


# --------------------------------------------------------------------------- #
#  H. even-game validation                                                   #
# --------------------------------------------------------------------------- #

def test_arena_odd_games_raises_value_error():
    net, cfg = _net(0)
    cfg.arena_games = 5
    with pytest.raises(ValueError):
        arena.play_match(net, net, cfg, num_games=5)


# --------------------------------------------------------------------------- #
#  I. no root noise                                                          #
# --------------------------------------------------------------------------- #

def test_arena_no_root_noise_with_openings():
    net, cfg = _net(0)
    cfg.arena_root_noise = False
    cfg.max_game_length = 400
    w = _FakeMCTS()
    b = _FakeMCTS()
    arena._play_arena_game(w, b, cfg, num_sims=4, opening_moves=["d2d4", "d7d5"])
    for searcher in (w, b):
        for _, add_root_noise in searcher.seen:
            assert add_root_noise is False, "arena search must not add root noise"


# --------------------------------------------------------------------------- #
#  J. aggregate scoring compatibility                                        #
# --------------------------------------------------------------------------- #

def test_arena_scoring_format_unchanged():
    # play_match still returns exactly {"a", "b", "draws"} so train._arena_gate's
    # standard score (wins + 0.5*draws)/games is unchanged.
    net, cfg = _net(0)
    cfg.arena_games = 4
    cfg.arena_simulations = 2
    cfg.max_game_length = 20
    result = arena.play_match(net, net, cfg, num_games=4)
    assert set(result) == {"a", "b", "draws"}
    wins, losses, draws = result["a"], result["b"], result["draws"]
    games = wins + losses + draws
    score = (wins + 0.5 * draws) / games
    assert 0.0 <= score <= 1.0
