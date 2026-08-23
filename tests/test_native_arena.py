"""Ticket C — native-arena correctness tests (docs/tickets.md, design §9).

Locks the native-arena adapter (``native_arena.py``, Ticket B) against three
invariants:

1. **Determinism golden** — a full 20-game / 40-sim match driven by the shared
   deterministic fake evaluator (``benchmarks/native_mcts.FakeEvaluator``) at
   the production suite settings (``arena_seed=424242``, 8 opening plies,
   temperature 0, no root noise) must produce byte-identical per-game move
   transcripts on two independent runs and match a recorded golden digest.
2. **Parity vs ``mcts.py``** — the native MCTS core and the python reference
   must select the identical temperature-0 move on every ``BENCH_POSITIONS``
   position where the most-visited move is strictly unique.  Where the top
   visit count ties, the engines may *legitimately* select different moves
   (native: first child in ascending action-index / CSR order; python:
   ``np.argmax`` over python-chess ``legal_moves`` order — the documented
   divergence boundary, design §8/§9), so on tied positions we assert each
   engine's documented tie-break rule and the equality of the max-visit sets
   instead of the selected move or the visit distribution.
3. **Adjudication** — python-chess game-level terminal adjudication is
   preserved end-to-end: mate (both colors), threefold repetition and the
   length cap all yield the correct ``white_result``.

All tests are CPU-only and fully deterministic (no RNG is ever drawn: the
native MCTS seed is inert because ``dirichlet_epsilon=0`` and the fake
evaluator is a pure function of its inputs).

Run::

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_native_arena.py -q
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import chess
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmarks.native_mcts as bm  # noqa: E402  (shared fixture home)
import native_arena  # noqa: E402
from arena import _terminal_result  # noqa: E402
from config import Config  # noqa: E402
from encoding import move_to_index  # noqa: E402

# Scholar's mate (1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7#) — 7 plies, black mated.
SCHOLARS_MATE = ("e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7")
# 4x knight shuffle — the start position recurs for the 3rd time at ply 8.
THREEFOLD_LINE = ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8")


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _arena_cfg(**overrides) -> Config:
    """A Config with the arena-relevant fields forced to production values."""
    cfg = Config()
    cfg.device = "cpu"
    cfg.arena_games = 20
    cfg.arena_simulations = 40
    cfg.max_game_length = 400
    cfg.arena_seed = 424242
    cfg.arena_opening_plies = 8
    cfg.arena_root_noise = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _transcripts_hash(transcripts) -> str:
    """Stable digest of the ordered per-game move transcripts."""
    h = hashlib.blake2b(digest_size=16)
    for game in transcripts:
        for uci in game:
            h.update(uci.encode("utf-8"))
            h.update(b"\x00")
        h.update(b"\xff")
    return h.hexdigest()


def _run_match_with_transcripts(cfg, num_games, evaluate_a, evaluate_b):
    """Run the REAL native arena match and capture per-game move transcripts.

    The production loop is NOT replaced: ``_play_native_game`` and
    ``_select_move`` are wrapped so every game records ``opening + chosen
    moves`` exactly as the loop pushed them.  Returns ``(result, transcripts)``.
    """
    orig_play = native_arena._play_native_game
    orig_select = native_arena._select_move
    games = []           # list of [opening(list), chosen(list)]
    current = None

    def select_spy(policy):
        move = orig_select(policy)
        current[1].append(move)
        return move

    def play_spy(mcts_white, mcts_black, evaluate_white, evaluate_black,
                 cfg_, num_sims, opening_moves=(), max_batch=256):
        nonlocal current
        current = [list(opening_moves), []]
        games.append(current)
        return orig_play(mcts_white, mcts_black, evaluate_white, evaluate_black,
                         cfg_, num_sims, opening_moves=opening_moves,
                         max_batch=max_batch)

    native_arena._play_native_game = play_spy
    native_arena._select_move = select_spy
    try:
        result = native_arena.play_match(
            None, None, cfg, num_games,
            evaluate_a=evaluate_a, evaluate_b=evaluate_b,
        )
    finally:
        native_arena._play_native_game = orig_play
        native_arena._select_move = orig_select
    transcripts = [tuple(opening) + tuple(chosen) for opening, chosen in games]
    return result, transcripts


def _native_policy(fen, history, num_sims):
    """Temperature-0 native search -> (selected uci, {uci: visit_count}).

    ``max_batch=32`` matches ``mcts.py``'s ``batch_size`` (and the
    ``benchmarks/native_mcts.py`` harness ``BATCH_SIZE``), so the two engines
    batch simulations identically and only their child iteration order differs
    — the engine-level parity comparison is apples-to-apples.  (The arena's
    production driver uses ``max_batch=256``; that batching is covered by the
    determinism golden, test 1.)
    """
    mcts = bm.make_native_mcts(
        num_simulations=num_sims, c_puct=1.25, virtual_loss=3.0,
        dirichlet_epsilon=0.0, seed=42,
    )
    evaluator = bm.FakeEvaluator()
    policy = native_arena._run_native_search(
        mcts, fen, history, evaluator.logits_and_values, num_sims, max_batch=32,
    )
    counts = {uci: count for uci, count in mcts.root_visit_counts()}
    return native_arena._select_move(policy), counts


def _python_policy(fen, history, num_sims):
    """mcts.py temperature-0 search -> (selected uci, {uci: visit_count})."""
    board = chess.Board(fen)
    for uci in history:
        board.push_uci(uci)
    reference = bm.make_python_mcts(
        bm.FakeEvaluator(), num_simulations=num_sims, c_puct=1.25,
        virtual_loss=3.0,
    )
    pi = reference.search(board, temperature=0.0, num_sims=num_sims,
                          add_root_noise=False)
    counts = {m.uci(): reference.root.children[m].N
              for m in reference.root.children}
    return max(pi, key=pi.get).uci(), counts


# --------------------------------------------------------------------------- #
# 1. Determinism golden                                                       #
# --------------------------------------------------------------------------- #

# Recorded from a full 20-game / 40-sim match with the shared FakeEvaluator at
# arena_seed=424242 / 8 opening plies / temp 0 / no root noise.  Deterministic
# by construction (no RNG is ever drawn), so both digests are stable across
# runs.
GOLDEN_TRANSCRIPT_HASH = "971b2c2eef88b803a30989ff2206abb9"
GOLDEN_RESULT = {"a": 3, "b": 3, "draws": 14}


def test_native_arena_deterministic_fixed_seed():
    cfg = _arena_cfg()
    evaluator_a = bm.FakeEvaluator()
    evaluator_b = bm.FakeEvaluator()

    result_1, transcripts_1 = _run_match_with_transcripts(
        cfg, 20, evaluator_a.logits_and_values, evaluator_b.logits_and_values)
    result_2, transcripts_2 = _run_match_with_transcripts(
        cfg, 20, evaluator_a.logits_and_values, evaluator_b.logits_and_values)

    assert len(transcripts_1) == len(transcripts_2) == 20
    # Byte-identical ordered transcripts and identical aggregate result.
    assert transcripts_1 == transcripts_2
    assert result_1 == result_2
    # Golden pins: recorded transcript digest + recorded {a, b, draws}.
    assert _transcripts_hash(transcripts_1) == GOLDEN_TRANSCRIPT_HASH
    assert result_1 == GOLDEN_RESULT


# --------------------------------------------------------------------------- #
# 2. Paired openings + color swap                                             #
# --------------------------------------------------------------------------- #

def test_native_arena_preserves_paired_openings_and_color_swap(monkeypatch):
    cfg = _arena_cfg()
    calls = []

    def spy(mcts_white, mcts_black, evaluate_white, evaluate_black,
            cfg_, num_sims, opening_moves=(), max_batch=256):
        # Game A is candidate (mcts_a) White, Game B is champion (mcts_b)
        # White; record which MCTS object played White for this opening.
        calls.append((id(mcts_white), id(mcts_black), tuple(opening_moves)))
        return 0.0  # draw, so the match loop is exercised without search

    monkeypatch.setattr(native_arena, "_play_native_game", spy)
    evaluator = bm.FakeEvaluator()
    result = native_arena.play_match(
        None, None, cfg, num_games=20,
        evaluate_a=evaluator.logits_and_values,
        evaluate_b=evaluator.logits_and_values,
    )

    assert result == {"a": 0, "b": 0, "draws": 20}
    assert len(calls) == 20, "20 games expected (10 openings x 2 colors)"
    by_opening = {}
    for white_id, black_id, opening in calls:
        by_opening.setdefault(opening, []).append((white_id, black_id))
    assert len(by_opening) == 10, "expected 10 distinct openings"
    for opening, entries in by_opening.items():
        assert len(entries) == 2, f"opening {opening} not played exactly twice"
        (w1, b1), (w2, b2) = entries
        assert w1 != b1 and w2 != b2
        # Colors swapped: the object that was White in game A is Black in game B.
        assert w1 == b2 and b1 == w2, \
            f"opening {opening} colors not swapped"


# --------------------------------------------------------------------------- #
# 3. Result contract                                                          #
# --------------------------------------------------------------------------- #

def test_native_arena_result_contract():
    cfg = _arena_cfg(arena_simulations=4, max_game_length=60)
    evaluator = bm.FakeEvaluator()
    result = native_arena.play_match(
        None, None, cfg, num_games=4,
        evaluate_a=evaluator.logits_and_values,
        evaluate_b=evaluator.logits_and_values,
    )
    # Exactly the {"a", "b", "draws"} keys (no extras), summing to num_games.
    assert set(result) == {"a", "b", "draws"}
    assert result["a"] + result["b"] + result["draws"] == 4
    # Odd num_games raises ValueError before any game is played.
    with pytest.raises(ValueError):
        native_arena.play_match(
            None, None, cfg, num_games=5,
            evaluate_a=evaluator.logits_and_values,
            evaluate_b=evaluator.logits_and_values,
        )


# --------------------------------------------------------------------------- #
# 4. Native MCTS vs mcts.py parity                                            #
# --------------------------------------------------------------------------- #

def test_native_mcts_matches_python_reference_fixed_seed():
    num_sims = 40  # arena setting; the design verified b1c3 at 40 sims
    # BENCH_POSITIONS[0] IS the standard start position, so iterating the
    # corpus covers "START_FEN + benchmarks/native_mcts.BENCH_POSITIONS".
    assert bm.BENCH_POSITIONS[0][0] == native_arena.START_FEN
    for fen, history in bm.BENCH_POSITIONS:
        native_move, native_counts = _native_policy(fen, history, num_sims)
        py_move, py_counts = _python_policy(fen, history, num_sims)

        max_n = max(native_counts.values())
        max_p = max(py_counts.values())
        # Temperature 0 is one-hot on a most-visited move in both engines.
        assert native_counts[native_move] == max_n
        assert py_counts[py_move] == max_p

        tied_n = {u for u, c in native_counts.items() if c == max_n}
        tied_p = {u for u, c in py_counts.items() if c == max_p}

        if len(tied_n) == 1 and len(tied_p) == 1:
            # Strictly unique most-visited move in both engines: the
            # temperature-0 move is identical (parity).
            assert native_move == py_move == next(iter(tied_n)) == next(iter(tied_p)), \
                (fen, history, native_move, py_move)
        else:
            # Documented tie-break boundary (design §9): the engines search
            # identically (identical max-visit SET) but order children
            # differently, so a top tie may select different moves — e.g. the
            # bare-rooks position picks a1a2 (native) vs e1f2 (python) with
            # the same 14-move tied set.  Assert the selected move via each
            # engine's documented rule, never the visit distribution.
            assert tied_n == tied_p, (fen, history, tied_n, tied_p)
            # native policy(0.0): ties go to the first child in ascending
            # action-index (CSR) order — np.argmax semantics (mcts.cpp).
            native_first = min(tied_n, key=lambda u: move_to_index(chess.Move.from_uci(u)))
            assert native_move == native_first, (fen, history, native_move, tied_n)
            # mcts.py._get_policy: np.argmax over python-chess legal_moves order.
            board = chess.Board(fen)
            for uci in history:
                board.push_uci(uci)
            py_first = next(u for u in (m.uci() for m in board.legal_moves)
                            if u in tied_p)
            assert py_move == py_first, (fen, history, py_move, tied_p)


# --------------------------------------------------------------------------- #
# 5. Terminal adjudication                                                    #
# --------------------------------------------------------------------------- #

def test_native_arena_terminal_adjudication():
    # (a) mate for White: _terminal_result sees a checkmate won by White.
    board = chess.Board()
    for uci in SCHOLARS_MATE:
        board.push_uci(uci)
    assert board.is_checkmate()
    assert _terminal_result(board) == (True, 1.0)

    # (b) mate for Black: mirror fixture (black Ra8 + Kg3 vs white Kh1).
    board = chess.Board("r7/8/8/8/8/6k1/8/7K b - - 0 1")
    board.push_uci("a8a1")
    assert board.is_checkmate()
    assert _terminal_result(board) == (True, -1.0)

    # (c) threefold repetition: the 8-ply knight shuffle claims a draw.
    board = chess.Board()
    for uci in THREEFOLD_LINE:
        board.push_uci(uci)
    assert board.is_repetition(3)
    assert _terminal_result(board) == (True, 0.0)

    # (d) the game loop adjudicates before any search:
    #     - mate opening -> white_result 1.0, zero searches;
    #     - threefold opening -> draw 0.0, zero searches;
    #     - opening of exactly max_game_length plies -> length cap draw 0.0,
    #       even though the capped position is checkmate (cap checked first).
    cfg = _arena_cfg()
    mcts = bm.make_native_mcts(num_simulations=4, dirichlet_epsilon=0.0)
    evaluator = bm.FakeEvaluator()
    assert native_arena._play_native_game(
        mcts, mcts, evaluator.logits_and_values, evaluator.logits_and_values,
        cfg, num_sims=4, opening_moves=SCHOLARS_MATE,
    ) == 1.0
    assert native_arena._play_native_game(
        mcts, mcts, evaluator.logits_and_values, evaluator.logits_and_values,
        cfg, num_sims=4, opening_moves=THREEFOLD_LINE,
    ) == 0.0
    capped = _arena_cfg(max_game_length=len(SCHOLARS_MATE))
    assert native_arena._play_native_game(
        mcts, mcts, evaluator.logits_and_values, evaluator.logits_and_values,
        capped, num_sims=4, opening_moves=SCHOLARS_MATE,
    ) == 0.0
