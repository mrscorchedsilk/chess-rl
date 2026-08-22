"""Differential tests: python-chess 1.999 + encoding.py vs chess_rl_native.

Strict TDD: written against the reference before the native policy map and
encoder exist (RED), then the native pieces are implemented to make them pass
(GREEN).  Routine pytest stays small (<= 1000 positions); the full 100k run is
a CLI exercise, never part of pytest.

Covers two deliverables:

1. Action map -- ``move_to_index`` / ``index_to_move`` round-trip recovers the
   exact UCI for every legal move of every directed fixture (promotions,
   castling, en passant, underpromotions), plus ``legal_move_indices`` and
   ``policy_to_vector`` parity.

2. Encoder -- ``Position.encode`` / ``encode_fen`` are ``np.array_equal`` to
   ``encoding.encode_board`` on directed fixtures and a deterministic random
   corpus, including the *raw* en-passant plane (plane 101) which preserves
   the verbatim FEN field / last double-push even when no legal capture
   exists.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import chess
import chess_rl_native

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmarks.native_policy_encoder_parity as harness  # noqa: E402
import encoding  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# Constants / schema
# ---------------------------------------------------------------------------


def test_schema_and_defaults():
    assert harness.SCHEMA == "native_policy_encoder_parity/v1"
    assert harness.DEFAULT_SEED == 0x20260822
    assert harness.DEFAULT_POSITIONS == 100_000
    assert harness.START_FEN == START_FEN
    assert harness.HISTORY_STEPS == 8


def test_policy_constants_exposed_and_consistent():
    assert chess_rl_native.POLICY_PLANES == 73
    assert chess_rl_native.POLICY_SIZE == 4672
    assert chess_rl_native.QUEEN_PLANES == 56
    assert chess_rl_native.KNIGHT_PLANES == 8
    assert chess_rl_native.UNDERPROMOTION_PLANES == 9
    assert chess_rl_native.POLICY_PLANES == 56 + 8 + 9
    assert chess_rl_native.POLICY_SIZE == 64 * 73


def test_directed_fixtures_unique_and_parseable():
    names = [n for n, _, _ in harness.DIRECTED_FIXTURES]
    fens = [fen for _, fen, _ in harness.DIRECTED_FIXTURES]
    assert len(names) == len(set(names))
    assert len(fens) == len(set(fens))
    for _, fen, moves in harness.DIRECTED_FIXTURES:
        board = chess.Board(fen)
        for uci in moves:
            board.push_uci(uci)  # every fixture (incl. history) must parse


def test_fixture_repetition_is_really_twofold():
    fen, moves = next(
        (f, m) for n, f, m in harness.DIRECTED_FIXTURES if n == "repetition"
    )
    oracle = chess.Board(fen)
    for uci in moves:
        oracle.push_uci(uci)
    assert oracle.is_repetition(2)


# ---------------------------------------------------------------------------
# Action map round-trip (directed fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n, _, _ in harness.DIRECTED_FIXTURES])
def test_action_map_round_trip_recovers_exact_uci(name):
    fen, moves = next(
        (f, m) for n, f, m in harness.DIRECTED_FIXTURES if n == name
    )
    board = chess.Board(fen)
    for uci in moves:
        board.push_uci(uci)
    side = "w" if board.turn == chess.WHITE else "b"
    for move in board.legal_moves:
        uci = move.uci()
        index = chess_rl_native.move_to_index(uci)
        assert index == encoding.move_to_index(move), (name, uci)
        assert chess_rl_native.index_to_move(index, side) == uci, (name, uci)


@pytest.mark.parametrize("name", [n for n, _, _ in harness.DIRECTED_FIXTURES])
def test_board_aware_index_to_move_exact_round_trip(name):
    fen, moves = next(
        (f, m) for n, f, m in harness.DIRECTED_FIXTURES if n == name
    )
    native = chess_rl_native.Position.from_uci_history(fen, list(moves))
    board = chess.Board(fen)
    for uci in moves:
        board.push_uci(uci)
    for move in board.legal_moves:
        uci = move.uci()
        index = chess_rl_native.move_to_index(uci)
        assert native.index_to_move(index) == uci, (name, uci)


def test_board_aware_index_to_move_distinguishes_queen_from_promotion():
    # A queen on the 7th rank stepping onto the back rank must NOT be read as a
    # promotion; the board-aware method is exact, whereas the board-free
    # heuristic would append 'q'.
    fen = "4k3/4Q3/8/8/8/8/8/4K3 w - - 0 1"
    native = chess_rl_native.Position.from_fen(fen)
    board = chess.Board(fen)
    for move in board.legal_moves:
        uci = move.uci()
        index = chess_rl_native.move_to_index(uci)
        assert native.index_to_move(index) == uci, uci
    # The queen's diagonal step onto the back rank is not a promotion.
    assert "e7d8" in {m.uci() for m in board.legal_moves}
    assert native.index_to_move(chess_rl_native.move_to_index("e7d8")) == "e7d8"


def test_move_to_index_matches_reference_for_every_legal_move_of_every_fixture():
    for _, fen, moves in harness.DIRECTED_FIXTURES:
        board = chess.Board(fen)
        for uci in moves:
            board.push_uci(uci)
        for move in board.legal_moves:
            assert chess_rl_native.move_to_index(move.uci()) == encoding.move_to_index(move)


@pytest.mark.parametrize("name", [n for n, _, _ in harness.DIRECTED_FIXTURES])
def test_legal_move_indices_match_reference(name):
    fen, moves = next(
        (f, m) for n, f, m in harness.DIRECTED_FIXTURES if n == name
    )
    native = chess_rl_native.Position.from_uci_history(fen, list(moves))
    board = chess.Board(fen)
    for uci in moves:
        board.push_uci(uci)
    expected = sorted(encoding.move_to_index(m) for m in board.legal_moves)
    assert native.legal_move_indices() == expected


def test_policy_to_vector_matches_reference():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves = list(board.legal_moves)
    prob = 1.0 / len(moves)
    policy = {m.uci(): prob for m in moves}
    native = chess_rl_native.policy_to_vector(policy)
    oracle = encoding.policy_to_vector({m: prob for m in moves})
    assert native.shape == (4672,)
    assert native.dtype == np.float32
    assert np.array_equal(native, oracle)


# ---------------------------------------------------------------------------
# Raw en-passant parity (the subtle part)
# ---------------------------------------------------------------------------


def test_raw_ep_capturable():
    native = chess_rl_native.Position.from_fen(
        "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"
    )
    assert native.raw_ep_square() == "e3"
    assert native.ep_square() == "e3"


def test_raw_ep_pinned_preserves_verbatim_fen_field():
    # The e5 pawn is pinned by the e8 rook, so no legal en passant exists; the
    # native *legal* ep_square() reports "-", but the raw field "d6" survives.
    native = chess_rl_native.Position.from_fen(
        "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    )
    assert native.ep_square() == "-"
    assert native.raw_ep_square() == "d6"
    oracle = chess.Board("4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert oracle.ep_square is not None
    assert chess.square_name(oracle.ep_square) == "d6"


def test_raw_ep_set_by_double_push_even_without_capturer():
    # e2e4 creates an ep square regardless of whether black can capture it.
    native = chess_rl_native.Position.from_fen(
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    )
    native.push_uci("e2e4")
    assert native.raw_ep_square() == "e3"
    # No black pawn can capture on e3 -> legal ep is "-".
    assert native.ep_square() == "-"


def test_raw_ep_cleared_after_non_double_push():
    native = chess_rl_native.Position.from_fen(START_FEN)
    native.push_uci("e2e4")
    assert native.raw_ep_square() == "e3"
    native.push_uci("e7e5")
    assert native.raw_ep_square() == "e6"
    native.push_uci("g1f3")  # knight move -> raw ep cleared
    assert native.raw_ep_square() == "-"


def test_raw_ep_restored_on_pop():
    native = chess_rl_native.Position.from_fen(START_FEN)
    native.push_uci("e2e4")
    assert native.raw_ep_square() == "e3"
    native.push_uci("g8f6")
    assert native.raw_ep_square() == "-"
    native.pop()
    assert native.raw_ep_square() == "e3"
    native.pop()
    assert native.raw_ep_square() == "-"


# ---------------------------------------------------------------------------
# Encoder parity (directed fixtures + deterministic random corpus)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n, _, _ in harness.DIRECTED_FIXTURES])
def test_encoder_parity_fixture(name):
    fen, moves = next(
        (f, m) for n, f, m in harness.DIRECTED_FIXTURES if n == name
    )
    native = chess_rl_native.Position.from_uci_history(fen, list(moves))
    board = chess.Board(fen)
    for uci in moves:
        board.push_uci(uci)
    assert np.array_equal(
        np.asarray(native.encode(history_steps=8)),
        encoding.encode_board(board, history_steps=8),
    )


def test_encode_fen_free_function_matches_encode():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    native = chess_rl_native.Position.from_fen(fen)
    board = chess.Board(fen)
    assert np.array_equal(
        np.asarray(chess_rl_native.encode_fen(fen, history_steps=8)),
        encoding.encode_board(board, history_steps=8),
    )
    assert np.array_equal(
        np.asarray(chess_rl_native.encode_fen(fen, history_steps=8)),
        np.asarray(native.encode(history_steps=8)),
    )


def test_encoder_shape_and_dtype():
    native = chess_rl_native.Position.from_fen(START_FEN)
    arr = np.asarray(native.encode(history_steps=8))
    assert arr.shape == (104, 8, 8)
    assert arr.dtype == np.float32


def test_encoder_pinned_ep_plane_is_raw():
    # Plane 101 must reflect the RAW ep square "d6", not the legal "-".
    fen = "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    native = chess_rl_native.Position.from_fen(fen)
    board = chess.Board(fen)
    native_enc = np.asarray(native.encode(history_steps=8))
    oracle_enc = encoding.encode_board(board, history_steps=8)
    assert np.array_equal(native_enc, oracle_enc)
    # d6 -> rank 5 (index 5), file 3.
    assert native_enc[101, 5, 3] == 1.0
    assert native_enc[101].sum() == 1.0


def test_encoder_repetition_plane():
    # The twofold-repetition fixture must set plane 103 to 1.0 in both engines.
    fen, moves = next(
        (f, m) for n, f, m in harness.DIRECTED_FIXTURES if n == "repetition"
    )
    native = chess_rl_native.Position.from_uci_history(fen, list(moves))
    board = chess.Board(fen)
    for uci in moves:
        board.push_uci(uci)
    assert native.is_repetition(2)
    assert board.is_repetition(2)
    native_enc = np.asarray(native.encode(history_steps=8))
    assert native_enc[103, 0, 0] == 1.0
    assert np.array_equal(native_enc, encoding.encode_board(board, history_steps=8))


# ---------------------------------------------------------------------------
# Small deterministic differential run (routine pytest size <= 1000)
# ---------------------------------------------------------------------------

GOLDEN_DIGEST_1000 = "84af53b87b0295521cb7372486cd77cf9a9e4a6c0b9229e05f4894640848d63b"
GOLDEN_DIRECTED_1000 = 20
GOLDEN_RANDOM_1000 = 980


def test_small_differential_1000():
    summary = harness.run_differential(1000, seed=harness.DEFAULT_SEED)
    assert summary["status"] == "ok"
    assert summary["count"] == 1000
    assert summary["mismatch_count"] == 0
    assert summary["directed_count"] + summary["random_count"] == 1000
    assert summary["seed"] == harness.DEFAULT_SEED
    assert summary["schema"] == harness.SCHEMA
    assert summary["directed_count"] == GOLDEN_DIRECTED_1000
    assert summary["random_count"] == GOLDEN_RANDOM_1000
    assert summary["corpus_digest_sha256"] == GOLDEN_DIGEST_1000


def test_differential_is_deterministic():
    first = harness.run_differential(500, seed=harness.DEFAULT_SEED)
    second = harness.run_differential(500, seed=harness.DEFAULT_SEED)
    assert first["corpus_digest_sha256"] == second["corpus_digest_sha256"]
    assert first["directed_count"] == second["directed_count"]
    assert first["random_count"] == second["random_count"]


def test_differential_exact_count_and_zero():
    assert harness.run_differential(7, seed=1)["count"] == 7
    zero = harness.run_differential(0, seed=1)
    assert zero["count"] == 0
    assert zero["mismatch_count"] == 0


def test_fail_closed_raises_with_reproduction(monkeypatch):
    real_encode = encoding.encode_board

    def corrupting_encode(board, history_steps=8):
        arr = real_encode(board, history_steps=history_steps)
        # Desync plane 96 (side to move) only on positions with history, so the
        # very first random position (START_FEN, empty history) stays clean and
        # the fail-closed record carries a START_FEN trajectory with a complete
        # UCI history (directed fixtures are excluded via fixtures=()).
        if board.move_stack:
            arr[96, 0, 0] += 1.0
        return arr

    monkeypatch.setattr(encoding, "encode_board", corrupting_encode)
    with pytest.raises(harness.DifferentialMismatch) as excinfo:
        harness.run_differential(50, seed=harness.DEFAULT_SEED, fixtures=())
    record = excinfo.value.record
    assert record["status"] == "mismatch"
    assert record["start_fen"] == START_FEN
    assert record["uci_history"], "complete history must be present"
    assert "encoder" in record["mismatched"]
    assert record["count_requested"] == 50
    assert record["position_index"] >= 1
    payload = json.loads(json.dumps(record))
    assert payload["schema"] == harness.SCHEMA
    assert isinstance(payload["uci_history"], list)
    assert isinstance(payload["mismatched"], dict)


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "native_policy_encoder_parity.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )


def test_cli_json_small():
    proc = _run_cli("--positions", "100", "--seed", "0x20260822", "--json")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["status"] == "ok"
    assert summary["count"] == 100
    assert summary["mismatch_count"] == 0
    assert summary["seed"] == harness.DEFAULT_SEED
    assert summary["directed_count"] + summary["random_count"] == 100
    assert summary["corpus_digest_sha256"]
    assert summary["schema"] == harness.SCHEMA
    assert summary["versions"]["python_chess_dist"] == "1.999"
    assert summary["versions"]["chess_rl_native_abi"] == "1"


def test_cli_list_fixtures():
    proc = _run_cli("--list-fixtures")
    assert proc.returncode == 0
    assert "checkmate" in proc.stdout
    assert "en_passant_pinned_illegal" in proc.stdout
    assert "repetition" in proc.stdout
