"""Differential harness tests: python-chess 1.999 oracle vs chess_rl_native.Position.

Strict TDD: these tests are written against the harness contract before the
harness exists (RED), then the harness is implemented to make them pass
(GREEN). The routine pytest run stays small (1000 positions); the full
100k run is a CLI exercise, never part of pytest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import chess
import chess_rl_native

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmarks.native_legal_differential as harness  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --------------------------------------------------------------------------
# Constants / schema
# --------------------------------------------------------------------------


def test_schema_and_defaults():
    assert harness.SCHEMA == "native_legal_differential/v1"
    assert harness.DEFAULT_SEED == 0x20260822
    assert harness.DEFAULT_POSITIONS == 100_000
    assert harness.START_FEN == START_FEN


def test_directed_fixtures_unique_names_and_fens():
    names = [name for name, _ in harness.DIRECTED_FIXTURES]
    fens = [fen for _, fen in harness.DIRECTED_FIXTURES]
    assert len(names) == len(set(names))
    assert len(fens) == len(set(fens))
    for _, fen in harness.DIRECTED_FIXTURES:
        chess.Board(fen)  # every fixture must parse under the oracle


# --------------------------------------------------------------------------
# Oracle semantics of the directed fixtures (test design inputs)
# --------------------------------------------------------------------------

FIXTURE_SEMANTICS = {
    "castling_white": {"must_contain": {"e1g1", "e1c1"}},
    "castling_black": {"must_contain": {"e8g8", "e8c8"}},
    "en_passant_capturable": {
        "must_contain": {"d4e3"},
        "legal_ep": "e3",
    },
    "en_passant_pinned_illegal": {
        "must_not_contain": {"e5d6"},
        "legal_ep": "-",
    },
    "en_passant_pinned_bishop_covers_ep": {
        "must_contain": {"f5e6"},
        "must_not_contain": {"d5e6"},
        "legal_ep": "-",
    },
    "promotion_white": {"must_contain": {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}},
    "promotion_white_capture": {
        "must_contain": {"b7a8q", "b7a8r", "b7a8b", "b7a8n", "b7b8q"}
    },
    "promotion_black": {"must_contain": {"a2a1q", "a2a1r", "a2a1b", "a2a1n"}},
    "promotion_black_capture": {
        "must_contain": {"b2a1q", "b2a1r", "b2a1b", "b2a1n", "b2b1q"}
    },
    "pin_knight": {"exact": {"d1c1", "d1c2", "d1e1", "d1e2"}},
    "double_check": {"exact": {"e1d1", "e1f1", "e1f2"}},
    "checkmate": {"empty": True, "checkmate": True},
    "stalemate": {"empty": True, "stalemate": True},
    "insufficient_kk": {"insufficient": True},
    "insufficient_kn": {"insufficient": True},
    "insufficient_kb": {"insufficient": True},
    "halfmove_99": {"halfmove": 99},
    "halfmove_100": {"halfmove": 100},
    "fullmove_50_black": {"fullmove": 50},
}


@pytest.mark.parametrize("name", [n for n, _ in harness.DIRECTED_FIXTURES])
def test_fixture_oracle_semantics(name):
    fen = dict(harness.DIRECTED_FIXTURES)[name]
    board = chess.Board(fen)
    moves = {str(m) for m in board.legal_moves}
    spec = FIXTURE_SEMANTICS[name]
    assert name in FIXTURE_SEMANTICS, "fixture missing semantic spec"
    if "must_contain" in spec:
        assert spec["must_contain"] <= moves
    if "must_not_contain" in spec:
        assert moves.isdisjoint(spec["must_not_contain"])
    if "exact" in spec:
        assert moves == spec["exact"]
    if spec.get("empty"):
        assert moves == set()
    if "checkmate" in spec:
        assert board.is_checkmate()
    if "stalemate" in spec:
        assert board.is_stalemate()
    if "insufficient" in spec:
        assert board.is_insufficient_material()
    if "halfmove" in spec:
        assert board.halfmove_clock == spec["halfmove"]
    if "fullmove" in spec:
        assert board.fullmove_number == spec["fullmove"]
    if "legal_ep" in spec:
        expected = spec["legal_ep"]
        actual = (
            chess.square_name(board.ep_square)
            if board.has_legal_en_passant()
            else "-"
        )
        assert actual == expected


# --------------------------------------------------------------------------
# Legal-EP semantics helpers
# --------------------------------------------------------------------------


def test_oracle_legal_ep_pinned_position():
    # d6 is the raw FEN field, but the pinned e5 pawn cannot capture it.
    board = chess.Board("4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert board.ep_square == chess.parse_square("d6")
    assert not board.has_legal_en_passant()
    assert board.fen().split()[3] == "-"


def test_native_effective_ep_capturable_and_pinned():
    capturable = chess_rl_native.Position.from_fen(
        "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"
    )
    assert capturable.ep_square() == "e3"
    assert harness.effective_ep(capturable) == "e3"

    pinned = chess_rl_native.Position.from_fen(
        "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    )
    assert pinned.ep_square() == "-"
    assert harness.effective_ep(pinned) == "-"

    none = chess_rl_native.Position.from_fen(START_FEN)
    assert harness.effective_ep(none) == "-"


def test_native_effective_ep_bishop_covering_ep_square():
    # Regression from the 100k corpus (position 84422): after black's e7e5
    # the EP square would be e6, but the d5 pawn is pinned by the d8 queen
    # (so d5xe6 is illegal) and the f5 bishop can legally land on e6. A
    # non-pawn move onto the EP square must not be mistaken for a legal en
    # passant capture. The native position stores *legal-EP* semantics
    # (setFen validates the EP square and makeMove<true> records it only when
    # a legal capture exists), so ep_square() reports "-" here.
    before = chess_rl_native.Position.from_fen(
        "1r1q4/p1p1p3/Pp3n1k/3P1Br1/1PP2P1p/8/3KNP1P/RN1R4 b - - 0 45"
    )
    before.push_uci("e7e5")
    assert before.ep_square() == "-"  # no legal en passant (d5 pawn pinned)
    assert "d5e6" not in before.legal_moves_uci()  # pinned pawn cannot take e.p.
    assert "f5e6" in before.legal_moves_uci()  # bishop lands on the EP square
    assert harness.effective_ep(before) == "-"


# --------------------------------------------------------------------------
# Differential agreement at every directed fixture
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n, _ in harness.DIRECTED_FIXTURES])
def test_fixture_native_oracle_agreement(name):
    fen = dict(harness.DIRECTED_FIXTURES)[name]
    native = chess_rl_native.Position.from_fen(fen)
    oracle = chess.Board(fen)
    record = harness.compare_position(native, oracle, index=0)
    assert record is None, f"{name}: {record}"


# --------------------------------------------------------------------------
# Small deterministic differential run (the routine pytest size)
# --------------------------------------------------------------------------

# Pinned after the first GREEN run: deterministic digest + phase counts for
# seed 0x20260822 over 1000 counted positions. Changing the corpus format,
# fixture set, or RNG stream breaks this test by design.
GOLDEN_DIGEST_1000 = "17cb47da1f0733dc295a16d314e9bf9fdf1f7198db37e771dfb6f83e39271932"
GOLDEN_DIRECTED_1000 = 19
GOLDEN_RANDOM_1000 = 981


def test_small_differential_1000():
    summary = harness.run_differential(1000, seed=harness.DEFAULT_SEED)
    assert summary["status"] == "ok"
    assert summary["count"] == 1000
    assert summary["mismatch_count"] == 0
    assert summary["directed_count"] + summary["random_count"] == 1000
    assert summary["seed"] == harness.DEFAULT_SEED
    assert summary["schema"] == harness.SCHEMA
    assert GOLDEN_DIGEST_1000 != "0" * 64, "golden digest not pinned"
    assert summary["corpus_digest_sha256"] == GOLDEN_DIGEST_1000
    assert summary["directed_count"] == GOLDEN_DIRECTED_1000
    assert summary["random_count"] == GOLDEN_RANDOM_1000


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


# --------------------------------------------------------------------------
# Fail closed on first mismatch with a machine-readable reproduction
# --------------------------------------------------------------------------


def test_fail_closed_raises_with_reproduction(monkeypatch):
    real_push_uci = chess.Board.push_uci

    def corrupting_push(self, uci):
        real_push_uci(self, uci)
        # Deterministically desync the oracle: bump its halfmove clock after
        # every push so the next counted position mismatches on halfmove.
        self.halfmove_clock += 1

    monkeypatch.setattr(chess.Board, "push_uci", corrupting_push)
    with pytest.raises(harness.DifferentialMismatch) as excinfo:
        harness.run_differential(50, seed=harness.DEFAULT_SEED)
    record = excinfo.value.record
    assert record["status"] == "mismatch"
    # Phase A counts one position per fixture without pushing further, so the
    # first corrupted push lands in Phase B and the mismatch's start FEN is the
    # start FEN; the record must carry the complete history from there.
    assert record["start_fen"] == START_FEN
    assert record["uci_history"], "complete history must be present"
    assert "halfmove" in record["mismatched"]
    assert record["count_requested"] == 50
    assert record["position_index"] >= 1
    # Machine-readable: the record serializes to JSON with the essential keys.
    payload = json.loads(json.dumps(record))
    assert payload["schema"] == harness.SCHEMA
    assert isinstance(payload["uci_history"], list)
    assert isinstance(payload["mismatched"], dict)


# --------------------------------------------------------------------------
# CLI behaviour
# --------------------------------------------------------------------------


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "native_legal_differential.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
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


def test_cli_exact_requested_count():
    for n in (1, 17, 1000):
        proc = _run_cli("--positions", str(n), "--seed", "5", "--json")
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["count"] == n


def test_cli_output_file(tmp_path):
    out = tmp_path / "result.json"
    proc = _run_cli("--positions", "50", "--seed", "7", "--json", "--output", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    summary = json.loads(out.read_text())
    assert summary["count"] == 50
    assert summary["result_path"] == str(out)


def test_cli_list_fixtures():
    proc = _run_cli("--list-fixtures")
    assert proc.returncode == 0
    assert "checkmate" in proc.stdout
    assert "en_passant_pinned_illegal" in proc.stdout
