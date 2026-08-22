"""Deterministic native-vs-python parity harness for the policy map and encoder.

Compares ``chess_rl_native`` (chess-library) against the pinned oracle
``python-chess==1.999`` + the reference ``encoding.py`` at every counted
position along deterministic trajectories.  Three exact, bit-for-bit checks
run per position:

1. **Action map (forward)** — ``Position.legal_move_indices()`` equals the
   sorted ``encoding.move_to_index(move)`` for every legal move; and
   ``policy_to_vector`` (uniform mass over the legal moves) equals
   ``encoding.policy_to_vector`` exactly.
2. **Encoder** — ``Position.encode(history_steps=8)`` is ``np.array_equal``
   to ``encoding.encode_board(chess.Board(...), history_steps=8)``: 104
   float32 planes, white orientation, no color flip, history most-recent
   first, and the 8 meta planes (side, castling, *raw* en-passant, halfmove,
   repetition) taken from the current position.

The inverse ``index_to_move(index, side_to_move)`` is a documented *lossy*
heuristic (the 4672 index carries from-square + plane but not the piece type,
so a queen-like plane landing a non-pawn on the back rank cannot be told from
a queen promotion without the board — exactly as in ``encoding.index_to_move``
which takes the board for that reason).  Its exact round-trip is asserted in
``tests/test_native_policy_encoding.py`` over the directed fixtures (which are
unambiguous); it is not part of the 100k random-phase bit-exact parity.

Trajectories in the random phase start at the start FEN, choose each next
move with ``random.Random(seed)`` over the *sorted* legal UCI list, and
continue until terminal (no legal moves, insufficient material, or a 100
halfmove-clock boundary), then reset deterministically to the start FEN.
Directed fixtures contribute exactly one counted position each at the head of
the corpus and cover castling for both colors, capturable and pinned
(illegal) en passant, bishop-covers-the-ep-square, all promotion types for
both colors with and without capture, pins, double check, checkmate,
stalemate, insufficient material, halfmove/fullmove boundary FENs, and a
twofold-repetition position (built via a UCI history, since repetition is a
history property).

The harness fails closed: the first mismatch raises ``DifferentialMismatch``
carrying a complete machine-readable reproduction (start FEN + complete UCI
history + per-field native/oracle values), and the CLI exits 2.

Deterministic corpus digest: SHA-256 over one canonical line per counted
position::

    index<TAB>placement<TAB>side<TAB>castling<TAB>raw_ep<TAB>halfmove<TAB>fullmove<TAB>moves<TAB>action<TAB>encoder_sha256<TAB>indices_sha256

where ``moves`` is the sorted legal UCI list joined by commas, ``action`` is
``NEXT:<uci>``, ``TERMINAL`` or ``END``, ``encoder_sha256`` is the SHA-256 of
the oracle's 104-plane float32 bytes and ``indices_sha256`` the SHA-256 of the
canonical sorted index list.

Schema of the JSON summary / mismatch record: ``native_policy_encoder_parity/v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time

import chess
import numpy as np

import chess_rl_native

# The reference encoder is the single source of truth for both the action map
# and the 104-plane encoder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encoding  # noqa: E402

SCHEMA = "native_policy_encoder_parity/v1"
DEFAULT_SEED = 0x20260822
DEFAULT_POSITIONS = 100_000
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
HISTORY_STEPS = 8

# (name, FEN, UCI history) directed fixtures injected at the head of every
# corpus.  Most are bare FENs (empty history); the repetition fixture carries
# the twofold knight-cycle so both engines agree it is a repetition.
DIRECTED_FIXTURES = (
    ("castling_white", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ()),
    ("castling_black", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", ()),
    ("en_passant_capturable", "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", ()),
    ("en_passant_pinned_illegal", "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ()),
    (
        "en_passant_pinned_bishop_covers_ep",
        "1r1q4/p1p5/Pp3n1k/3PpBr1/1PP2P1p/8/3KNP1P/RN1R4 w - e6 0 46",
        (),
    ),
    ("promotion_white", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ()),
    ("promotion_white_capture", "r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", ()),
    ("promotion_black", "4k3/8/8/8/8/8/p7/4K3 b - - 0 1", ()),
    ("promotion_black_capture", "4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1", ()),
    ("pin_knight", "3r2k1/8/8/8/8/8/3N4/3K4 w - - 0 1", ()),
    ("double_check", "4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1", ()),
    ("checkmate", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", ()),
    ("stalemate", "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", ()),
    ("insufficient_kk", "8/8/8/4k3/8/8/8/4K3 w - - 0 1", ()),
    ("insufficient_kn", "8/8/8/4k3/8/8/8/4K1N1 w - - 0 1", ()),
    ("insufficient_kb", "8/8/8/4k3/8/8/8/4K2B w - - 0 1", ()),
    ("halfmove_99", "4k3/8/8/8/8/8/8/4K3 w - - 99 1", ()),
    ("halfmove_100", "4k3/8/8/8/8/8/8/4K3 w - - 100 1", ()),
    ("fullmove_50_black", "4k3/8/8/8/8/8/8/4K3 b - - 0 50", ()),
    ("repetition", START_FEN, ("g1f3", "g8f6", "f3g1", "f6g8")),
)


class DifferentialMismatch(RuntimeError):
    """Raised on the first differential mismatch; carries the record."""

    def __init__(self, record: dict) -> None:
        self.record = record
        super().__init__(json.dumps(record, indent=2, sort_keys=True))


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "python_chess_dist": importlib.metadata.version("python-chess"),
        "python_chess_module": getattr(chess, "__version__", "unknown"),
        "chess_rl_native_module": getattr(chess_rl_native, "__version__", "unknown"),
        "chess_rl_native_abi": chess_rl_native.native_abi_version(),
        "chess_library_commit": chess_rl_native.chess_library_commit(),
        "chess_library_header_sha256": chess_rl_native.chess_library_header_sha256(),
        "numpy": importlib.metadata.version("numpy"),
    }


def native_from_fixture(fen: str, moves):
    if moves:
        return chess_rl_native.Position.from_uci_history(fen, list(moves))
    return chess_rl_native.Position.from_fen(fen)


def oracle_from_fixture(fen: str, moves):
    board = chess.Board(fen)
    for uci in moves:
        board.push_uci(uci)
    return board


def oracle_raw_ep(board: chess.Board) -> str:
    """python-chess RAW ep square (the verbatim 4th FEN field / last double push)."""
    if board.ep_square is None:
        return "-"
    return chess.square_name(board.ep_square)


def compare_position(native, oracle: chess.Board, index: int):
    """Compare action map (forward) + encoder. Returns None or a mismatch dict."""
    mismatched = {}

    # 1. Action map: legal_move_indices (sorted) vs oracle move_to_index.
    native_indices = native.legal_move_indices()
    oracle_indices = sorted(encoding.move_to_index(m) for m in oracle.legal_moves)
    if native_indices != oracle_indices:
        mismatched["legal_move_indices"] = {
            "native": native_indices,
            "oracle": oracle_indices,
        }

    # 2. Action map: policy_to_vector (uniform mass over legal moves).
    moves = list(oracle.legal_moves)
    if moves:
        prob = 1.0 / len(moves)
        policy = {str(m): prob for m in moves}
        native_vec = chess_rl_native.policy_to_vector(policy)
        oracle_vec = encoding.policy_to_vector({m: prob for m in moves})
        if not np.array_equal(native_vec, oracle_vec):
            diff = int(np.flatnonzero(native_vec != oracle_vec)[0])
            mismatched["policy_to_vector"] = {
                "first_diff_index": diff,
                "native": float(native_vec[diff]),
                "oracle": float(oracle_vec[diff]),
            }

    # 3. Encoder: 104-plane bit-for-bit parity.
    native_enc = np.asarray(native.encode(history_steps=HISTORY_STEPS))
    oracle_enc = encoding.encode_board(oracle, history_steps=HISTORY_STEPS)
    if native_enc.shape != oracle_enc.shape:
        mismatched["encoder_shape"] = {
            "native": list(native_enc.shape),
            "oracle": list(oracle_enc.shape),
        }
    elif not np.array_equal(native_enc, oracle_enc):
        diff_plane, diff_rank, diff_file = np.argwhere(native_enc != oracle_enc)[0]
        mismatched["encoder"] = {
            "first_diff": [int(diff_plane), int(diff_rank), int(diff_file)],
            "native": float(native_enc[diff_plane, diff_rank, diff_file]),
            "oracle": float(oracle_enc[diff_plane, diff_rank, diff_file]),
        }

    if not mismatched:
        return None
    return {
        "position_index": index,
        "mismatched": mismatched,
    }


def _index_digest_line(indices):
    return ",".join(str(i) for i in indices)


def run_differential(
    count: int,
    seed: int = DEFAULT_SEED,
    fixtures=DIRECTED_FIXTURES,
    start_fen: str = START_FEN,
    progress_cb=None,
) -> dict:
    """Run the parity over exactly ``count`` counted positions.

    Raises ``DifferentialMismatch`` (fail closed) on the first mismatch.
    """
    if count < 0:
        raise ValueError("count must be non-negative")

    rng = random.Random(seed)
    digest = hashlib.sha256()
    started = time.perf_counter()
    counted = 0
    directed_count = 0

    def digest_line(native, oracle, index, action, encoder_bytes, indices):
        side = "w" if oracle.turn == chess.WHITE else "b"
        moves = ",".join(sorted(str(m) for m in oracle.legal_moves))
        enc_digest = hashlib.sha256(encoder_bytes).hexdigest()
        idx_digest = hashlib.sha256(_index_digest_line(indices).encode()).hexdigest()
        return (
            f"{index}\t{oracle.board_fen()}\t{side}\t{oracle.castling_xfen()}\t"
            f"{oracle_raw_ep(oracle)}\t{oracle.halfmove_clock}\t"
            f"{oracle.fullmove_number}\t{moves}\t{action}\t{enc_digest}\t{idx_digest}\n"
        )

    def step(native, oracle, history, start, directed) -> bool:
        nonlocal counted, directed_count
        index = counted
        record = compare_position(native, oracle, index)
        if record is not None:
            record.update(
                {
                    "schema": SCHEMA,
                    "status": "mismatch",
                    "start_fen": start,
                    "uci_history": list(history),
                    "seed": seed,
                    "count_requested": count,
                    "versions": versions(),
                }
            )
            raise DifferentialMismatch(record)
        if directed:
            directed_count += 1

        native_indices = native.legal_move_indices()
        oracle_enc = encoding.encode_board(oracle, history_steps=HISTORY_STEPS)
        encoder_bytes = np.ascontiguousarray(oracle_enc).tobytes()

        moves = native.legal_moves_uci()
        if index + 1 >= count:
            action = "END"
        elif not moves or oracle.is_game_over() or oracle.halfmove_clock >= 100:
            action = "TERMINAL"
        else:
            action = "NEXT:" + rng.choice(moves)

        digest.update(
            digest_line(native, oracle, index, action, encoder_bytes, native_indices).encode()
        )
        counted += 1
        if action.startswith("NEXT:"):
            move = action[len("NEXT:"):]
            native.push_uci(move)
            oracle.push_uci(move)
            history.append(move)
        if progress_cb is not None and counted % 10_000 == 0:
            progress_cb(counted)
        return counted < count and action != "TERMINAL"

    # Phase A: directed fixtures -- exactly one counted position each.
    for name, fen, moves in fixtures:
        if counted >= count:
            break
        native = native_from_fixture(fen, moves)
        oracle = oracle_from_fixture(fen, moves)
        step(native, oracle, [], fen, directed=True)

    # Phase B: random trajectories from the start FEN with deterministic reset.
    if counted < count:
        native = chess_rl_native.Position.from_fen(start_fen)
        oracle = chess.Board(start_fen)
        history = []
        while counted < count:
            if not step(native, oracle, history, start_fen, directed=False):
                native = chess_rl_native.Position.from_fen(start_fen)
                oracle = chess.Board(start_fen)
                history = []

    if counted != count:
        raise RuntimeError(f"internal error: counted {counted} != requested {count}")

    elapsed = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "status": "ok",
        "versions": versions(),
        "seed": seed,
        "count": counted,
        "directed_count": directed_count,
        "random_count": counted - directed_count,
        "elapsed_seconds": round(elapsed, 6),
        "positions_per_second": round(counted / elapsed, 2) if elapsed > 0 else 0.0,
        "mismatch_count": 0,
        "corpus_digest_sha256": digest.hexdigest(),
        "start_fen": start_fen,
        "result_path": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seed(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid seed: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="native_policy_encoder_parity.py",
        description=(
            "Deterministic differential: chess_rl_native action map + 104-plane "
            "encoder vs python-chess==1.999 + encoding.py. Fails closed (exit 2) "
            "on the first mismatch with a machine-readable record."
        ),
    )
    parser.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    parser.add_argument("--seed", type=_parse_seed, default=DEFAULT_SEED)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", metavar="PATH", default=None)
    parser.add_argument("--list-fixtures", action="store_true")
    return parser


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _print_human(summary: dict) -> None:
    v = summary["versions"]
    print(f"schema: {summary['schema']}")
    print(f"status: {summary['status']}")
    print(
        "versions: python=%s python_chess_dist=%s python_chess_module=%s "
        "native_module=%s native_abi=%s chess_library_commit=%s "
        "chess_library_header_sha256=%s numpy=%s"
        % (
            v["python"],
            v["python_chess_dist"],
            v["python_chess_module"],
            v["chess_rl_native_module"],
            v["chess_rl_native_abi"],
            v["chess_library_commit"],
            v["chess_library_header_sha256"],
            v["numpy"],
        )
    )
    print(f"seed: 0x{summary['seed']:x}")
    print(f"count: {summary['count']}")
    print(f"directed_count: {summary['directed_count']}")
    print(f"random_count: {summary['random_count']}")
    print(f"elapsed_seconds: {summary['elapsed_seconds']}")
    print(f"positions_per_second: {summary['positions_per_second']}")
    print(f"mismatch_count: {summary['mismatch_count']}")
    print(f"corpus_digest_sha256: {summary['corpus_digest_sha256']}")
    print(f"result_path: {summary['result_path']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_fixtures:
        for name, fen, moves in DIRECTED_FIXTURES:
            print(f"{name}\t{fen}\t{' '.join(moves)}")
        return 0

    if args.positions < 0:
        print("--positions must be non-negative", file=sys.stderr)
        return 2

    progress = None if args.json else (lambda n: print(f"counted {n}...", file=sys.stderr))
    try:
        summary = run_differential(args.positions, seed=args.seed, progress_cb=progress)
    except DifferentialMismatch as exc:
        record = exc.record
        if args.output:
            record["result_path"] = os.path.abspath(args.output)
            _write_json(record["result_path"], record)
        print(json.dumps(record, indent=2, sort_keys=True))
        print(
            f"FAIL: mismatch at position {record['position_index']} "
            f"(start_fen={record['start_fen']!r}, uci_history={record['uci_history']!r})",
            file=sys.stderr,
        )
        return 2

    if args.output:
        summary["result_path"] = os.path.abspath(args.output)
        _write_json(summary["result_path"], summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
