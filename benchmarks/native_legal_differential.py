"""Deterministic native-vs-python legal-position differential harness.

Compares ``chess_rl_native.Position`` (chess-library) against the pinned
oracle ``python-chess==1.999`` at every counted position along deterministic
trajectories:

* sorted legal UCI move set (``legal_moves_uci()`` vs ``Board.legal_moves``)
* the six FEN fields -- placement, side to move, castling rights, en passant,
  halfmove clock, fullmove number -- where the en passant field uses
  python-chess *legal-EP semantics* (``ep_square`` is only reported when a
  legal en passant capture exists, matching ``Board.fen()``'s default
  ``en_passant="legal"`` serialization).

Trajectories in the random phase start at the start FEN, choose each next
move with ``random.Random(seed)`` over the *sorted* legal UCI list, and
continue until terminal (no legal moves, insufficient material, or a 100
halfmove-clock boundary), then reset deterministically to the start FEN.
Directed fixtures contribute exactly one counted position each at the head of
the corpus and cover castling for both colors, capturable and pinned
(illegal) en passant, all promotion types for both colors with and without
capture, pins, double check, checkmate, stalemate, insufficient material, and
halfmove/fullmove boundary FENs.

The harness fails closed: the first mismatch raises ``DifferentialMismatch``
carrying a complete machine-readable reproduction (start FEN + complete UCI
history + per-field native/oracle values), and the CLI exits 2.

Deterministic corpus digest: SHA-256 over one canonical line per counted
position::

    index<TAB>placement<TAB>side<TAB>castling<TAB>ep<TAB>halfmove<TAB>fullmove<TAB>moves<TAB>action

where ``moves`` is the sorted legal UCI list joined by commas and ``action``
is ``NEXT:<uci>``, ``TERMINAL`` (trajectory ended at this position) or ``END``
(last counted position).

Schema of the JSON summary / mismatch record: ``native_legal_differential/v1``.
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
import chess_rl_native

SCHEMA = "native_legal_differential/v1"
DEFAULT_SEED = 0x20260822
DEFAULT_POSITIONS = 100_000
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# (name, FEN) directed fixtures injected at the head of every corpus.
DIRECTED_FIXTURES = (
    ("castling_white", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("castling_black", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
    ("en_passant_capturable", "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"),
    ("en_passant_pinned_illegal", "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
    (
        "en_passant_pinned_bishop_covers_ep",
        "1r1q4/p1p5/Pp3n1k/3PpBr1/1PP2P1p/8/3KNP1P/RN1R4 w - e6 0 46",
    ),
    ("promotion_white", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ("promotion_white_capture", "r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1"),
    ("promotion_black", "4k3/8/8/8/8/8/p7/4K3 b - - 0 1"),
    ("promotion_black_capture", "4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1"),
    ("pin_knight", "3r2k1/8/8/8/8/8/3N4/3K4 w - - 0 1"),
    ("double_check", "4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1"),
    ("checkmate", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
    ("stalemate", "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"),
    ("insufficient_kk", "8/8/8/4k3/8/8/8/4K3 w - - 0 1"),
    ("insufficient_kn", "8/8/8/4k3/8/8/8/4K1N1 w - - 0 1"),
    ("insufficient_kb", "8/8/8/4k3/8/8/8/4K2B w - - 0 1"),
    ("halfmove_99", "4k3/8/8/8/8/8/8/4K3 w - - 99 1"),
    ("halfmove_100", "4k3/8/8/8/8/8/8/4K3 w - - 100 1"),
    ("fullmove_50_black", "4k3/8/8/8/8/8/8/4K3 b - - 0 50"),
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
    }


def oracle_legal_ep(board: chess.Board) -> str:
    """python-chess legal-EP semantics: square name or ``-``."""
    if board.has_legal_en_passant():
        ep = board.ep_square
        assert ep is not None  # guaranteed by has_legal_en_passant()
        return chess.square_name(ep)
    return "-"


def _piece_at(placement: str, square: str):
    """Piece character ('P'/'p'/...) at ``square`` from a FEN placement."""
    row = placement.split("/")[8 - int(square[1])]
    file_index = ord(square[0]) - ord("a")
    col = 0
    for char in row:
        if char.isdigit():
            col += int(char)
        else:
            if col == file_index:
                return char
            col += 1
    return None


def effective_ep(native) -> str:
    """Native EP square normalized to python-chess legal-EP semantics.

    The native position already stores legal-EP semantics (``setFen`` drops an
    EP square with no capturer and ``makeMove<true>`` records it only when a
    legal capture exists), matching python-chess's legal-EP FEN field. This
    helper re-derives the legal-EP value from the native legal move set as a
    defence-in-depth: the EP capture is the only legal *pawn* move from the
    adjacent-file pawn square onto the (empty) EP square, so a non-pawn move
    landing on the EP square (for example a bishop or queen) must not be
    counted.
    """
    raw = native.ep_square()
    if raw == "-":
        return "-"
    ep_file = ord(raw[0]) - ord("a")
    ep_rank = int(raw[1])
    from_rank = ep_rank - 1 if native.side_to_move() == "w" else ep_rank + 1
    capturers = {
        chr(ord("a") + ep_file - 1) + str(from_rank),
        chr(ord("a") + ep_file + 1) + str(from_rank),
    }
    placement = native.fen().split(" ", 1)[0]
    for move in native.legal_moves_uci():
        if move[2:4] == raw and move[0:2] in capturers:
            if _piece_at(placement, move[0:2]) in ("P", "p"):
                return raw
    return "-"


def compare_position(native, oracle: chess.Board, index: int):
    """Compare one position. Returns None or a mismatch dict (no trajectory)."""
    native_moves = native.legal_moves_uci()
    oracle_moves = sorted(str(m) for m in oracle.legal_moves)
    native_fields = {
        "placement": native.fen().split(" ", 1)[0],
        "side": native.side_to_move(),
        "castling": native.castling_rights(),
        "ep": effective_ep(native),
        "halfmove": int(native.halfmove_clock()),
        "fullmove": int(native.fullmove_number()),
    }
    oracle_fields = {
        "placement": oracle.board_fen(),
        "side": "w" if oracle.turn == chess.WHITE else "b",
        "castling": oracle.castling_xfen(),
        "ep": oracle_legal_ep(oracle),
        "halfmove": int(oracle.halfmove_clock),
        "fullmove": int(oracle.fullmove_number),
    }
    mismatched = {}
    for field in native_fields:
        if native_fields[field] != oracle_fields[field]:
            mismatched[field] = {
                "native": native_fields[field],
                "oracle": oracle_fields[field],
            }
    if native_moves != oracle_moves:
        mismatched["legal_moves"] = {
            "native": native_moves,
            "oracle": oracle_moves,
        }
    if not mismatched:
        return None
    return {
        "position_index": index,
        "mismatched": mismatched,
        "native_fields": native_fields,
        "oracle_fields": oracle_fields,
    }


def run_differential(
    count: int,
    seed: int = DEFAULT_SEED,
    fixtures=DIRECTED_FIXTURES,
    start_fen: str = START_FEN,
    progress_cb=None,
) -> dict:
    """Run the differential over exactly ``count`` counted positions.

    Raises ``DifferentialMismatch`` (fail closed) on the first mismatch.
    Returns the summary dict with schema, versions, seed, count, directed and
    random counts, elapsed time, positions/sec, mismatch count and the
    deterministic corpus digest.
    """
    if count < 0:
        raise ValueError("count must be non-negative")

    rng = random.Random(seed)
    digest = hashlib.sha256()
    started = time.perf_counter()
    counted = 0
    directed_count = 0

    def digest_line(native, oracle, index, action):
        side = "w" if oracle.turn == chess.WHITE else "b"
        moves = ",".join(sorted(str(m) for m in oracle.legal_moves))
        return (
            f"{index}\t{oracle.board_fen()}\t{side}\t{oracle.castling_xfen()}\t"
            f"{oracle_legal_ep(oracle)}\t{oracle.halfmove_clock}\t"
            f"{oracle.fullmove_number}\t{moves}\t{action}\n"
        )

    def step(native, oracle, history, start, directed) -> bool:
        """Compare one counted position, digest it, and advance one move.

        Returns False when the trajectory is terminal or the count is
        exhausted (the caller resets or stops).
        """
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

        moves = native.legal_moves_uci()
        if index + 1 >= count:
            action = "END"
        elif not moves or oracle.is_game_over() or oracle.halfmove_clock >= 100:
            action = "TERMINAL"
        else:
            action = "NEXT:" + rng.choice(moves)

        digest.update(digest_line(native, oracle, index, action).encode())
        counted += 1
        if action.startswith("NEXT:"):
            move = action[len("NEXT:"):]
            native.push_uci(move)
            oracle.push_uci(move)
            history.append(move)
        if progress_cb is not None and counted % 10_000 == 0:
            progress_cb(counted)
        return counted < count and action != "TERMINAL"

    # Phase A: directed fixtures -- exactly one counted position each, then the
    # next fixture. Trajectory continuation belongs to the random phase below.
    for name, fen in fixtures:
        if counted >= count:
            break
        native = chess_rl_native.Position.from_fen(fen)
        oracle = chess.Board(fen)
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

    if counted != count:  # internal invariant: exact requested count
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
        return int(value, 0)  # accepts 0x20260822 and plain decimal
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid seed: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="native_legal_differential.py",
        description=(
            "Deterministic differential: chess_rl_native.Position vs "
            "python-chess==1.999 at every counted position. Fails closed "
            "(exit 2) on the first mismatch with a machine-readable record."
        ),
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=DEFAULT_POSITIONS,
        help=f"exact number of counted positions (default: {DEFAULT_POSITIONS})",
    )
    parser.add_argument(
        "--seed",
        type=_parse_seed,
        default=DEFAULT_SEED,
        help=f"deterministic RNG seed, decimal or 0x hex (default: 0x{DEFAULT_SEED:x})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the summary (or mismatch record) as JSON on stdout",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="also write the result JSON to PATH",
    )
    parser.add_argument(
        "--list-fixtures",
        action="store_true",
        help="list the directed fixture FENs and exit",
    )
    return parser


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _print_human(summary: dict) -> None:
    versions = summary["versions"]
    print(f"schema: {summary['schema']}")
    print(f"status: {summary['status']}")
    print(
        "versions: python=%s python_chess_dist=%s python_chess_module=%s "
        "native_module=%s native_abi=%s chess_library_commit=%s "
        "chess_library_header_sha256=%s"
        % (
            versions["python"],
            versions["python_chess_dist"],
            versions["python_chess_module"],
            versions["chess_rl_native_module"],
            versions["chess_rl_native_abi"],
            versions["chess_library_commit"],
            versions["chess_library_header_sha256"],
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
        for name, fen in DIRECTED_FIXTURES:
            print(f"{name}\t{fen}")
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
