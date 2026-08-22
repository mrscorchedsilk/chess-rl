import chess
import pytest

import chess_rl_native


START_FEN = chess.STARTING_FEN
KNIGHT_CYCLE = ("g1f3", "g8f6", "f3g1", "f6g8")


def expected_outcome(board: chess.Board, *, claim_draw: bool) -> dict | None:
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is None:
        return None
    return {
        "winner": {chess.WHITE: "white", chess.BLACK: "black", None: None}[outcome.winner],
        "termination": outcome.termination.name.lower(),
    }


def board_from_history(start_fen: str, moves: tuple[str, ...]) -> chess.Board:
    board = chess.Board(start_fen)
    for uci in moves:
        board.push_uci(uci)
    return board


def native_from_history(start_fen: str, moves: tuple[str, ...]):
    return chess_rl_native.Position.from_uci_history(start_fen, list(moves))


def expected_history_fens(board: chess.Board, max_steps: int = 8) -> list[str]:
    result = [board.fen()]
    copy = board.copy(stack=True)
    while copy.move_stack and len(result) < max_steps:
        copy.pop()
        result.append(copy.fen())
    return result


def test_from_uci_history_round_trips_complete_uci_history_and_fens_current_first():
    moves = ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5")
    board = board_from_history(START_FEN, moves)
    position = native_from_history(START_FEN, moves)

    assert position.fen() == board.fen()
    assert position.history_uci() == list(moves)
    assert position.history_fens() == expected_history_fens(board)
    assert position.history_fens(3) == expected_history_fens(board, 3)
    with pytest.raises((TypeError, ValueError)):
        position.history_fens(0)


def test_push_pop_preserves_history_repetition_and_history_fens():
    moves = KNIGHT_CYCLE + ("g1f3", "g8f6")
    position = native_from_history(START_FEN, moves)
    before_fen = position.fen()
    before_history = position.history_uci()
    before_fens = position.history_fens()

    position.pop()
    position.pop()
    assert position.history_uci() == list(moves[:-2])
    assert not position.is_repetition(3)

    position.push_uci("g1f3")
    position.push_uci("g8f6")
    assert position.fen() == before_fen
    assert position.history_uci() == before_history
    assert position.history_fens() == before_fens


@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        (
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            {"winner": "black", "termination": "checkmate"},
        ),
        ("7k/5Q2/7K/8/8/8/8/8 b - - 0 1", {"winner": None, "termination": "stalemate"}),
        (
            "8/8/8/8/8/8/8/K6k w - - 0 1",
            {"winner": None, "termination": "insufficient_material"},
        ),
    ],
)
def test_outcome_matches_python_chess_for_terminal_positions(fen, expected):
    board = chess.Board(fen)
    position = native_from_history(fen, ())
    assert expected_outcome(board, claim_draw=False) == expected
    assert position.outcome(claim_draw=False) == expected
    assert position.outcome(claim_draw=True) == expected


def test_repetition_uses_total_occurrences_and_complete_history_not_final_fen():
    twofold_moves = KNIGHT_CYCLE
    twofold_board = board_from_history(START_FEN, twofold_moves)
    twofold = native_from_history(START_FEN, twofold_moves)
    same_final_fen_without_history = chess_rl_native.Position.from_fen(twofold_board.fen())

    assert twofold.is_repetition(1)
    assert twofold.is_repetition(2)
    assert not twofold.is_repetition(3)
    assert not same_final_fen_without_history.is_repetition(2)
    assert twofold.outcome(claim_draw=True) == expected_outcome(twofold_board, claim_draw=True)

    threefold_moves = KNIGHT_CYCLE * 2
    threefold_board = board_from_history(START_FEN, threefold_moves)
    threefold = native_from_history(START_FEN, threefold_moves)
    assert threefold.is_repetition(3)
    assert threefold.outcome(claim_draw=False) == expected_outcome(threefold_board, claim_draw=False)
    assert threefold.outcome(claim_draw=True) == expected_outcome(threefold_board, claim_draw=True)

    fivefold_moves = KNIGHT_CYCLE * 4
    fivefold_board = board_from_history(START_FEN, fivefold_moves)
    fivefold = native_from_history(START_FEN, fivefold_moves)
    assert fivefold.is_repetition(5)
    assert fivefold.outcome(claim_draw=False) == expected_outcome(fivefold_board, claim_draw=False)
    assert fivefold.outcome(claim_draw=True) == expected_outcome(fivefold_board, claim_draw=True)


def test_threefold_claim_by_next_move_matches_python_chess():
    moves = KNIGHT_CYCLE + ("g1f3", "g8f6", "f3g1")
    board = board_from_history(START_FEN, moves)
    position = native_from_history(START_FEN, moves)

    assert not position.is_repetition(3)
    assert position.outcome(claim_draw=False) == expected_outcome(board, claim_draw=False) is None
    assert position.outcome(claim_draw=True) == expected_outcome(board, claim_draw=True) == {
        "winner": None,
        "termination": "threefold_repetition",
    }


@pytest.mark.parametrize(
    ("halfmove", "expected_without_claim", "expected_with_claim"),
    [
        (99, None, {"winner": None, "termination": "fifty_moves"}),
        (100, None, {"winner": None, "termination": "fifty_moves"}),
        (149, None, {"winner": None, "termination": "fifty_moves"}),
        (150, {"winner": None, "termination": "seventyfive_moves"}, {"winner": None, "termination": "seventyfive_moves"}),
    ],
)
def test_halfmove_boundaries_and_fifty_claim_by_next_move_match_python_chess(
    halfmove, expected_without_claim, expected_with_claim
):
    fen = f"8/8/8/8/8/8/R7/K6k w - - {halfmove} 1"
    board = chess.Board(fen)
    position = native_from_history(fen, ())
    assert expected_outcome(board, claim_draw=False) == expected_without_claim
    assert position.outcome(claim_draw=False) == expected_without_claim
    assert expected_outcome(board, claim_draw=True) == expected_with_claim
    assert position.outcome(claim_draw=True) == expected_with_claim


def test_checkmate_has_priority_over_seventyfive_moves():
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 150 3"
    board = chess.Board(fen)
    position = native_from_history(fen, ())
    expected = {"winner": "black", "termination": "checkmate"}
    assert expected_outcome(board, claim_draw=False) == expected
    assert position.outcome(claim_draw=False) == expected


@pytest.mark.parametrize(
    ("start_fen", "moves"),
    [
        ("not a fen", ()),
        (START_FEN, ("e2e5",)),
        (START_FEN, ("e2e4", "e7e5", "g1f9")),
        (START_FEN, ("e2e4 ",)),
    ],
)
def test_from_uci_history_rejects_invalid_start_or_complete_history(start_fen, moves):
    with pytest.raises((TypeError, ValueError)):
        native_from_history(start_fen, moves)
