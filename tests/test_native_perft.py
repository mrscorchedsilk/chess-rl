import pytest

import chess_rl_native


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R "
    "w KQkq - 0 1"
)


def assert_perft(fen, depth, expected):
    actual = chess_rl_native.perft(fen, depth)
    if actual == expected:
        return
    root_divide = {}
    if depth > 0:
        position = chess_rl_native.Position.from_fen(fen)
        for move in position.legal_moves_uci():
            position.push_uci(move)
            root_divide[move] = chess_rl_native.perft(position.fen(), depth - 1)
            position.pop()
    pytest.fail(
        f"perft mismatch for depth {depth}: expected {expected}, got {actual}; "
        f"root divide={root_divide}"
    )


@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        (START_FEN, [1, 20, 400, 8902, 197281, 4865609]),
        (KIWIPETE_FEN, [None, 48, 2039, 97862, 4085603, 193690690]),
    ],
)
def test_perft_reference_vectors(fen, expected):
    for depth, count in enumerate(expected):
        if count is not None:
            assert_perft(fen, depth, count)


def test_perft_depth_is_nonnegative():
    with pytest.raises((TypeError, ValueError)):
        chess_rl_native.perft(START_FEN, -1)


def test_position_metadata_legal_moves_and_history():
    position = chess_rl_native.Position.from_fen(START_FEN)

    assert position.fen() == START_FEN
    assert position.side_to_move() == "w"
    assert position.halfmove_clock() == 0
    assert position.fullmove_number() == 1
    assert position.castling_rights() == "KQkq"
    assert position.ep_square() == "-"
    assert position.legal_moves_uci() == sorted(position.legal_moves_uci())
    assert len(position.legal_moves_uci()) == 20
    assert "e2e4" in position.legal_moves_uci()

    position.push_uci("e2e4")
    assert position.side_to_move() == "b"
    assert position.halfmove_clock() == 0
    assert position.fullmove_number() == 1
    assert position.ep_square() == "-"
    assert position.fen() == (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )

    position.push_uci("c7c5")
    assert position.side_to_move() == "w"
    assert position.fullmove_number() == 2
    assert position.ep_square() == "-"
    position.pop()
    assert position.fen().endswith(" b KQkq - 0 1")
    position.pop()
    assert position.fen() == START_FEN

    ep_position = chess_rl_native.Position.from_fen(
        "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"
    )
    assert ep_position.ep_square() == "e3"


def test_position_rejects_malformed_or_illegal_moves():
    position = chess_rl_native.Position.from_fen(START_FEN)
    for move in ("e2e5", "e2e4q", "E2E4", "e2e", "zzzz", "e1e3"):
        with pytest.raises((TypeError, ValueError)):
            position.push_uci(move)
    assert position.fen() == START_FEN


def test_position_from_fen_rejects_malformed_fen():
    with pytest.raises((TypeError, ValueError)):
        chess_rl_native.Position.from_fen("not a fen")


def test_perft_returns_python_int():
    result = chess_rl_native.perft(START_FEN, 5)
    assert type(result) is int
    assert result == 4_865_609
