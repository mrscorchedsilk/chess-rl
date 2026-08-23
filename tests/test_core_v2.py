"""Sprint B core tests — AlphaZero 73-plane / 4672-action mapping and 104-plane encoding.

Strict-TDD slices, run with:  .venv/bin/python -m pytest tests/test_core_v2.py -q
Slice 1 (this file, first revision): 4672 config contract, action-plane mapping
uniqueness, exact round-trip (promotions / castling / en passant), masks, and
policy probability-mass preservation.
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


# --------------------------------------------------------------------------- corpus

CORPUS_FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",          # both sides may castle
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "4k3/8/8/8/3pP3/8/8/4K3 w - d3 0 1",              # white en passant available
    "4k3/8/8/3Pp3/8/8/8/4K3 b - d6 0 1",              # black en passant available
    "8/P7/8/8/8/8/8/k6K w - - 0 1",                   # single white promotion pawn
    "8/p7/8/8/8/8/8/K6k b - - 0 1",                   # single black promotion pawn
    "8/PPP5/8/8/8/8/8/k6K w - - 0 1",                 # three promotion pawns, same rank
    "8/2P5/8/8/8/8/8/k6K w - - 0 1",
    "k7/8/8/8/8/8/1P6/K7 b - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "3k4/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/8/8/8/8/8/5k2/7K w - - 0 1",
]


def _random_corpus_boards(n_games=3, max_plies=40, seed=7):
    """Deterministic random-play boards (promotions/underpromotions can appear)."""
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n_games):
        b = chess.Board()
        boards.append(b.copy())
        for _ in range(max_plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(moves[int(rng.integers(len(moves)))])
            boards.append(b.copy())
    return boards


def _corpus_boards():
    boards = [chess.Board(fen) for fen in CORPUS_FENS]
    boards += _random_corpus_boards()
    return boards


# ----------------------------------------------------------- 4672 config contract

def test_config_4672_contract():
    cfg = Config()
    assert cfg.history_steps == 8
    assert cfg.num_input_planes == 104
    assert cfg.num_input_planes == 12 * cfg.history_steps + 8
    assert cfg.policy_planes == 73
    assert cfg.policy_size == 4672
    assert cfg.policy_size == cfg.policy_planes * 64
    assert cfg.replay_buffer_size == 50000
    assert cfg.selfplay_workers == 8
    assert cfg.checkpoint_every_iterations == 20
    assert cfg.result_timeout_seconds == 60
    assert cfg.training_epochs == 3
    assert cfg.arena_root_noise is False


def test_checkpoint_dir_is_v2():
    cfg = Config()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.normpath(cfg.checkpoint_dir) == os.path.normpath(
        os.path.join(root, "checkpoints", "v2")
    ), cfg.checkpoint_dir
    legacy = os.path.normpath(os.path.join(root, "checkpoints"))
    assert os.path.normpath(cfg.checkpoint_dir) != legacy
    assert os.path.normpath(cfg.checkpoint_dir).startswith(legacy + os.sep)


# ------------------------------------------------------------- action mapping

def test_move_to_index_layout():
    """Index = from_square * 73 + plane; plane = 56 queen-like + 8 knight + 9 underpromo."""
    # queen-like planes: 8 directions x 7 distances
    assert encoding.move_to_index(chess.Move.from_uci("e2e4")) == 12 * 73 + 1   # N, dist 2
    assert encoding.move_to_index(chess.Move.from_uci("e1e8")) == 4 * 73 + 6    # N, dist 7
    assert encoding.move_to_index(chess.Move.from_uci("a1a8")) == 0 * 73 + 6
    assert encoding.move_to_index(chess.Move.from_uci("e4f5")) == 28 * 73 + 7   # NE, dist 1
    assert encoding.move_to_index(chess.Move.from_uci("e4h7")) == 28 * 73 + 9   # NE, dist 3
    assert encoding.move_to_index(chess.Move.from_uci("e4d3")) == 28 * 73 + 35  # SW, dist 1
    assert encoding.move_to_index(chess.Move.from_uci("b2b4")) == 9 * 73 + 1
    assert encoding.move_to_index(chess.Move.from_uci("a1h8")) == 0 * 73 + 13   # NE, dist 7
    # castling rides the queen-like planes (east/west, dist 2)
    assert encoding.move_to_index(chess.Move.from_uci("e1g1")) == 4 * 73 + 15
    assert encoding.move_to_index(chess.Move.from_uci("e1c1")) == 4 * 73 + 43
    assert encoding.move_to_index(chess.Move.from_uci("e8g8")) == 60 * 73 + 15
    # knight planes 56..63
    assert encoding.move_to_index(chess.Move.from_uci("e4f6")) == 28 * 73 + 57
    assert encoding.move_to_index(chess.Move.from_uci("e4g5")) == 28 * 73 + 56
    assert encoding.move_to_index(chess.Move.from_uci("e4d6")) == 28 * 73 + 58
    assert encoding.move_to_index(chess.Move.from_uci("e4c3")) == 28 * 73 + 60
    # underpromotion planes 64..72: piece*3 + (file_delta + 1)
    assert encoding.move_to_index(chess.Move.from_uci("a7a8n")) == 48 * 73 + 65  # N straight
    assert encoding.move_to_index(chess.Move.from_uci("a7b8b")) == 48 * 73 + 69  # B +1 file
    assert encoding.move_to_index(chess.Move.from_uci("a7b8r")) == 48 * 73 + 72  # R +1 file
    assert encoding.move_to_index(chess.Move.from_uci("h7g8r")) == 55 * 73 + 70  # R -1 file
    assert encoding.move_to_index(chess.Move.from_uci("b7a8n")) == 49 * 73 + 64  # N -1 file
    # queen promotion uses the normal queen-like plane
    assert encoding.move_to_index(chess.Move.from_uci("a7a8q")) == 48 * 73 + 0   # N, dist 1
    assert encoding.move_to_index(chess.Move.from_uci("a7b8q")) == 48 * 73 + 7   # NE, dist 1
    # en passant: plain diagonal queen-like plane (e5->d6 is NW, plane 49)
    assert encoding.move_to_index(chess.Move.from_uci("e5d6")) == 36 * 73 + 49


def test_move_to_index_is_injective_over_legal_moves():
    """Two distinct legal moves must never share an index (uniqueness)."""
    for b in _corpus_boards():
        idxs = [encoding.move_to_index(m) for m in b.legal_moves]
        assert len(set(idxs)) == len(idxs), f"collision in {b.fen()}"
        for i in idxs:
            assert 0 <= i < 4672


def test_roundtrip_all_legal_moves():
    """index_to_move(board, move_to_index(m)) == m for EVERY legal move."""
    for b in _corpus_boards():
        for m in b.legal_moves:
            m2 = encoding.index_to_move(b, encoding.move_to_index(m))
            assert m2 == m, f"{m.uci()} -> {m2.uci()} in {b.fen()}"


def test_roundtrip_promotions_castling_en_passant():
    # every promotion piece round-trips with its piece type intact
    b = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    for m in b.legal_moves:
        assert encoding.index_to_move(b, encoding.move_to_index(m)) == m
    for uci, promo in (("a7a8q", chess.QUEEN), ("a7a8n", chess.KNIGHT),
                       ("a7a8b", chess.BISHOP), ("a7a8r", chess.ROOK),
                       ("a7b8n", chess.KNIGHT), ("a7b8r", chess.ROOK)):
        m = chess.Move.from_uci(uci)
        m2 = encoding.index_to_move(b, encoding.move_to_index(m))
        assert m2 == m and m2.promotion == promo, (uci, m2)
    # black promotion
    b = chess.Board("8/p7/8/8/8/8/8/K6k b - - 0 1")
    for m in b.legal_moves:
        assert encoding.index_to_move(b, encoding.move_to_index(m)) == m
    # multi-pawn same-rank promotions stay unique and round-trip
    b = chess.Board("8/PPP5/8/8/8/8/8/k6K w - - 0 1")
    for m in b.legal_moves:
        assert encoding.index_to_move(b, encoding.move_to_index(m)) == m
    # castling both directions, both colors
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    for m in b.legal_moves:
        assert encoding.index_to_move(b, encoding.move_to_index(m)) == m
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    for m in b.legal_moves:
        assert encoding.index_to_move(b, encoding.move_to_index(m)) == m
    # en passant both colors
    for fen in ("4k3/8/8/8/3pP3/8/8/4K3 w - d3 0 1",
                "4k3/8/8/3Pp3/8/8/8/4K3 b - d6 0 1"):
        b = chess.Board(fen)
        for m in b.legal_moves:
            assert encoding.index_to_move(b, encoding.move_to_index(m)) == m


# ------------------------------------------------------------------- masks / mass

def test_masks_cover_legal_moves_exactly():
    for b in _corpus_boards():
        mask = encoding.legal_moves_mask(b)
        assert mask.shape == (4672,)
        assert mask.dtype == np.float32
        idxs = [encoding.move_to_index(m) for m in b.legal_moves]
        for m, i in zip(b.legal_moves, idxs):
            assert mask[i] == 1.0, f"legal move {m} not masked in {b.fen()}"
        assert int(mask.sum()) == len(idxs), f"mask mass wrong in {b.fen()}"
        assert np.array_equal(mask, encoding.moves_to_mask(b.legal_moves))


def test_policy_mass_preserved():
    """policy_to_vector preserves probability mass exactly (no shared planes)."""
    b = chess.Board("8/PPP5/8/8/8/8/8/k6K w - - 0 1")  # 12 promo moves + king moves
    legal = list(b.legal_moves)
    pi = {m: 1.0 / len(legal) for m in legal}
    vec = encoding.policy_to_vector(pi)
    assert vec.shape == (4672,)
    assert abs(float(vec.sum()) - 1.0) < 1e-5
    assert int((vec > 0).sum()) == len(legal)  # one distinct plane per move
    for m in legal:
        assert vec[encoding.move_to_index(m)] > 0.0


# --------------------------------------------------- 104-plane board encoding

def test_encode_board_shape_and_start_position():
    b = chess.Board()
    p = encoding.encode_board(b)
    assert p.shape == (104, 8, 8)
    assert p.dtype == np.float32
    # white king e1 (square 4 = rank 0, file 4) -> plane 5 of step 0
    assert p[5][0][4] == 1.0
    # black king e8 (square 60 = rank 7, file 4) -> plane 11 of step 0
    assert p[11][7][4] == 1.0
    # side-to-move plane 96: all ones while white to move
    assert p[96].sum() == 64.0
    # castling-rights planes 97..100 (K Q k q): all 1.0 at the start position
    for pl in (97, 98, 99, 100):
        assert p[pl].sum() == 64.0, pl
    # en-passant 101, halfmove 102, repetition 103 are all zero at start
    for pl in (101, 102, 103):
        assert p[pl].sum() == 0.0, pl
    # history steps 1..7 (planes 12..95) are zero-padded
    assert p[12:96].sum() == 0.0


def test_history_exactly_padded_to_8_steps():
    """8-step history: most recent first, older positions zero-padded at game start."""
    hist = encoding._history_boards(chess.Board(), 8)
    assert len(hist) == 8
    assert hist[0] is not None
    assert all(h is None for h in hist[1:])

    b = chess.Board()
    b.push_san("e4")
    b.push_san("e5")
    b.push_san("Nf3")
    p = encoding.encode_board(b)
    assert p.shape == (104, 8, 8)
    # step 0 = current: white Nf3 -> plane 1 at rank 2, file 5
    assert p[0 + 1][2][5] == 1.0
    # step 1 = after e5: black pawn e5 -> plane 6 at rank 4, file 4
    assert p[12 + 6][4][4] == 1.0
    # step 2 = after e4: white pawn e4 -> plane 0 at rank 3, file 4
    assert p[24 + 0][3][4] == 1.0
    # step 3 = the start position (identical piece planes to a fresh board)
    start = encoding.encode_board(chess.Board())
    assert np.array_equal(p[36:48], start[0:12])
    # steps 4..7 are all-zero padding
    assert p[48:96].sum() == 0.0


def test_meta_plane_details():
    # en passant d3, halfmove clock 47, black to move
    b = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - d3 47 100")
    p = encoding.encode_board(b)
    assert p[96].sum() == 0.0                                   # black to move
    assert p[101].sum() == 1.0 and p[101][2][3] == 1.0          # ep d3 (rank 2, file 3)
    assert abs(p[102][0][0] - 0.47) < 1e-5
    assert abs(p[102].sum() - 64.0 * 0.47) < 1e-4
    # castling-rights plane order is K, Q, k, q
    b2 = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")
    p2 = encoding.encode_board(b2)
    assert p2[97].sum() == 64.0 and p2[98].sum() == 0.0
    assert p2[99].sum() == 0.0 and p2[100].sum() == 64.0
    # repetition flag: 1.0 once the position has occurred before
    b3 = chess.Board()
    b3.push_uci("g1f3"); b3.push_uci("g8f6")
    b3.push_uci("f3g1"); b3.push_uci("f6g8")
    assert b3.is_repetition(2)
    assert encoding.encode_board(b3)[103].sum() == 64.0


def test_bb_to_plane_numpy2_safe():
    """Bitboard conversion must not rely on 0d uint64 .view(uint8) (NumPy >= 2 rejects it)."""
    pl = encoding._bb_to_plane(chess.Board().pieces(chess.PAWN, chess.WHITE))
    assert pl.shape == (8, 8) and pl.dtype == np.float32
    # a2 = square 8 (rank 1, file 0); b2 = square 9 (rank 1, file 1)
    assert pl[1][0] == 1.0 and pl[1][1] == 1.0
    assert int(pl.sum()) == 8  # eight white pawns at start
    assert encoding._bb_to_plane(0).sum() == 0.0
    assert encoding._bb_to_plane(0xFFFFFFFFFFFFFFFF).sum() == 64.0
    # every piece set must convert without raising
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    for pt in range(1, 7):
        encoding._bb_to_plane(b.pieces(pt, chess.WHITE))
        encoding._bb_to_plane(b.pieces(pt, chess.BLACK))


def test_encode_planes_match_piece_bitboards():
    """Plane content must match python-chess bitboards exactly (orientation proof)."""
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    p = encoding.encode_board(b)
    for step in range(8):
        off = step * 12
        if p[off:off + 12].sum() == 0.0 and step > 0:
            continue  # zero-padded history
        hb = encoding._history_boards(b, 8)[step]
        if hb is None:
            continue
        for pt in range(1, 7):
            for color, base in ((chess.WHITE, 0), (chess.BLACK, 6)):
                bb = hb.pieces(pt, color)
                plane = p[off + base + pt - 1]
                assert int(plane.sum()) == int(bb).bit_count(), (step, pt, color)
                for sq in chess.scan_forward(int(bb)):
                    assert plane[chess.square_rank(sq)][chess.square_file(sq)] == 1.0


def test_scalar_batch_encoder_equivalence():
    boards = _corpus_boards()
    batch = encoding.encode_batch(boards)
    assert batch.shape == (len(boards), 104, 8, 8)
    for i, b in enumerate(boards):
        assert np.array_equal(batch[i], encoding.encode_board(b)), b.fen()


def test_scalar_batch_encoder_equivalence_randomized():
    """Randomised scalar/batch equivalence over games of varying history depth."""
    rng = np.random.default_rng(20260822)
    boards = []
    for _ in range(4):
        b = chess.Board()
        boards.append(b.copy())
        for _ in range(60):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(moves[int(rng.integers(len(moves)))])
            boards.append(b.copy())
    batch = encoding.encode_batch(boards)
    for i, b in enumerate(boards):
        assert np.array_equal(batch[i], encoding.encode_board(b)), b.fen()
    # non-default history depth must agree too
    batch4 = encoding.encode_batch(boards, history_steps=4)
    for i, b in enumerate(boards):
        assert np.array_equal(batch4[i], encoding.encode_board(b, history_steps=4)), b.fen()


# --------------------------------------------------------- model head contract

def _cpu_net(seed=0):
    torch.manual_seed(seed)
    cfg = Config()
    cfg.device = "cpu"
    return ChessNet(cfg).to("cpu").eval(), cfg


def test_model_policy_head_is_spatial_conv2d():
    """The policy head must be a true spatial Conv2d (73 out planes), no linear head."""
    net, _ = _cpu_net()
    assert isinstance(net.policy_conv, torch.nn.Conv2d)
    assert net.policy_conv.out_channels == 73
    assert net.policy_conv.kernel_size == (1, 1)
    assert not hasattr(net, "policy_fc")


def test_model_nhwc_flatten_order_from_square_times_73():
    """policy_logits[b, from_square * 73 + plane] == spatial conv at (plane, rank, file)."""
    net, _ = _cpu_net()
    x = torch.randn(1, 104, 8, 8)
    with torch.no_grad():
        logits, _ = net(x)
        body = net.body(x)
        conv = net.policy_conv(body)
    assert tuple(logits.shape) == (1, 4672)
    assert tuple(conv.shape) == (1, 73, 8, 8)
    # NHWC flatten: square-major, plane-minor
    expected = conv.permute(0, 2, 3, 1).reshape(1, -1)
    assert torch.equal(logits, expected)
    # explicit from_square * 73 + plane identity on a few squares
    lr = logits[0]
    for sq in (0, 4, 28, 63):
        r, f = chess.square_rank(sq), chess.square_file(sq)
        for plane in (0, 7, 49, 56, 64, 72):
            assert lr[sq * 73 + plane] == conv[0, plane, r, f], (sq, plane)


def test_model_head_order_and_value_range():
    """forward returns (policy_logits, value) — policy first — with correct shapes."""
    net, _ = _cpu_net()
    x = torch.randn(4, 104, 8, 8)
    out = net(x)
    assert isinstance(out, tuple) and len(out) == 2
    logits, value = out
    assert tuple(logits.shape) == (4, 4672)
    assert tuple(value.shape) == (4, 1)
    assert float(value.min().detach()) >= -1.0 and float(value.max().detach()) <= 1.0
