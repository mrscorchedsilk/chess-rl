"""Board <-> network-tensor encoding, and move <-> action-index mapping (Sprint B).

Input planes (104 x 8 x 8, all float32), always from White's orientation:

  * 8 history positions, most recent FIRST (current position = step 0), each
    12 piece planes: 0..5 white P N B R Q K, 6..11 black P N B R Q K.
    Positions before the start of the game are empty (all-zero) planes.
  * 96     side to move (1.0 everywhere = white to move)
  * 97..100 castling rights K Q k q (broadcast 1.0 / 0.0)
  * 101    en-passant target square (single 1.0)
  * 102    halfmove clock / 100 (broadcast, in [0, 1])
  * 103    repetition indicator (1.0 if this position occurred before)

Policy: AlphaZero 73-plane action map, flat index = from_square * 73 + plane,
4672 entries.  The plane tells the piece ON `from_square` how to move:

  * planes  0..55  queen-like moves: 8 directions x 7 distances
                   (direction index = plane // 7, distance = plane % 7 + 1)
  * planes 56..63  knight moves (8 offsets)
  * planes 64..72  underpromotions: piece (N=0, B=1, R=2) * 3 + (to_file -
                   from_file + 1); the from square is the promoting pawn.

Queen promotions are NOT a separate plane: they use the ordinary queen-like
plane for the pawn's direction/distance.  Because the from square is part of
the index, the mapping is injective over every legal move of every position
(including several pawns underpromoting to the same piece on the same rank),
so masks and policy targets preserve probability mass exactly.

Value is from the side-to-move's perspective.
"""
import numpy as np
import chess

# (dr, df) deltas, rank 0 = white's back rank.  Direction order is the plane order.
QUEEN_DIRECTIONS = (
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
)
KNIGHT_OFFSETS = (
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
PROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

QUEEN_PLANES = 56
KNIGHT_PLANES = 8
UNDERPROMOTION_PLANES = 9
POLICY_PLANES = QUEEN_PLANES + KNIGHT_PLANES + UNDERPROMOTION_PLANES  # 73
POLICY_SIZE = 64 * POLICY_PLANES                                     # 4672


# ------------------------------------------------------------------ action mapping

def move_to_index(move):
    """Map a chess.Move to its 4672 action index = from_square * 73 + plane.

    Injective over all legal moves of any position (promotions, castling and
    en passant included); queen promotion rides the queen-like plane.
    """
    from_sq, to_sq = move.from_square, move.to_square
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    df = chess.square_file(to_sq) - chess.square_file(from_sq)

    promo = move.promotion
    if promo is not None and promo != chess.QUEEN:
        plane = QUEEN_PLANES + KNIGHT_PLANES + PROMOTION_PIECES.index(promo) * 3 + (df + 1)
        return from_sq * POLICY_PLANES + plane

    try:
        ki = KNIGHT_OFFSETS.index((dr, df))
    except ValueError:
        ki = -1
    if ki >= 0:
        return from_sq * POLICY_PLANES + QUEEN_PLANES + ki

    di = QUEEN_DIRECTIONS.index((dr // dist, df // dist)) if (dist := max(abs(dr), abs(df))) else -1
    plane = di * 7 + (dist - 1)
    return from_sq * POLICY_PLANES + plane


def index_to_move(board, idx):
    """Invert an action index for `board` (requires the position: the index
    carries the from square and the plane, but promotion legality is inferred
    from the board).  For every legal move m, this returns m exactly."""
    from_sq, plane = divmod(idx, POLICY_PLANES)
    if plane < QUEEN_PLANES:
        di, dist = divmod(plane, 7)
        dist += 1
        dr, df = QUEEN_DIRECTIONS[di]
        to_sq = from_sq + (dr * 8 + df) * dist
        piece = board.piece_at(from_sq)
        promo = None
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_rank(to_sq) in (0, 7)
        ):
            promo = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promo)
    if plane < QUEEN_PLANES + KNIGHT_PLANES:
        dr, df = KNIGHT_OFFSETS[plane - QUEEN_PLANES]
        return chess.Move(from_sq, from_sq + dr * 8 + df)
    p = plane - QUEEN_PLANES - KNIGHT_PLANES
    piece = PROMOTION_PIECES[p // 3]
    df = p % 3 - 1
    to_rank = 7 if board.turn == chess.WHITE else 0
    to_sq = chess.square(chess.square_file(from_sq) + df, to_rank)
    return chess.Move(from_sq, to_sq, promotion=piece)


def moves_to_mask(moves):
    """4672-length 0/1 mask built from an *already-generated* legal-move list.

    Avoids re-running python-chess legal-move generation (the single biggest
    CPU cost in MCTS) when the caller has already enumerated the moves.
    """
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    for m in moves:
        mask[move_to_index(m)] = 1.0
    return mask


def legal_moves_mask(board):
    """4672-length 0/1 mask of the action indices that are legal right now."""
    return moves_to_mask(board.legal_moves)


def policy_to_vector(pi):
    """pi: dict {chess.Move: prob} -> 4672 numpy vector (mass preserved exactly)."""
    vec = np.zeros(POLICY_SIZE, dtype=np.float32)
    for move, prob in pi.items():
        vec[move_to_index(move)] += prob
    return vec


# ------------------------------------------------------------- board encoding

def _history_boards(board, history_steps):
    """Last `history_steps` positions, most recent first; None pads the start.

    Always returns exactly `history_steps` entries so both the scalar and the
    batched encoders can index positions uniformly: the last entries are None
    for positions before the start of the game (encoded as all-zero planes).
    """
    out = []
    b = board
    while len(out) < history_steps:
        out.append(b)
        if not b.move_stack:
            break
        b = b.copy()
        b.pop()
    out.extend([None] * (history_steps - len(out)))
    return out


def _bb_to_plane(bitboard):
    """Bitboard -> (8, 8) float32 plane (row = rank, col = file).

    NumPy >= 2 refuses to change the itemsize of a 0-d array via .view(), so
    the scalar path goes through tobytes()/frombuffer() instead of the
    uint64.view(uint8) trick that works for 1-d batch arrays.
    """
    return np.unpackbits(
        np.frombuffer(np.uint64(bitboard).tobytes(), dtype=np.uint8),
        bitorder="little",
    ).astype(np.float32).reshape(8, 8)


def _encode_meta(planes, i, board, meta):
    """Fill the 8 meta planes (side, castling, ep, halfmove, repetition)."""
    planes[i, meta + 0] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[i, meta + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[i, meta + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[i, meta + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[i, meta + 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        planes[i, meta + 5, chess.square_rank(board.ep_square),
               chess.square_file(board.ep_square)] = 1.0
    planes[i, meta + 6] = board.halfmove_clock / 100.0
    planes[i, meta + 7] = 1.0 if board.is_repetition(2) else 0.0


def encode_board(board, history_steps=8):
    """Single-board encoder: (12*history_steps + 8, 8, 8) float32 planes."""
    planes = np.zeros((12 * history_steps + 8, 8, 8), dtype=np.float32)
    for step, hb in enumerate(_history_boards(board, history_steps)):
        if hb is None:
            continue
        off = step * 12
        for pt in range(1, 7):
            planes[off + pt - 1] = _bb_to_plane(hb.pieces(pt, chess.WHITE))
            planes[off + pt + 5] = _bb_to_plane(hb.pieces(pt, chess.BLACK))
    _encode_meta(planes[None], 0, board, 12 * history_steps)
    return planes


def encode_batch(boards, history_steps=8):
    """Vectorised batch encoder: (n, 12*history_steps + 8, 8, 8) float32.

    Piece planes are built from python-chess bitboards + numpy unpackbits (a
    whole history step across the batch in a few vectorised ops); meta planes
    are filled per board.  Results are bit-for-bit IDENTICAL to encode_board().
    """
    n = len(boards)
    planes = np.zeros((n, 12 * history_steps + 8, 8, 8), dtype=np.float32)
    all_hist = [_history_boards(b, history_steps) for b in boards]
    for step in range(history_steps):
        off = step * 12
        have = [i for i in range(n) if all_hist[i][step] is not None]
        if not have:
            continue
        sub = [all_hist[i][step] for i in have]
        for pt in range(1, 7):
            white = np.array([s.pieces(pt, chess.WHITE) for s in sub], dtype=np.uint64)
            black = np.array([s.pieces(pt, chess.BLACK) for s in sub], dtype=np.uint64)
            planes[have, off + pt - 1] = np.unpackbits(
                white.view(np.uint8), bitorder="little"
            ).astype(np.float32).reshape(len(have), 8, 8)
            planes[have, off + pt + 5] = np.unpackbits(
                black.view(np.uint8), bitorder="little"
            ).astype(np.float32).reshape(len(have), 8, 8)
    meta = 12 * history_steps
    for i, b in enumerate(boards):
        _encode_meta(planes, i, b, meta)
    return planes
