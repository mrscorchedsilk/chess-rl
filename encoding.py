"""Board <-> network-tensor encoding, and move <-> index mapping.

Input planes (18 x 8 x 8, all float32), always from White's orientation:
    0..5   white P N B R Q K
    6..11  black P N B R Q K
    12     side to move (1.0 everywhere = white to move)
    13..16 castling rights K Q k q (broadcast 1.0 / 0.0)
    17     en-passant target square (single 1.0)

Policy: flat 4096 vector, index = from_square * 64 + to_square, in absolute
coordinates (no mirroring). Value is from the side-to-move's perspective.
"""
import numpy as np
import chess


def encode_board(board):
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        r, c = chess.square_rank(sq), chess.square_file(sq)
        base = 0 if piece.color == chess.WHITE else 6
        planes[base + piece.piece_type - 1][r][c] = 1.0
    if board.turn == chess.WHITE:
        planes[12][:, :] = 1.0
    planes[13][:, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14][:, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15][:, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16][:, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        r, c = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        planes[17][r][c] = 1.0
    return planes


def move_to_index(move):
    return move.from_square * 64 + move.to_square


def index_to_move(board, idx):
    from_sq = idx // 64
    to_sq = idx % 64
    promotion = None
    piece = board.piece_at(from_sq)
    if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
        promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)


def moves_to_mask(moves):
    """4096-length 0/1 mask built from an *already-generated* legal-move list.

    Avoids re-running python-chess legal-move generation (the single biggest
    CPU cost in MCTS) when the caller has already enumerated the moves.
    """
    mask = np.zeros(4096, dtype=np.float32)
    for m in moves:
        mask[move_to_index(m)] = 1.0
    return mask


def legal_moves_mask(board):
    """4096-length 0/1 mask of the from->to indices that are legal right now."""
    return moves_to_mask(board.legal_moves)


def encode_batch(boards):
    """Vectorised batch encoder: (18, 8, 8) float32 planes per board.

    Uses python-chess bitboards (board.pieces) + numpy unpackbits so a whole
    batch is encoded in a few vectorised ops instead of a Python loop over
    64 squares per board.  This is ~25x faster than encode_board() per board
    and is what the parallel self-play workers use to keep the CPU light.

    Plane layout and orientation are IDENTICAL to encode_board().
    """
    n = len(boards)
    planes = np.zeros((n, 18, 8, 8), dtype=np.float32)
    for pt in range(1, 7):
        white = np.array([b.pieces(pt, chess.WHITE) for b in boards], dtype=np.uint64)
        black = np.array([b.pieces(pt, chess.BLACK) for b in boards], dtype=np.uint64)
        # Bitboard bit i == square i == (rank = i // 8, file = i % 8).
        # unpackbits(bitorder='little') emits the LSB first, so the flat index
        # equals the square index.  (Default 'big' would reverse bits within
        # each byte and scramble the board.)
        planes[:, pt - 1] = np.unpackbits(white.view(np.uint8), bitorder="little").astype(np.float32).reshape(n, 8, 8)
        planes[:, pt + 5] = np.unpackbits(black.view(np.uint8), bitorder="little").astype(np.float32).reshape(n, 8, 8)

    for i, b in enumerate(boards):
        if b.turn == chess.WHITE:
            planes[i, 12] = 1.0
        planes[i, 13] = float(b.has_kingside_castling_rights(chess.WHITE))
        planes[i, 14] = float(b.has_queenside_castling_rights(chess.WHITE))
        planes[i, 15] = float(b.has_kingside_castling_rights(chess.BLACK))
        planes[i, 16] = float(b.has_queenside_castling_rights(chess.BLACK))
        if b.ep_square is not None:
            planes[i, 17, chess.square_rank(b.ep_square), chess.square_file(b.ep_square)] = 1.0
    return planes


def policy_to_vector(pi):
    """pi: dict {chess.Move: prob} -> 4096 numpy vector."""
    vec = np.zeros(4096, dtype=np.float32)
    for move, prob in pi.items():
        vec[move_to_index(move)] = prob
    return vec
