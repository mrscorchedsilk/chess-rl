"""Self-play: generate one full game of AlphaZero-style training data.

play_game(net, cfg) -> list of (state, pi, z) training examples:

    state  (18, 8, 8) float32  -- encoding.encode_board(board)
    pi     (4096,)    float32  -- MCTS visit distribution as flat policy vector
    z      float in {-1.0, 0.0, 1.0} -- outcome from the side-to-move's view
"""

import chess
import numpy as np

import encoding
from mcts import MCTS


def _terminal_result(board):
    """Return (is_terminal, white_result) for the given board.

    white_result: 1.0 = White won, -1.0 = Black won, 0.0 = draw.
    Checkmate, stalemate, insufficient material, threefold repetition and
    the fifty-move rule all terminate the game.
    """
    if board.is_checkmate():
        # The side to move is checkmated, so the other side won.
        return True, -1.0 if board.turn == chess.WHITE else 1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_repetition(3)
        or board.is_fifty_moves()
    ):
        return True, 0.0
    return False, None


def _select_move(pi, temperature):
    """Pick the move to play from the MCTS distribution.

    temperature == 0.0 -> argmax (search already returns a single move);
    otherwise sample proportionally to the visit distribution.
    """
    if not pi:
        raise RuntimeError("MCTS returned an empty policy dict")
    if temperature == 0.0:
        return max(pi, key=pi.get)
    moves = list(pi.keys())
    probs = np.array([pi[m] for m in moves], dtype=np.float64)
    probs = probs / probs.sum()  # normalize defensively
    return moves[np.random.choice(len(moves), p=probs)]


def play_game(net, cfg):
    """Play one self-play game with MCTS and return the training examples.

    At each ply the search runs with temperature=cfg.temperature for the
    first cfg.temperature_threshold plies and temperature=0.0 afterwards.
    The game is capped at cfg.max_game_length plies (treated as a draw).
    """
    mcts = MCTS(net, cfg)
    board = chess.Board()
    examples = []  # (state, pi_vector, side_to_move); z is filled in once the game ends

    while True:
        if len(board.move_stack) >= cfg.max_game_length:
            break  # length cap -> draw
        terminal, _ = _terminal_result(board)
        if terminal:
            break

        ply = len(board.move_stack)
        temperature = cfg.temperature if ply < cfg.temperature_threshold else 0.0
        pi = mcts.search(board, temperature=temperature, num_sims=cfg.num_simulations)

        state = encoding.encode_board(board)
        examples.append((state, encoding.policy_to_vector(pi), board.turn))

        move = _select_move(pi, temperature)
        board.push(move)

    # Game over: compute the result from each side-to-move's perspective.
    terminal, white_result = _terminal_result(board)
    if not terminal:
        # Reached only via the max_game_length cap.
        white_result = 0.0

    out = []
    for state, pi_vec, side in examples:
        z = float(white_result if side == chess.WHITE else -white_result)
        out.append((state, pi_vec, z))
    return out
