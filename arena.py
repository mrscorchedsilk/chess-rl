"""Arena: head-to-head evaluation between two networks.

play_match(net_a, net_b, cfg, num_games) -> {'a': wins_a, 'b': wins_b, 'draws': draws}

Both nets search with temperature 0.0 and cfg.arena_simulations simulations.
Colors alternate game by game so each net plays White and Black equally often.
"""

import chess

from mcts import MCTS


def _terminal_result(board):
    """Return (is_terminal, white_result); same rules as selfplay."""
    if board.is_checkmate():
        return True, -1.0 if board.turn == chess.WHITE else 1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_repetition(3)
        or board.is_fifty_moves()
    ):
        return True, 0.0
    return False, None


def _play_arena_game(mcts_white, mcts_black, cfg, num_sims):
    """Play one arena game; return the result from White's perspective."""
    board = chess.Board()
    while True:
        if len(board.move_stack) >= cfg.max_game_length:
            return 0.0  # length cap -> draw
        terminal, white_result = _terminal_result(board)
        if terminal:
            return white_result

        searcher = mcts_white if board.turn == chess.WHITE else mcts_black
        pi = searcher.search(board, temperature=0.0, num_sims=num_sims)
        move = max(pi, key=pi.get)
        board.push(move)


def play_match(net_a, net_b, cfg, num_games):
    """Play num_games arena games between net_a and net_b.

    Colors alternate: net_a is White on even games, net_b is White on odd
    games. Returns {'a': wins_a, 'b': wins_b, 'draws': draws}.
    """
    mcts_a = MCTS(net_a, cfg)
    mcts_b = MCTS(net_b, cfg)
    wins_a = wins_b = draws = 0

    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            white_result = _play_arena_game(
                mcts_a, mcts_b, cfg, num_sims=cfg.arena_simulations
            )
        else:
            white_result = _play_arena_game(
                mcts_b, mcts_a, cfg, num_sims=cfg.arena_simulations
            )

        if white_result > 0.0:
            if game_idx % 2 == 0:
                wins_a += 1
            else:
                wins_b += 1
        elif white_result < 0.0:
            if game_idx % 2 == 0:
                wins_b += 1
            else:
                wins_a += 1
        else:
            draws += 1

    return {"a": wins_a, "b": wins_b, "draws": draws}
