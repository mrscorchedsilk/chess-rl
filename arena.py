"""Arena: head-to-head evaluation between two networks.

play_match(net_a, net_b, cfg, num_games) -> {'a': wins_a, 'b': wins_b, 'draws': draws}

Both nets search with temperature 0.0 and cfg.arena_simulations simulations.

Arena diversity comes from a DETERMINISTIC paired-opening suite, not from
stochastic evaluation noise: for ``num_games`` total games, ``num_games // 2``
distinct shallow openings are generated from a stable ``cfg.arena_seed``, and
each opening is played twice with the candidate/champion colors swapped.  This
raises the effective sample size of a nominally-20-game arena from ~2 repeated
trajectories to 10 distinct starting positions while staying reproducible
across candidate evaluations.  Root Dirichlet noise stays OFF
(cfg.arena_root_noise=False) and temperature stays 0.0.
"""

import hashlib
import random

import chess

from mcts import MCTS
import telemetry


def _terminal_result(board):
    """Return (is_terminal, white_result); same rules as selfplay and MCTS
    (outcome(claim_draw=True) — claimable draws are terminal everywhere)."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return False, None
    if outcome.winner is None:
        return True, 0.0
    return True, 1.0 if outcome.winner == chess.WHITE else -1.0


# --------------------------------------------------------------------------- #
#  deterministic paired-opening suite                                         #
# --------------------------------------------------------------------------- #

def _random_opening(rng, opening_plies):
    """One legal random opening of exactly ``opening_plies`` plies, or None if
    the position terminates before the requested depth (caller regenerates)."""
    board = chess.Board()
    seq = []
    for _ in range(opening_plies):
        # Sort legal moves by UCI before indexing so the sequence is fully
        # deterministic for a given rng state (no unordered-collection drift).
        moves = sorted(board.legal_moves, key=lambda m: m.uci())
        if not moves:
            return None
        move = moves[rng.randrange(len(moves))]
        seq.append(move.uci())
        board.push(move)
        if board.is_game_over(claim_draw=True):
            return None  # terminal before requested depth -> regenerate
    return seq


def generate_arena_openings(num_pairs, opening_plies, seed):
    """``num_pairs`` distinct legal opening move sequences from a stable seed.

    Uses a dedicated ``random.Random(seed)`` (never the global training RNG, so
    candidate/champion weights and the self-play iteration seed cannot affect
    it).  Sequences are replayed as UCI moves from the standard start position
    (not FEN strings) so the eight-position history stack, castling, halfmove
    and repetition state are preserved exactly.
    """
    rng = random.Random(seed)
    openings = []
    seen = set()
    while len(openings) < num_pairs:
        seq = _random_opening(rng, opening_plies)
        if seq is None:
            continue
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        openings.append(seq)
    return openings


def arena_suite_hash(openings):
    """Stable digest of an opening suite (for arena-event observability)."""
    h = hashlib.blake2b(digest_size=16)
    for seq in openings:
        for uci in seq:
            h.update(uci.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  game / match                                                               #
# --------------------------------------------------------------------------- #

def _play_arena_game(mcts_white, mcts_black, cfg, num_sims, opening_moves=()):
    """Play one arena game from ``opening_moves``; return result from White's
    perspective.

    Arena searches use temperature 0.0 and NO Dirichlet root noise
    (cfg.arena_root_noise is False), so repeated matches are deterministic.
    ``opening_moves`` is a sequence of legal UCI moves pushed onto the board
    (history stack) BEFORE search begins.
    """
    board = chess.Board()
    for uci in opening_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal opening move {uci}")
        board.push(move)
    while True:
        if len(board.move_stack) >= cfg.max_game_length:
            return 0.0  # length cap -> draw
        terminal, white_result = _terminal_result(board)
        if terminal:
            return white_result

        searcher = mcts_white if board.turn == chess.WHITE else mcts_black
        pi = searcher.search(
            board, temperature=0.0, num_sims=num_sims,
            add_root_noise=cfg.arena_root_noise,
        )
        move = max(pi, key=pi.get)
        board.push(move)


def play_match(net_a, net_b, cfg, num_games, openings=None):
    """Play ``num_games`` arena games between net_a and net_b.

    ``num_games`` must be even: ``num_games // 2`` distinct deterministic
    openings are generated (or supplied via ``openings``) and each is played
    twice — Game A with net_a as White, Game B with net_b as White — so both
    nets face every opening from both sides.  Returns
    {'a': wins_a, 'b': wins_b, 'draws': draws}, unchanged.
    """
    if num_games % 2 != 0:
        raise ValueError(f"arena_games must be even (got {num_games})")
    mcts_a = MCTS(net_a, cfg)
    mcts_b = MCTS(net_b, cfg)
    wins_a = wins_b = draws = 0

    num_pairs = num_games // 2
    if openings is None:
        openings = generate_arena_openings(
            num_pairs,
            int(getattr(cfg, "arena_opening_plies", 8)),
            int(getattr(cfg, "arena_seed", 424242)),
        )

    # Telemetry is a pure SIDE EFFECT: the match loop and the return contract
    # {"a","b","draws"} are unchanged (existing tests pin the contract).
    with telemetry.PhaseTimer("arena") as _arena_timer:
        for opening_moves in openings:
            # Game A: candidate (net_a) White, champion (net_b) Black.
            white_result = _play_arena_game(
                mcts_a, mcts_b, cfg, num_sims=cfg.arena_simulations,
                opening_moves=opening_moves,
            )
            if white_result > 0.0:
                wins_a += 1
            elif white_result < 0.0:
                wins_b += 1
            else:
                draws += 1

            # Game B: colors swapped.
            white_result = _play_arena_game(
                mcts_b, mcts_a, cfg, num_sims=cfg.arena_simulations,
                opening_moves=opening_moves,
            )
            if white_result > 0.0:
                wins_b += 1
            elif white_result < 0.0:
                wins_a += 1
            else:
                draws += 1
    try:
        telemetry.safe_emit(cfg, {
            "type": "phase",
            "phase": "arena",
            "duration_s": _arena_timer.duration_s,
            "arena_games": int(num_games),
            "arena_sims": int(getattr(cfg, "arena_simulations", 0)),
        })
    except Exception:  # noqa: BLE001 - telemetry must never kill training
        pass

    return {"a": wins_a, "b": wins_b, "draws": draws}
