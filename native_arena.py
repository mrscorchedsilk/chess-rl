"""Native-arena adapter: replace the Python-MCTS search inside the arena gate
with the native `chess_rl_native.MCTS` core (docs/native-arena-design.md).

The game *semantics* of the arena are identical to ``arena.py``: the
deterministic paired-opening suite (``arena_seed=424242``, 8 plies), color
swaps, temperature 0, no root noise, 20 games at 40 simulations, and the
``{"a", "b", "draws"}`` return contract.  Only the *search engine* changes:
each ply's search is driven through the two-phase native API
(``set_root`` / ``gather_leaves`` / ``apply_evaluations`` / ``policy``)
against a persistent per-network ``InferenceRuntime``, while the GAME-level
adjudication (length cap + ``board.outcome(claim_draw=True)``) stays in
python-chess (``arena._terminal_result``).

Two MCTS objects (``mcts_a`` / ``mcts_b``) are created once per ``play_match``
and reused across every game/ply (``set_root`` performs the full reset), and
each network gets exactly one persistent ``InferenceRuntime`` for the life of a
run (``NativeArenaEngine``), so ``torch.compile`` is paid once, not per gate.

The simulation budget is tracked from the caller's ``num_sims``
(``cfg.arena_simulations``) — NOT from the un-exposed ``mcts.num_simulations``
(see design §1 defect; verified ``AttributeError``).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import chess
import chess_rl_native as native

from arena import generate_arena_openings, _terminal_result, arena_suite_hash
import telemetry

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# (inputs[B,104,8,8] f32, offsets[B+1] i32, indices[K] i32)
#   -> (legal_logits[K] f32, values[B,1] f32)     — exactly InferenceRuntime.evaluate
InferenceFn = Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _new_mcts(cfg, num_sims: int, seed: int) -> native.MCTS:
    """One native MCTS per side. Root noise is OFF for arena (production
    cfg.arena_root_noise=False), so dirichlet_epsilon is forced to 0.0; the
    seed is therefore inert (no RNG is ever drawn)."""
    eps = float(cfg.dirichlet_epsilon) if bool(getattr(cfg, "arena_root_noise", False)) else 0.0
    return native.MCTS(
        c_puct=float(cfg.c_puct),
        virtual_loss=float(cfg.virtual_loss),
        num_simulations=int(num_sims),
        dirichlet_alpha=float(cfg.dirichlet_alpha),
        dirichlet_epsilon=eps,
        seed=int(seed),
    )


def _run_native_search(
    mcts: native.MCTS,
    start_fen: str,
    history_moves: Sequence[str],
    evaluate: InferenceFn,
    num_sims: int,
    max_batch: int = 256,
) -> list[tuple[str, float]]:
    """Drive the two-phase API to completion; return policy(0.0).

    The simulation budget is tracked from `num_sims` (NOT from the un-exposed
    `mcts.num_simulations`; see design §1 defect)."""
    mcts.set_root(start_fen, list(history_moves))
    guard = int(num_sims) + 8
    while not mcts.is_complete():
        guard -= 1
        if guard < 0:
            raise RuntimeError("native arena search failed to terminate")
        tokens, inputs, offsets, indices = mcts.gather_leaves(int(max_batch))
        if not tokens:                      # internal terminal batch; sims still ran
            continue
        logits, values = evaluate(
            np.asarray(inputs), np.asarray(offsets), np.asarray(indices)
        )
        mcts.apply_evaluations(tokens, offsets, logits, values)
    return mcts.policy(0.0)


def _select_move(policy: list[tuple[str, float]]) -> str:
    """temperature-0 policy is one-hot on the most-visited move; max-prob is that
    move. (Ties already broken by native policy() in ascending-action-index
    order, matching np.argmax semantics.)"""
    if not policy:  # defensive: the game loop never searches a terminal root
        raise RuntimeError("empty policy from a non-terminal root")
    return max(policy, key=lambda p: p[1])[0]


def _play_native_game(
    mcts_white: native.MCTS,
    mcts_black: native.MCTS,
    evaluate_white: InferenceFn,
    evaluate_black: InferenceFn,
    cfg,
    num_sims: int,
    opening_moves: Sequence[str] = (),
    max_batch: int = 256,
) -> float:
    """Play one game from `opening_moves`; return result from White's
    perspective. Mirrors arena._play_arena_game, with the search replaced by
    native MCTS and the GAME-level adjudication still done by python-chess."""
    board = chess.Board()
    for uci in opening_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal opening move {uci}")
        board.push(move)
    while True:
        if len(board.move_stack) >= int(cfg.max_game_length):
            return 0.0                                   # length cap -> draw
        terminal, white_result = _terminal_result(board)  # python-chess adjudication
        if terminal:
            return white_result
        if board.turn == chess.WHITE:
            searcher, evaluate = mcts_white, evaluate_white
        else:
            searcher, evaluate = mcts_black, evaluate_black
        policy = _run_native_search(
            searcher,
            START_FEN,
            [m.uci() for m in board.move_stack],          # full history -> set_root
            evaluate,
            int(num_sims),
            int(max_batch),
        )
        move = _select_move(policy)
        board.push_uci(move)


def play_match(
    net_a, net_b, cfg, num_games, openings=None,
    evaluate_a: Optional[InferenceFn] = None,
    evaluate_b: Optional[InferenceFn] = None,
    max_batch: int = 256,
) -> dict:
    """Native-arena equivalent of arena.play_match. Identical return contract
    {'a': wins_a, 'b': wins_b, 'draws': draws}.

    `evaluate_a`/`evaluate_b` are the per-network InferenceFns (candidate /
    champion respectively). When omitted, GPU runtimes are built on the fly
    from `net_a`/`net_b` (fresh ChessNet + InferenceRuntime, weights copied via
    state_dict — the trainer's nets are never mutated). Tests inject
    deterministic fake fns and pass `net_a=net_b=None`.

    Telemetry (Ticket A): the body runs inside a `PhaseTimer("arena")` and a
    swallow-guarded `phase` record is emitted on exit (side effect only; the
    return contract is unchanged — design §3.4).
    """
    if num_games % 2 != 0:
        raise ValueError(f"arena_games must be even (got {num_games})")

    if evaluate_a is None:
        from native_selfplay import make_gpu_inference_fn
        evaluate_a = make_gpu_inference_fn(cfg)
        evaluate_a.update_weights(net_a.state_dict())
    if evaluate_b is None:
        from native_selfplay import make_gpu_inference_fn
        evaluate_b = make_gpu_inference_fn(cfg)
        evaluate_b.update_weights(net_b.state_dict())

    num_pairs = num_games // 2
    if openings is None:
        openings = generate_arena_openings(
            num_pairs,
            int(getattr(cfg, "arena_opening_plies", 8)),
            int(getattr(cfg, "arena_seed", 424242)),
        )

    mcts_a = _new_mcts(cfg, cfg.arena_simulations, seed=0)   # reused across games
    mcts_b = _new_mcts(cfg, cfg.arena_simulations, seed=0)
    wins_a = wins_b = draws = 0

    with telemetry.PhaseTimer("arena") as _arena_timer:
        for opening_moves in openings:
            # Game A: candidate (net_a / evaluate_a) White, champion Black.
            white_result = _play_native_game(
                mcts_a, mcts_b, evaluate_a, evaluate_b, cfg,
                num_sims=cfg.arena_simulations, opening_moves=opening_moves,
                max_batch=max_batch,
            )
            if white_result > 0.0:
                wins_a += 1
            elif white_result < 0.0:
                wins_b += 1
            else:
                draws += 1

            # Game B: colors swapped.
            white_result = _play_native_game(
                mcts_b, mcts_a, evaluate_b, evaluate_a, cfg,
                num_sims=cfg.arena_simulations, opening_moves=opening_moves,
                max_batch=max_batch,
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


class NativeArenaEngine:
    """Holds two persistent runtimes (candidate + champion) for a whole run.
    Constructed once in train.run_native; weights swapped in per gate."""

    def __init__(self, cfg):
        from native_selfplay import make_gpu_inference_fn
        self.cfg = cfg
        self.candidate_fn = make_gpu_inference_fn(cfg)   # fresh ChessNet + runtime
        self.best_fn = make_gpu_inference_fn(cfg)

    def play_match(self, candidate_sd, best_sd, num_games,
                   openings=None, max_batch=256) -> dict:
        self.candidate_fn.update_weights(candidate_sd)   # cheap state_dict load
        self.best_fn.update_weights(best_sd)
        return play_match(
            None, None, self.cfg, num_games, openings,
            evaluate_a=self.candidate_fn, evaluate_b=self.best_fn,
            max_batch=max_batch,
        )
