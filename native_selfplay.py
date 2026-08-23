"""Native self-play: drive the C++ multi-game Actor with the GPU runtime.

This is the Task 6/7 integration layer.  It replaces the Python worker-process
self-play path (`parallel.py`) with one native `Actor` that manages many games
in a single process and feeds the GPU through the persistent `InferenceRuntime`.

Data flow (one self-play round):

    teacher net (GPU)  ->  InferenceRuntime (sparse legal-logit gather)
    native Actor  <->  gather_leaves / apply_evaluations
    finished games  ->  replay examples [(state, pi, z), ...] for the trainer

The teacher weights are loaded into the runtime's resident model once per round;
the actor tags every game with an immutable ``(weight_version, generation)``
handle so a stale game can never be mislabelled as new-generation data.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

import chess_rl_native as native

# Inference callback: (inputs, legal_offsets, legal_indices) -> (legal_logits, values).
InferenceFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]

# SplitMix64 constants (mirror native/actor.cpp::derive_seed) so a round seed
# derived in Python matches the C++ per-game seed derivation exactly.
_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB


def derive_selfplay_seed(base_seed: int, iteration: int) -> int:
    """Stable unsigned 64-bit self-play round seed from ``(base_seed, iteration)``.

    A SplitMix64-style finaliser, mirroring the native Actor's own per-game
    seed derivation.  The result depends ONLY on the two integer inputs — never
    Python ``hash()``, the process id, wall-clock time, or ``optimizer_steps``
    (which resets after arena rejection) — so an uninterrupted run and a
    resumed run that both reach the same completed-iteration boundary derive
    the exact same round seed, while adjacent iterations always differ.
    """
    base = int(base_seed) & _MASK64
    it = int(iteration) & _MASK64
    x = (base + it + _GOLDEN) & _MASK64
    x = ((x ^ (x >> 30)) * _MIX1) & _MASK64
    x = ((x ^ (x >> 27)) * _MIX2) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


class NativeSelfPlay:
    """Runs self-play games with the native Actor + an injected inference fn.

    The inference fn is injected (not imported) so the trainer can pass the
    GPU `InferenceRuntime.evaluate` in production and a deterministic fake in
    tests — the actor never depends on a specific backend.
    """

    def __init__(
        self,
        cfg,
        inference_fn: InferenceFn,
        games: Optional[int] = None,
        weight_version: int = 0,
        generation: int = 0,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.inference_fn = inference_fn
        self.games = int(games if games is not None else cfg.games_per_iteration)
        self.weight_version = int(weight_version)
        self.generation = int(generation)
        # One explicit base seed per self-play ROUND.  The trainer derives this
        # from (cfg.seed, iteration) via derive_selfplay_seed so each iteration
        # plays genuinely new games; the C++ Actor then derives distinct
        # per-game seeds from it.  cfg.seed is NEVER mutated here.
        actor_seed = int(seed) if seed is not None else int(cfg.seed)
        self.round_seed = actor_seed
        self.actor = native.Actor(
            games=self.games,
            c_puct=float(cfg.c_puct),
            virtual_loss=float(cfg.virtual_loss),
            num_simulations=int(cfg.num_simulations),
            temperature=float(cfg.temperature),
            temperature_threshold=int(cfg.temperature_threshold),
            max_game_length=int(cfg.max_game_length),
            seed=actor_seed,
            num_threads=self.games,
        )
        self.actor.set_teacher(self.weight_version, self.generation)

    def run(self, max_batch: int = 256) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Drive gather/apply/advance to completion; return replay examples."""
        examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        while not self.actor.is_done():
            tokens, inputs, offsets, indices = self.actor.gather_leaves(max_batch)
            if len(tokens) == 0:
                self.actor.advance()
                continue
            logits, values = self.inference_fn(inputs, offsets, indices)
            self.actor.apply_evaluations(tokens, offsets, logits, values)
            self.actor.advance()
        for game in self.actor.finished_games():
            for state, pi, z in game["examples"]:
                examples.append((state, pi, z))
        return examples


def make_gpu_inference_fn(cfg, model=None):
    """Return an ``InferenceRuntime.evaluate``-compatible callback.

    Lazily builds a persistent runtime holding ``model`` (or a fresh ChessNet)
    on the GPU.  The returned closure swaps fresh teacher weights in via
    ``update_weights`` and then evaluates the batch.
    """
    from gpu_runtime import InferenceRuntime

    runtime = InferenceRuntime(cfg=cfg, model=model)

    def evaluate(inputs, legal_offsets, legal_indices):
        return runtime.evaluate(inputs, legal_offsets, legal_indices)

    def update_weights(state_dict):
        runtime.model.load_state_dict(state_dict)

    evaluate.update_weights = update_weights  # type: ignore[attr-defined]
    evaluate.runtime = runtime  # type: ignore[attr-defined]
    return evaluate
