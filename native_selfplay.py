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

import time
from typing import Callable, List, Optional, Tuple

import numpy as np

import chess_rl_native as native
import telemetry

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
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ):
        self.cfg = cfg
        self.inference_fn = inference_fn
        self.games = int(games if games is not None else cfg.games_per_iteration)
        self.weight_version = int(weight_version)
        self.generation = int(generation)
        self.run_id = run_id
        self.iteration = iteration
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

        # ---- round telemetry (Ticket A; semantic-free counters) ----
        # Accumulated inside run(); exposed so the trainer can merge them into
        # the `selfplay` phase record.  All values derive from data this class
        # already holds — the inference fn stays a pure injected callback.
        self.gather_calls = 0
        self.apply_calls = 0
        self.advance_calls = 0
        self.gather_s = 0.0
        self.apply_s = 0.0
        self.advance_s = 0.0
        self.batch_b: List[int] = []          # B per non-empty gather
        self.inference_calls = 0
        self.simulations = 0
        self.round_duration_s = 0.0
        self.trajectory_hashes: List[str] = []  # one BLAKE2 digest per game
        self.batch_stats = telemetry._percentiles(self.batch_b)

    def run(self, max_batch: int = 256) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Drive gather/apply/advance to completion; return replay examples."""
        examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        t_round = time.perf_counter()
        while not self.actor.is_done():
            t0 = time.perf_counter()
            tokens, inputs, offsets, indices = self.actor.gather_leaves(max_batch)
            self.gather_s += time.perf_counter() - t0
            self.gather_calls += 1
            if len(tokens) == 0:
                t_adv = time.perf_counter()
                self.actor.advance()
                self.advance_s += time.perf_counter() - t_adv
                self.advance_calls += 1
                continue
            self.batch_b.append(int(len(tokens)))
            self.inference_calls += 1
            logits, values = self.inference_fn(inputs, offsets, indices)
            t_apply = time.perf_counter()
            self.actor.apply_evaluations(tokens, offsets, logits, values)
            self.apply_s += time.perf_counter() - t_apply
            self.apply_calls += 1
            t_adv = time.perf_counter()
            self.actor.advance()
            self.advance_s += time.perf_counter() - t_adv
            self.advance_calls += 1
            self.simulations += int(self.cfg.num_simulations)
        self.round_duration_s = time.perf_counter() - t_round
        self.batch_stats = telemetry._percentiles(self.batch_b)
        for game in self.actor.finished_games():
            for state, pi, z in game["examples"]:
                examples.append((state, pi, z))
            # Generation-time trajectory identity: one BLAKE2 digest over the
            # game's ORDERED (state, pi, z) examples (design §2.3).
            self.trajectory_hashes.append(
                telemetry.game_trajectory_hash(game["examples"])
            )
        self._emit_round_telemetry(len(examples))
        return examples

    def _emit_round_telemetry(self, n_examples: int) -> None:
        """Emit the ``gather_apply_advance`` phase + ``selfplay_round``
        diversity records (design §3.2).  Swallow-guarded: never affects
        training or game semantics."""
        try:
            telemetry.safe_emit(self.cfg, {
                "type": "phase",
                "phase": "gather_apply_advance",
                "run_id": self.run_id,
                "iteration": self.iteration,
                "generation": self.generation,
                "duration_s": self.round_duration_s,
                "gather_calls": self.gather_calls,
                "apply_calls": self.apply_calls,
                "advance_calls": self.advance_calls,
                "gather_s": self.gather_s,
                "apply_s": self.apply_s,
                "advance_s": self.advance_s,
                "inference_calls": self.inference_calls,
                "simulations": self.simulations,
                **self.batch_stats,
            })
        except Exception:  # noqa: BLE001 - telemetry must never kill training
            pass
        try:
            counts = {}
            for h in self.trajectory_hashes:
                counts[h] = counts.get(h, 0) + 1
            telemetry.safe_emit(self.cfg, {
                "type": "diversity",
                "source": "selfplay_round",
                "run_id": self.run_id,
                "iteration": self.iteration,
                "generation": self.generation,
                "replay_size": int(n_examples),
                "unique_trajectory_hashes": len(counts),
                "most_repeated_trajectory_count": (
                    max(counts.values()) if counts else 0
                ),
                "trajectory_hashes": list(counts.keys())[:32],
            })
        except Exception:  # noqa: BLE001 - telemetry must never kill training
            pass


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
