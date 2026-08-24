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

import os
import queue
import threading
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


# Compact-plane contract (see native/position.cpp::encode_planes_u8): the
# native actor emits uint8 planes.  Every plane is binary except the
# halfmove-clock plane, which carries the RAW clock and must be divided by
# HALFMOVE_SCALE to recover the float encoding.
#
# The GPU runtime does this expansion on-device (InferenceRuntime._expand_u8_into);
# this host-side version exists for CPU consumers — test doubles, the
# CPU arena path, and anything else standing in for the network — so a
# stand-in evaluator sees exactly the features the real network sees.
HALFMOVE_PLANE = 12 * 8 + native.HALFMOVE_META_PLANE   # 102 for history_steps=8
HALFMOVE_SCALE = float(native.HALFMOVE_SCALE)


def expand_planes(planes: np.ndarray, halfmove_plane: int = HALFMOVE_PLANE
                  ) -> np.ndarray:
    """Expand compact uint8 planes to the float32 encoding, exactly.

    float32 input is returned unchanged (already expanded), so callers can
    accept either wire format without branching.  The divide is done in
    float32, which reproduces the C++ ``float(clock / 100.0)`` bit-for-bit
    for every storable clock value (tests/test_compact_planes.py).
    """
    arr = np.asarray(planes)
    if arr.dtype != np.uint8:
        return arr
    out = arr.astype(np.float32)
    out[..., halfmove_plane, :, :] /= np.float32(HALFMOVE_SCALE)
    return out


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


def _actor_termination_kwargs(cfg):
    """Resignation / adjudication arguments for the native Actor.

    Disabled features are passed as sentinels (-1.0 thresholds) rather than
    omitted, so the C++ side has a single code path and the validation there
    (notably: resignation without a playout fraction is refused) always runs.
    """
    resign_on = bool(getattr(cfg, "resign_enabled", False))
    draw_on = bool(getattr(cfg, "draw_adjudication_enabled", False))
    return {
        "resign_threshold": (float(getattr(cfg, "resign_threshold", -0.90))
                             if resign_on else -1.0001),
        "resign_consecutive": int(getattr(cfg, "resign_consecutive_plies", 2)),
        "resign_playout_fraction": float(
            getattr(cfg, "resign_playout_fraction", 0.10)),
        "draw_threshold": (float(getattr(cfg, "draw_threshold", 0.02))
                           if draw_on else -1.0),
        "draw_consecutive": int(getattr(cfg, "draw_consecutive_plies", 8)),
        "draw_min_ply": int(getattr(cfg, "draw_min_ply", 60)),
    }


def summarise_terminations(games):
    """Aggregate resignation calibration over a round's finished games.

    `false_resignation_rate` is measured ONLY over playout games that hit the
    condition — the games where resignation was suppressed so a real result
    existed to check against.  A resigned game has no ground truth, which is
    precisely why a fraction must be played out.
    """
    total = len(games)
    if total == 0:
        return {}
    playout = [g for g in games if g.get("playout")]
    would_resign = [g for g in playout if g.get("would_have_resigned")]
    would_draw = [g for g in playout if g.get("would_have_drawn")]
    false_resign = [g for g in would_resign if g.get("false_resignation")]
    false_draw = [g for g in would_draw if g.get("false_draw")]
    plies = [int(g.get("plies", 0)) for g in games]
    terms = {}
    for g in games:
        t = g.get("termination", "unknown")
        terms[t] = terms.get(t, 0) + 1
    return {
        "games": total,
        "resigned": sum(1 for g in games if g.get("resigned")),
        "adjudicated_draws": sum(1 for g in games if g.get("adjudicated_draw")),
        "playout_games": len(playout),
        "playout_fraction": len(playout) / total,
        "playout_would_resign": len(would_resign),
        "false_resignations": len(false_resign),
        # None (not 0.0) when no playout game hit the condition: "no evidence"
        # and "no false positives" are different states and must not be
        # confused by whoever reads the telemetry.
        "false_resignation_rate": (len(false_resign) / len(would_resign)
                                   if would_resign else None),
        "playout_would_draw": len(would_draw),
        "false_draws": len(false_draw),
        "false_draw_rate": (len(false_draw) / len(would_draw)
                            if would_draw else None),
        "mean_plies": sum(plies) / total if plies else 0.0,
        "terminations": terms,
    }


def _with_moves_left(game_examples, enabled):
    """Optionally attach the plies-remaining auxiliary target to one game.

    The label needs no engine support: a finished game's examples are already
    in ply order, so the position at index i has ``len - 1 - i`` plies left.
    Deriving it here rather than in C++ keeps the native Example struct and
    its serialisation untouched.

    ``enabled`` is False unless the moves-left head is configured, and then
    this yields the original ``(state, pi, z)`` triples untouched.  Emitting
    4-tuples unconditionally would change the published example contract for
    every consumer — replay, tests, the diversity audit — to no purpose when
    the head is off.
    """
    if not enabled:
        yield from game_examples
        return
    total = len(game_examples)
    for i, (state, pi, z) in enumerate(game_examples):
        yield state, pi, z, float(total - 1 - i)


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
        # Games in flight = concurrency.  Explicit argument wins; otherwise
        # cfg.selfplay_games_in_flight, falling back to the training-cadence
        # knob cfg.games_per_iteration so existing configs are unchanged.
        if games is not None:
            self.games = int(games)
        else:
            in_flight = getattr(cfg, "selfplay_games_in_flight", None)
            self.games = int(in_flight if in_flight else cfg.games_per_iteration)
        # Actor worker threads: explicit config, else let the actor clamp to
        # min(hardware_concurrency, games) itself.
        actor_threads = getattr(cfg, "selfplay_actor_threads", None)
        actor_threads = int(actor_threads) if actor_threads else self.games
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
        # Auxiliary moves-left labels are emitted only when the head that
        # consumes them is configured; otherwise the example contract stays
        # the original (state, pi, z) triple.
        self.emit_moves_left = bool(getattr(cfg, "moves_left_head", False))
        self.termination_stats: dict = {}
        self.actor = native.Actor(
            games=self.games,
            c_puct=float(cfg.c_puct),
            virtual_loss=float(cfg.virtual_loss),
            num_simulations=int(cfg.num_simulations),
            temperature=float(cfg.temperature),
            temperature_threshold=int(cfg.temperature_threshold),
            max_game_length=int(cfg.max_game_length),
            seed=actor_seed,
            num_threads=int(actor_threads),
            **_actor_termination_kwargs(cfg),
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

    def run(self, max_batch: Optional[int] = None,
            leaves_per_game: Optional[int] = None
            ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Drive gather/apply/advance to completion; return replay examples.

        ``max_batch``/``leaves_per_game`` default to ``cfg.selfplay_max_batch``
        and ``cfg.selfplay_leaves_per_game``.  With a fixed per-game leaf
        target the merged batch grows with the number of games in flight
        instead of being sliced out of a fixed budget.
        """
        if max_batch is None:
            max_batch = int(getattr(self.cfg, "selfplay_max_batch", 256))
        if leaves_per_game is None:
            leaves_per_game = int(
                getattr(self.cfg, "selfplay_leaves_per_game", 0) or 0
            )
        examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        t_round = time.perf_counter()
        while not self.actor.is_done():
            t0 = time.perf_counter()
            tokens, inputs, offsets, indices = self.actor.gather_leaves(
                max_batch, leaves_per_game)
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
        finished = self.actor.finished_games()
        self.termination_stats = summarise_terminations(finished)
        for game in finished:
            examples.extend(
                _with_moves_left(game["examples"], self.emit_moves_left))
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


class _InferenceService:
    """Serialises every GPU call onto ONE dedicated thread.

    A plain lock around the runtime is not enough.  ``torch.compile(mode=
    "reduce-overhead")`` runs the forward pass through CUDA graph trees, and
    the graph manager lives in THREAD-LOCAL storage — calling the compiled
    function from a second thread trips
    ``assert torch._C._is_key_in_tls("tree_manager_containers")`` deep inside
    inductor's cudagraph_trees.  Pinning all inference to a single owner
    thread keeps the compiled path usable while still overlapping: the shard
    threads descend their trees while this thread drives the GPU.

    Requests are served strictly FIFO, one at a time, so each shard sees
    exactly the results it would have seen in the serial loop.
    """

    _STOP = object()

    def __init__(self, inference_fn: InferenceFn):
        self.inference_fn = inference_fn
        self.queue: "queue.Queue" = queue.Queue()
        self.busy_s = 0.0
        self.calls = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self.queue.get()
            if item is self._STOP:
                return
            inputs, offsets, indices, box = item
            t0 = time.perf_counter()
            try:
                box["result"] = self.inference_fn(inputs, offsets, indices)
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                box["error"] = exc
            finally:
                self.busy_s += time.perf_counter() - t0
                self.calls += 1
                box["event"].set()

    def submit(self, inputs, offsets, indices):
        """Blocking call: enqueue a batch and wait for its result."""
        box = {"event": threading.Event(), "result": None, "error": None}
        self.queue.put((inputs, offsets, indices, box))
        box["event"].wait()
        if box["error"] is not None:
            raise box["error"]
        return box["result"]

    def close(self) -> None:
        self.queue.put(self._STOP)
        self._thread.join(timeout=30)

    def mark(self) -> float:
        """Current cumulative busy seconds, for per-round deltas."""
        return self.busy_s


class ShardedSelfPlay:
    """Self-play with CPU tree descent and GPU inference OVERLAPPED.

    The single-actor loop is strictly serial: gather -> evaluate -> apply ->
    advance.  While the C++ threads descend the trees the GPU has nothing
    queued, and while the GPU runs the forward pass every C++ thread is parked
    in join().  Measured on this machine the descent was ~78-93% of the round
    and inference ~15-17%, with neither overlapping the other.

    Here the games are split into ``shards`` independent actors, each driven by
    its own Python thread.  A lock serialises GPU access (InferenceRuntime is
    explicitly not thread-safe), so while one shard holds the GPU the others
    are descending.  The native bindings release the GIL around gather_leaves,
    apply_evaluations and advance, without which those Python threads would
    simply queue on the interpreter lock and nothing would overlap.

    Determinism
    -----------
    Each shard's actor is seeded independently and its search depends only on
    its own leaf evaluations, so a shard's games do not depend on how its GPU
    calls interleave with other shards'.  Finished games are concatenated in
    SHARD ORDER, so the returned example list is deterministic even though the
    threads finish in arbitrary order.  ``shards == 1`` uses the round seed
    directly and is byte-identical to NativeSelfPlay.
    """

    def __init__(
        self,
        cfg,
        inference_fn: InferenceFn,
        games: Optional[int] = None,
        shards: Optional[int] = None,
        weight_version: int = 0,
        generation: int = 0,
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ):
        self.cfg = cfg
        self.inference_fn = inference_fn
        self.weight_version = int(weight_version)
        self.generation = int(generation)
        self.run_id = run_id
        self.iteration = iteration

        if games is not None:
            total = int(games)
        else:
            in_flight = getattr(cfg, "selfplay_games_in_flight", None)
            total = int(in_flight if in_flight else cfg.games_per_iteration)
        if total <= 0:
            raise ValueError("games must be positive")
        self.games = total

        n_shards = shards if shards is not None else getattr(
            cfg, "selfplay_shards", 2)
        self.shards = max(1, min(int(n_shards or 1), total))

        self.round_seed = int(seed) if seed is not None else int(cfg.seed)
        self.emit_moves_left = bool(getattr(cfg, "moves_left_head", False))
        self.termination_stats: dict = {}
        self.max_batch = int(getattr(cfg, "selfplay_max_batch", 256))
        self.leaves_per_game = int(
            getattr(cfg, "selfplay_leaves_per_game", 0) or 0)

        # Split games as evenly as possible; earlier shards take the remainder.
        base, rem = divmod(total, self.shards)
        self._shard_games = [base + (1 if i < rem else 0)
                             for i in range(self.shards)]

        # Actor worker threads are a machine-wide resource: divide the budget
        # across shards rather than giving every shard the whole CPU.
        cfg_threads = getattr(cfg, "selfplay_actor_threads", None)
        hw = os.cpu_count() or 1
        budget = int(cfg_threads) if cfg_threads else hw
        per_shard_threads = max(1, budget // self.shards)

        self.actors = []
        for i, g in enumerate(self._shard_games):
            # shards == 1 must reproduce NativeSelfPlay exactly, so the single
            # actor gets the round seed unchanged rather than a derived one.
            shard_seed = (self.round_seed if self.shards == 1
                          else derive_selfplay_seed(self.round_seed, i + 1))
            actor = native.Actor(
                games=g,
                c_puct=float(cfg.c_puct),
                virtual_loss=float(cfg.virtual_loss),
                num_simulations=int(cfg.num_simulations),
                temperature=float(cfg.temperature),
                temperature_threshold=int(cfg.temperature_threshold),
                max_game_length=int(cfg.max_game_length),
                seed=shard_seed,
                num_threads=min(per_shard_threads, g),
                **_actor_termination_kwargs(cfg),
            )
            actor.set_teacher(self.weight_version, self.generation)
            self.actors.append(actor)

        # Backwards-compatible handle for callers that read sp.actor.
        self.actor = self.actors[0]

        # ---- merged round telemetry ----
        self.gather_s = self.apply_s = self.advance_s = 0.0
        self.infer_s = 0.0
        self.gather_calls = self.apply_calls = 0
        self.advance_calls = self.inference_calls = 0
        self.simulations = 0
        self.batch_b: List[int] = []
        self.batch_stats: dict = {}
        self.trajectory_hashes: List[str] = []
        self.round_duration_s = 0.0
        self.gpu_busy_s = 0.0

    # ------------------------------------------------------------------ #

    def _drive(self, index: int, stats: dict) -> None:
        """One shard's driver loop; runs on its own Python thread."""
        actor = self.actors[index]
        while not actor.is_done():
            t0 = time.perf_counter()
            tokens, inputs, offsets, indices = actor.gather_leaves(
                self.max_batch, self.leaves_per_game)
            t1 = time.perf_counter()
            stats["gather_s"] += t1 - t0
            stats["gather_calls"] += 1
            if len(tokens) == 0:
                actor.advance()
                stats["advance_calls"] += 1
                stats["advance_s"] += time.perf_counter() - t1
                continue
            stats["batch_b"].append(int(len(tokens)))

            # All GPU work goes through the single inference thread.  The
            # wall time spent here includes queueing behind another shard —
            # which is precisely the overlap signal: waiting means the GPU
            # was already busy with someone else's batch.
            acquired = time.perf_counter()
            logits, values = self._service.submit(inputs, offsets, indices)
            done = time.perf_counter()
            stats["infer_s"] += done - acquired
            stats["inference_calls"] += 1

            t2 = time.perf_counter()
            actor.apply_evaluations(tokens, offsets, logits, values)
            t3 = time.perf_counter()
            stats["apply_s"] += t3 - t2
            stats["apply_calls"] += 1
            actor.advance()
            stats["advance_s"] += time.perf_counter() - t3
            stats["advance_calls"] += 1
            stats["simulations"] += int(self.cfg.num_simulations)

    def run(self, max_batch: Optional[int] = None,
            leaves_per_game: Optional[int] = None
            ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Drive every shard concurrently; return replay examples in shard order."""
        if max_batch is not None:
            self.max_batch = int(max_batch)
        if leaves_per_game is not None:
            self.leaves_per_game = int(leaves_per_game)

        # Reuse the runtime's long-lived serving thread when there is one.
        # A fresh thread per round would break torch.compile's CUDA graph
        # trees all over again: their manager is thread-local, so the first
        # compiled call on a NEW thread trips the inductor TLS assertion.
        shared = getattr(self.inference_fn, "_service", None)
        if shared is not None:
            self._service = shared
            self._owns_service = False
        else:
            self._service = _InferenceService(self.inference_fn)
            self._owns_service = True
        busy_before = self._service.mark()
        per_shard = [
            {"gather_s": 0.0, "apply_s": 0.0, "advance_s": 0.0,
             "infer_s": 0.0, "gather_calls": 0,
             "apply_calls": 0, "advance_calls": 0, "inference_calls": 0,
             "simulations": 0, "batch_b": []}
            for _ in range(self.shards)
        ]
        errors: List[BaseException] = []
        error_lock = threading.Lock()

        def target(i):
            try:
                self._drive(i, per_shard[i])
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                with error_lock:
                    errors.append(exc)

        t_round = time.perf_counter()
        try:
            threads = [threading.Thread(target=target, args=(i,), daemon=True)
                       for i in range(self.shards)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            self.gpu_busy_s = self._service.mark() - busy_before
            if self._owns_service:
                self._service.close()
        self.round_duration_s = time.perf_counter() - t_round
        if errors:
            raise errors[0]

        for st in per_shard:
            self.gather_s += st["gather_s"]
            self.apply_s += st["apply_s"]
            self.advance_s += st["advance_s"]
            self.infer_s += st["infer_s"]
            self.gather_calls += st["gather_calls"]
            self.apply_calls += st["apply_calls"]
            self.advance_calls += st["advance_calls"]
            self.inference_calls += st["inference_calls"]
            self.simulations += st["simulations"]
            self.batch_b.extend(st["batch_b"])
        self.batch_stats = telemetry._percentiles(self.batch_b)

        # Shard order, then the actor's own game order: deterministic despite
        # the threads finishing in whatever order they finish.
        examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        all_finished = []
        for actor in self.actors:
            for game in actor.finished_games():
                all_finished.append(game)
                examples.extend(
                    _with_moves_left(game["examples"], self.emit_moves_left))
                self.trajectory_hashes.append(
                    telemetry.game_trajectory_hash(game["examples"])
                )
        self.termination_stats = summarise_terminations(all_finished)
        self._emit_round_telemetry(len(examples))
        return examples

    # ------------------------------------------------------------------ #

    @property
    def gpu_busy_fraction(self) -> float:
        """Share of the round during which the GPU held a batch.

        This is the number the overlap work exists to raise: in the serial
        loop it is bounded by the inference share of one thread's timeline.
        """
        if self.round_duration_s <= 0:
            return 0.0
        return self.gpu_busy_s / self.round_duration_s

    def _emit_round_telemetry(self, examples: int) -> None:
        # Keep emitting the ORIGINAL gather_apply_advance record with its
        # original field names.  The sharded driver is now the only self-play
        # driver the trainer uses, so dropping this record would silently
        # remove a phase that existing consumers (and the telemetry schema
        # test) expect.  The per-shard timings are simply summed.
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
            telemetry.safe_emit(self.cfg, {
                "type": "phase",
                "phase": "selfplay_sharded",
                "run_id": self.run_id,
                "iteration": self.iteration,
                "generation": self.generation,
                "duration_s": self.round_duration_s,
                "shards": self.shards,
                "shard_games": list(self._shard_games),
                "games": self.games,
                "examples": examples,
                "gather_s": self.gather_s,
                "apply_s": self.apply_s,
                "advance_s": self.advance_s,
                "infer_s": self.infer_s,
                "gpu_busy_s": self.gpu_busy_s,
                "gpu_busy_fraction": self.gpu_busy_fraction,
                "inference_calls": self.inference_calls,
                "gather_calls": self.gather_calls,
                "simulations": self.simulations,
                "leaves_per_game": self.leaves_per_game,
                "max_batch": self.max_batch,
                **{f"term_{k}": v
                   for k, v in self.termination_stats.items()
                   if k != "terminations"},
                **self.batch_stats,
            })
        except Exception:  # noqa: BLE001 - telemetry must never kill training
            pass
        # Replay-diversity audit.  This is the record that caught the
        # fixed-seed self-play collapse; the sharded driver must keep emitting
        # it or the regression it guards against becomes invisible again.
        try:
            counts: dict = {}
            for h in self.trajectory_hashes:
                counts[h] = counts.get(h, 0) + 1
            telemetry.safe_emit(self.cfg, {
                "type": "diversity",
                "source": "selfplay_round",
                "run_id": self.run_id,
                "iteration": self.iteration,
                "generation": self.generation,
                "replay_size": int(examples),
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

    def _raw_evaluate(inputs, legal_offsets, legal_indices):
        return runtime.evaluate(inputs, legal_offsets, legal_indices)

    # One serving thread for the runtime's whole life.  Every GPU call — from
    # any number of self-play shards — is executed on it, which both honours
    # "InferenceRuntime is not thread-safe" and keeps torch.compile's
    # thread-local CUDA graph state valid.
    service = _InferenceService(_raw_evaluate)

    def evaluate(inputs, legal_offsets, legal_indices):
        return service.submit(inputs, legal_offsets, legal_indices)

    def update_weights(state_dict):
        runtime.model.load_state_dict(state_dict)

    def close():
        service.close()

    evaluate.update_weights = update_weights  # type: ignore[attr-defined]
    evaluate.runtime = runtime                # type: ignore[attr-defined]
    evaluate._service = service               # type: ignore[attr-defined]
    evaluate.close = close                    # type: ignore[attr-defined]
    return evaluate
