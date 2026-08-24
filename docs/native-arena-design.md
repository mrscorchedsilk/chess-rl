# Native Arena Design — Replace Python-MCTS Arena Search with `chess_rl_native.MCTS`

## 0. Objective

Replace only the **search engine** inside the arena gate with the existing
native MCTS core (`chess_rl_native.MCTS`), while keeping every *game-semantic*
of the arena identical: the deterministic paired-opening suite, color swaps,
temperature 0, no root noise, 20 games, 40 simulations, the standard score
`(wins + 0.5*draws)/games`, the `0.55` accept threshold, and the
candidate-vs-champion identity. The native search is already proven by
`benchmarks/native_mcts.py` (`run_native_search` drives the two-phase API), so
this is an **adapter**, not a new search.

**Hard constraint — do not change semantics.** Native and Python MCTS may
legitimately produce *different visit distributions* (different child ordering
and tie-break order; see §8). The gate's *decision procedure* (score vs
threshold, which network is candidate vs champion) is unchanged; *individual
move choices* may differ. Because the gate is a deterministic function of the
network weights + fixed opening suite, a fixed-seed parity test is mandatory
(§9).

---

## 1. Current state (what already exists)

- `arena.py` — the Python-MCTS arena. `play_match(net_a, net_b, cfg, num_games,
  openings=None)` builds `MCTS(net_a, cfg)` / `MCTS(net_b, cfg)` and loops
  `_play_arena_game(...)` twice per opening (color swap), returning
  `{"a": wins_a, "b": wins_b, "draws": draws}`. Lines 128–176.
- `arena._terminal_result(board)` (lines 25–33) — python-chess
  `board.outcome(claim_draw=True)` adjudication; **must be preserved**.
- `arena.generate_arena_openings(num_pairs, opening_plies, seed)` (59–80) and
  `arena.arena_suite_hash(openings)` (83–90) — the deterministic suite; reuse
  verbatim.
- `chess_rl_native.MCTS` (native/mcts.h, native/mcts.cpp, bound in
  native/chess_rl_native.cpp:265–278) — pinned API:
  `set_root(start_fen, history_moves)`, `gather_leaves(max_batch)`,
  `apply_evaluations(tokens, offsets, logits, values)`, `is_complete()`,
  `policy(temperature)`. `set_root` fully resets search state, so **one MCTS
  object per side is reusable across all plies and all games**.
- `gpu_runtime.InferenceRuntime` — the persistent GPU evaluator:
  `evaluate(inputs, offsets, indices) -> (legal_logits[K] f32, values[B,1] f32)`.
- `native_selfplay.make_gpu_inference_fn(cfg, model=None)` — returns a closure
  `evaluate` with `.update_weights(state_dict)` and `.runtime`; builds a *fresh*
  `ChessNet` + `InferenceRuntime` (never the trainer's nets).
- `train._arena_gate(cfg, net, best_net)` (train.py:850) — generates the suite,
  hashes it, calls `play_match(net, best_net, cfg, num_games=cfg.arena_games)`,
  computes score/threshold.

### Known defect to work around (do not copy blindly)

`benchmarks/native_mcts.py::run_native_search` (line 179) reads
`mcts.num_simulations`, which is **not exposed** by the pybind `MCTS` binding
(verified: `AttributeError`). The adapter must therefore track the simulation
budget itself from `cfg.arena_simulations` (see `_run_native_search`, §4) and
must NOT reuse that helper verbatim. The driver loop below is the corrected
form.

---

## 2. New module

**Name:** `native_arena.py`
**Location:** repo root, sibling to `native_selfplay.py` (so `train.py` can
`import native_arena` exactly as it already does `import native_selfplay`).

It depends only on: `chess`, `numpy`, `chess_rl_native`, `arena`
(`generate_arena_openings`, `_terminal_result`, `arena_suite_hash`), and
`native_selfplay.make_gpu_inference_fn`. No circular import: `native_arena` is
imported by `train.py` and (optionally) lazily by `arena.py`, never the reverse.

---

## 3. Data flow

```
candidate net (trainer)  ──state_dict──▶ InferenceRuntime A (persistent)
champion net (trainer)   ──state_dict──▶ InferenceRuntime B (persistent)

for each opening (arena_seed=424242, 8 plies):
  Game A: native.MCTS(white=candidate, eval=A)  vs  native.MCTS(black=champion, eval=B)
  Game B: native.MCTS(white=champion,  eval=B)  vs  native.MCTS(black=candidate, eval=A)

  each ply:
    1. python-chess length cap / _terminal_result adjudication (unchanged)
    2. mcts.set_root(START_FEN, [uci for uci in board.move_stack])   # full history
    3. two-phase gather/apply loop against the side's InferenceRuntime
    4. policy(0.0) -> one-hot most-visited move -> board.push_uci(move)
```

Two MCTS objects (`mcts_a`, `mcts_b`) are created **once per `play_match`** and
reused across all 20 games and all plies (the Python path likewise reuses one
`MCTS` per side; `set_root` performs the reset). Two `InferenceRuntime`s are
created **once per run** and only have weights refreshed per gate (§5).

---

## 4. Exact signatures (module contract)

```python
# native_arena.py
from __future__ import annotations
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
import chess
import chess_rl_native as native
from arena import generate_arena_openings, _terminal_result, arena_suite_hash

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
    `mcts.num_simulations`; see §1 defect)."""
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
    deterministic fake fns and pass `net_a=net_b=None`."""
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

    return {"a": wins_a, "b": wins_b, "draws": draws}
```

---

## 5. Persistent InferenceRuntime per network

**Each network gets exactly one persistent `InferenceRuntime` for the life of a
run** (built once, weights refreshed per gate). This avoids recompiling
`torch.compile` every arena gate (arena fires every 20 iterations).

```python
# native_arena.py (continued)
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
```

Key properties:

- `make_gpu_inference_fn(cfg)` builds a **fresh** `ChessNet` + `InferenceRuntime`
  (eval mode, channels_last, no-grad), so the trainer's `net` / `best_net`
  (which continue to be trained/checkpointed) are **never** moved to
  channels_last, converted to eval-only, or have their `requires_grad` cleared.
- `update_weights(state_dict)` copies the candidate/champion weights into the
  runtime's resident model. The architecture is guaranteed to match because
  both the trainer nets and the runtime nets are built from the same `cfg`.
- VRAM cost: three runtimes total (one self-play + two arena) at ~2.17M params
  each is negligible against 11 GB; the arena runtimes exist only when the
  native arena backend is enabled.

---

## 6. Feeding `set_root(start_fen, history_moves)` from the paired-opening suite

The opening suite is already materialized as a list of UCI move sequences by
`arena.generate_arena_openings` (stable seed `arena_seed=424242`, `opening_plies=8`).

- `_play_native_game` first pushes every opening UCI onto a python-chess
  `board` (so the GAME-level adjudication sees the true position).
- At every ply, the search is rooted at the **standard start position plus the
  complete move stack**: `set_root(START_FEN, [m.uci() for m in board.move_stack])`.

This preserves the 8-position history stack, castling rights, halfmove clock,
en-passant and repetition state exactly — the same guarantee the Python path
gets by replaying the sequence onto a `chess.Board` before searching (see
`arena.generate_arena_openings` docstring, lines 60–67). The native
`Position::from_uci_history` (native/position.h:22) reconstructs all of this
from `(start_fen, history_moves)`, and `set_root` does not consume the caller's
board.

---

## 7. Temperature-0 move selection

`mcts.policy(0.0)` (native/mcts.cpp `policy`, temperature==0 branch) returns a
UCI-sorted list `[(uci, prob)]` that is **one-hot on the most-visited move**
(ties broken to the first child in ascending-action-index order, matching
`np.argmax`). `_select_move` takes `max(policy, key=lambda p: p[1])[0]`, which
returns that move. No sampling, no randomness, no root noise — identical in
intent to `arena.py` line 124 (`move = max(pi, key=pi.get)`).

---

## 8. python-chess terminal adjudication is preserved

Two levels of adjudication exist and BOTH are preserved:

1. **Game level** (`_play_native_game`): before each search the loop checks
   `len(board.move_stack) >= cfg.max_game_length` (draw) and
   `_terminal_result(board)` = `board.outcome(claim_draw=True)`. This is the
   exact python-chess terminal logic from `arena.py` and is unchanged.
2. **Search level**: the native core already mirrors `mcts.py` internally — it
   marks claimable draws (threefold repetition, fifty-move claim) as terminal
   leaves during descent via `pos_->outcome(claim_draw=true)`
   (native/mcts.cpp `gather_leaves`) and backs them up without a network call.
   This is *equivalent to*, not a replacement for, `mcts.py`'s behaviour, and
   the adapter does not touch it.

The arena never searches a terminal root: the game loop returns before calling
`_run_native_search`, exactly as the Python path returns before `search()`.

---

## 9. Divergence note + mandatory fixed-seed parity test

Native and Python MCTS may produce **different visit distributions** because:

- child iteration order differs (python: `board.legal_moves` order; native:
  ascending action index, CSR order), so exact PUCT ties break differently;
- `select_child` tie-break is "first child in CSR order" (strict `>`), while
  `mcts.py._select` keeps insertion order.

Consequences:

- The **gate semantics are unchanged**: same score formula, same threshold,
  same candidate/champion identity, same opening suite. Only individual move
  *decisions* can differ from a hypothetical Python-MCTS arena at the same
  weights.
- Because the arena is otherwise fully deterministic (temp 0, no root noise,
  fixed suite, fixed weights), the native arena is **exactly reproducible** for
  a fixed seed and fixed evaluator.

**Required fixed-seed parity test** (in `tests/test_native_arena.py`):

1. **Determinism golden**: drive `native_arena.play_match` with a deterministic
   fake evaluator (hash-based logits + zero values, mirroring
   `tests/test_native_mcts.py::fake_evaluator`), `arena_seed=424242`, 20 games,
   40 sims. Assert the ordered transcript of every game (list of UCI moves) is
   byte-identical across two runs and matches a recorded golden, and that the
   final `{a,b,draws}` is stable.
2. **Single-search parity vs `mcts.py`**: on the standard start position (and
   the `benchmarks/native_mcts.py::BENCH_POSITIONS` corpus), run native MCTS and
   `mcts.py` with the *same* `FakeEvaluator` (strictly-monotonic logits), same
   `c_puct`/`virtual_loss`/`num_sims`, no root noise. Assert the temperature-0
   move is identical. (Verified today: both pick `b1c3` on the start position at
   40 sims.) Where priors are strictly distinct this also pins the visit
   distribution; where ties exist, assert only the selected move — this is the
   documented divergence boundary.
3. **Adjudication**: force a line that reaches mate / threefold via the fake
   evaluator and assert `_terminal_result` returns the correct `white_result`
   and the game terminates (length cap included).

---

## 10. Exact integration point in `arena.py::play_match`

Keep the Python path **byte-for-byte intact**. The native adapter plugs in as a
delegation at the very top of `play_match`, and the opening suite is threaded
through so the suite `_arena_gate` hashes is *guaranteed* to be the one played.

```python
# arena.py
def play_match(net_a, net_b, cfg, num_games, openings=None):
    if getattr(cfg, "arena_backend", "python") == "native":
        from native_arena import play_match as _native_play_match
        return _native_play_match(net_a, net_b, cfg, num_games, openings)
    # ---- existing python path, unchanged from here on ----
    if num_games % 2 != 0:
        raise ValueError(...)
    mcts_a = MCTS(net_a, cfg)
    mcts_b = MCTS(net_b, cfg)
    ...
```

`train._arena_gate` (train.py:850) is updated minimally so the already-generated
`openings` are passed (removing the current regenerate-and-hope-it-matches
reliance), and so a persistent engine can be injected:

```python
# train.py
def _arena_gate(cfg, net, best_net, arena_engine=None):
    ...
    openings = generate_arena_openings(...)
    suite_hash = arena_suite_hash(openings)
    if arena_engine is not None:
        result = arena_engine.play_match(
            net.state_dict(), best_net.state_dict(),
            num_games=cfg.arena_games, openings=openings)
    else:
        result = play_match(net, best_net, cfg,
                            num_games=cfg.arena_games, openings=openings)
    ...  # score / accepted / return dict unchanged
```

`run_native` (train.py:1408) constructs the engine once (after the self-play
`inference_fn` at line 1496) and passes it into every `_arena_gate` call:

```python
    arena_engine = None
    if getattr(cfg, "arena_backend", "python") == "native":
        import native_arena
        arena_engine = native_arena.NativeArenaEngine(cfg)
    ...
    outcome = _arena_gate(cfg, net, best_net, arena_engine=arena_engine)
```

`config.py` gains `arena_backend = "python"` (default) — so every existing test
and the serial/parallel paths are unchanged until a run opts into native.

---

## 11. Checklist — what MUST be preserved (verify in review)

- [ ] Paired openings: `arena_seed=424242`, `arena_opening_plies=8`,
      10 distinct openings, each played twice.
- [ ] Color swap: candidate White in Game A, champion White in Game B.
- [ ] Temperature 0 (`policy(0.0)` one-hot), no root noise
      (`dirichlet_epsilon=0`).
- [ ] `arena_games=20`, `arena_simulations=40`.
- [ ] Score `(wins + 0.5*draws)/games`, accept threshold `0.55`.
- [ ] Candidate vs champion identity (`net_a`=candidate, `net_b`=champion in
      the call from `_arena_gate`).
- [ ] `play_match` return contract `{"a", "b", "draws"}` (existing tests assert
      `set(result) == {"a","b","draws"}`).
- [ ] python-chess terminal adjudication (length cap + `outcome(claim_draw=True)`).

---

## 12. Edge cases & risks

- **Terminal opening**: `generate_arena_openings` already regenerates any
  sequence that terminates before `opening_plies` (lines 40–56); the adapter
  inherits this.
- **Empty policy**: a terminal root yields `policy(0.0) == []`; the adapter
  never searches a terminal root (game loop adjudicates first), so
  `_select_move` is never called on an empty policy. Add a defensive
  `if not policy: raise RuntimeError(...)` in `_select_move` for safety.
- **`values` shape**: `InferenceRuntime.evaluate` returns `values[B,1]`; the
  pybind `apply_evaluations` flattens C-contiguous arrays to length `B`, so
  passing `[B,1]` is valid (this is exactly what `NativeSelfPlay.run` already
  does). The adapter passes `values` through unchanged.
- **max_batch**: arena uses `max_batch=256` (matches self-play). With 40 sims
  the runtime buckets to 64, but the 256 cap is harmless and keeps the
  contract uniform.
- **Two runtimes + compile**: `NativeArenaEngine` builds two `torch.compile`
  runtimes once; this is a one-time cost per run, amortized over every arena
  gate, and is the reason the engine (not `play_match`) owns construction.
