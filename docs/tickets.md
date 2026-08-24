# Flash Implementation Tickets — Telemetry → Native-Arena → Tests → Benchmark

Implementation-only tickets for the V4 Flash agent. Planning/design authority
is `docs/telemetry-design.md` and `docs/native-arena-design.md`; these tickets
only bound the work. The reviewer (V4 Pro) signs off each ticket before the
next starts.

**Dependency order:** A (telemetry) → B (native-arena adapter) → C (correctness
tests) → D (benchmark). C gates D (the benchmark is only meaningful once the
adapter is proven correct and deterministic). A is first so the arena speedup
can be measured end-to-end.

**Do NOT** touch (project-wide red lines): checkpoint frequency, batch
512/1024 defaults, `games_per_iteration` without a benchmark, actor sharding,
training+inference overlap, simulation count bumps for GPU utilization, or the
v2 architecture. All work is behind a flag with the python path preserved as
the default until C passes.

---

## Ticket A — Permanent phase telemetry

**Goal:** land the telemetry layer per `docs/telemetry-design.md` with zero
semantic change and hard swallow-guards.

**Files to change:**
- NEW `telemetry.py` — `PhaseTimer`, `emit`, `sample_resources`,
  `replay_diversity`, `game_trajectory_hash` (contract in §4 of the design).
- `gpu_runtime.py` — add `call_count` / `batch_b` deque / `total_forward_s`
  counters + `stats()` in `InferenceRuntime`; touch `evaluate` (lines 404–413)
  only, no tensor/RNG/order change.
- `native_selfplay.py` — instrument `NativeSelfPlay.run` (lines 98–112):
  gather/apply/advance counts + timings, per-gather `B`, inference-call count,
  sims/sec, games/hour, per-game trajectory hashes.
- `train.py` — wrap self-play / training / arena / checkpoint phases in
  `run_native` (loop at 1505–1567) with `PhaseTimer`; emit `phase`, `resource`,
  and (on cadence) `diversity` records; add `cfg.telemetry_*` knobs.
- `arena.py` + NEW `native_arena.py` — wrap `play_match` bodies with
  `PhaseTimer`; emit `phase="arena"` (side effect only; return contract
  unchanged).
- `config.py` — add `telemetry_enabled=True`, `telemetry_path`,
  `telemetry_resource_every=1`, `telemetry_diversity_every=<arena_every>`.

**Exact test commands:**
```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_telemetry.py -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
    tests/test_native_train_e2e.py tests/test_selfplay_seed.py \
    tests/test_observability.py tests/test_milestones.py \
    tests/test_arena_openings.py -q
```
(New `tests/test_telemetry.py` per §6 of the design: swallow-guard, schema,
determinism, diversity-parity, resource-grace.)

**Acceptance thresholds:**
- All tests green, including the pre-existing suite.
- A forced `emit` failure does NOT abort `train.run_native` (test proves it).
- Replay examples / move transcripts are byte-identical with telemetry on vs
  off (determinism test proves it).
- `telemetry.replay_diversity` matches `scripts/audit_replay.audit_replay` on a
  synthetic buffer.
- Overhead: steady-state < 1% wall time; diversity audit ≤ 0.5 s and only on
  cadence.

**Rollback:**
```
git checkout -- gpu_runtime.py native_selfplay.py train.py arena.py config.py
rm -f telemetry.py tests/test_telemetry.py
```
(or `git revert <ticket-commit>`). Telemetry is additive + swallow-guarded, so
revert is a clean removal of the added calls and the new module.

---

## Ticket B — Native-arena adapter

**Goal:** implement `native_arena.py` and wire the flag per
`docs/native-arena-design.md`, with the Python arena path byte-for-byte intact
and the default still `"python"`.

**Files to change:**
- NEW `native_arena.py` — `_new_mcts`, `_run_native_search`, `_select_move`,
  `_play_native_game`, `play_match`, `NativeArenaEngine` (exact signatures in
  §4–§5 of the design). **Do not reuse** `benchmarks/native_mcts.run_native_search`
  verbatim (it reads the un-exposed `mcts.num_simulations`; the design's
  `_run_native_search` tracks the budget from `cfg.arena_simulations`).
- `arena.py` — add the delegation at the top of `play_match` (line 128) on
  `getattr(cfg, "arena_backend", "python") == "native"`; pass `openings`
  through. Python path otherwise untouched.
- `train.py` — `_arena_gate` (line 850) gains optional `arena_engine=None`
  param and passes the already-generated `openings`; `run_native` (line 1408)
  constructs `NativeArenaEngine` once and passes it in.
- `config.py` — add `arena_backend = "python"` (default).

**Exact test commands (smoke only — full correctness is Ticket C):**
```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_arena_openings.py -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_native_mcts.py tests/test_native_train_e2e.py -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python -c "import native_arena"   # import smoke
```
**Acceptance thresholds:**
- Python backend: all `tests/test_arena_openings.py` still pass unchanged
  (flag default = python ⇒ no behaviour change).
- Native backend (ad-hoc, fake evaluator): `native_arena.play_match` returns
  `{a,b,draws}` summing to `num_games`, plays 10 distinct openings twice with
  colors swapped, temp-0, no root noise, and is deterministic for a fixed
  fake evaluator + `arena_seed=424242`.
- `native_arena` never mutates the trainer's nets (only `state_dict()` copies).

**Rollback:**
```
rm -f native_arena.py
git checkout -- arena.py train.py config.py
```
(Flag default is `"python"`, so reverting the three edits restores exact prior
behaviour.)

---

## Ticket C — Native-arena correctness tests (gates Ticket D)

**Goal:** lock determinism + parity + adjudication per §9 of
`docs/native-arena-design.md`.

**Files to change:**
- NEW `tests/test_native_arena.py` with, at minimum:
  1. `test_native_arena_deterministic_fixed_seed` — two runs with the same
     deterministic fake evaluator (hash logits + zero values) produce
     byte-identical per-game move transcripts and identical `{a,b,draws}`;
     pin a recorded golden transcript hash.
  2. `test_native_arena_preserves_paired_openings_and_color_swap` — spy on
     `_play_native_game` to assert 10 distinct openings × 2 color swaps
     (mirror `test_arena_plays_each_opening_twice_with_swapped_colors`).
  3. `test_native_arena_result_contract` — `{a,b,draws}` keys only, sum ==
     `num_games`; odd `num_games` raises `ValueError`.
  4. `test_native_mcts_matches_python_reference_fixed_seed` — on
     `START_FEN` + `benchmarks/native_mcts.BENCH_POSITIONS`, drive native MCTS
     and `mcts.py` with the *same* `FakeEvaluator`, same `c_puct`/`virtual_loss`
     /`num_sims`, no root noise; assert identical temperature-0 move. Document
     the tie-break boundary (assert selected move, not visit distribution, on
     tied positions).
  5. `test_native_arena_terminal_adjudication` — force mate and threefold lines;
     assert `_terminal_result` yields the correct `white_result` and the length
     cap returns a draw.
- Reuse `benchmarks/native_mcts.FakeEvaluator`/`FakeNet` as the shared fixture
  (do not duplicate).

**Exact test command:**
```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_native_arena.py -q
```

**Acceptance thresholds:**
- All 5 tests pass deterministically (run twice, identical results).
- The golden transcript is recorded and stable across the ticket's final run.
- Parity holds on every reference position where priors are strictly distinct.

**Rollback:**
```
rm -f tests/test_native_arena.py
```
(Ticket B's adapter is reverted separately if C exposes a defect.)

---

## Ticket D — Arena benchmark

**Goal:** quantify the Python-arena → native-arena speedup and confirm the
phase telemetry works end-to-end. **Prerequisite: Ticket C is green.**

**Files to change:**
- NEW `benchmarks/arena_bench.py` — two modes:
  - `--compare-python` (CPU-only, `CUDA_VISIBLE_DEVICES=""`, deterministic
    `FakeEvaluator`): time the full 20-game / 40-sim / 8-ply / seed-424242
    suite through `arena.play_match` (python) vs `native_arena.play_match`
    (native, fake evaluator). Reports per-suite wall time and the native/python
    speedup.
  - `--gpu` (real `InferenceRuntime` on the 2080 Ti): time `native_arena.play_match`
    end-to-end with the production runtimes; also emit the `phase="arena"`
    telemetry record to prove the Ticket-A path.
- Optionally extend `benchmarks/native_mcts.py` (do NOT modify its
  `run_native_search` beyond fixing the `num_simulations` reference if reused).

**Exact benchmark commands:**
```
CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmarks/arena_bench.py \
    --sims 40 --games 20 --opening-plies 8 --seed 424242 --compare-python
.venv/bin/python benchmarks/arena_bench.py \
    --gpu --sims 40 --games 20 --opening-plies 8 --seed 424242
```

**Acceptance thresholds** (grounded in verified facts: Python arena median
~125.49 s, range 112.54–141.49 s):
- Native arena full-suite median ≤ **60 s** (≥ 2× speedup vs the Python
  arena), with the CPU fake-evaluator number as the correctness-neutral
  floor and the GPU number as the production number.
- The native suite's `{a,b,draws}` sum to 20 and its `score` uses the exact
  `(wins + 0.5*draws)/games` formula with the 0.55 threshold (semantics
  unchanged).
- The run is reproducible (repeat the GPU run; transcript hash matches the
  golden from Ticket C for the fake-evaluator path).
- A `phase="arena"` telemetry record is emitted and parses (proves A+B+C
  integration).

**Rollback:**
```
rm -f benchmarks/arena_bench.py
```
No production behaviour changes in this ticket beyond what A–C already gated.

---

## Cross-cutting notes for the implementer

- Keep the python self-play / arena path and the serial `run()` untouched in
  behaviour; every new default is the old behaviour (`arena_backend="python"`,
  telemetry swallow-guarded).
- Each ticket is independently revertible; commit them separately so a revert
  of D never forces a revert of A.
- Re-run the milestone tests after B/C (`tests/test_milestones.py`,
  `tests/test_arena_openings.py`) — the arena adapter must not disturb the
  milestone capture or the paired-opening contract.
