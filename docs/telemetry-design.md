# Telemetry Design — Permanent Phase & Resource Telemetry

## 0. Objectives & non-negotiable constraints

Permanent, machine-readable, low-overhead telemetry for the native training
loop that measures:

1. wall-clock timing of each phase — native self-play, arena, training,
   checkpoint, and the native gather-apply-advance inner loop;
2. inference call count and batch distribution (min / mean / p50 / p90 / max);
3. simulations/second and games/hour;
4. GPU / CPU / RAM / swap / VRAM;
5. replay diversity and trajectory hashes.

**Two hard guarantees, enforced by design and by test:**

- **Telemetry must never kill training.** Every emit/sample is wrapped in
  `try/except Exception: pass` (the exact pattern already used by
  `train._log_metrics`, train.py:107–134). A telemetry failure is swallowed and
  logged nowhere near the control flow.
- **Telemetry must not change training or game semantics.** All measurements
  are `time.perf_counter()` deltas around existing code, or counters derived
  from arrays that are already materialized. There is **no** extra forward
  pass, no RNG draw, no data mutation, no reseeding, and no reordering. A
  run's replay examples, move choices, checkpoints and scores must be
  bit-identical with telemetry on vs off (verified by a determinism test).

---

## 1. Output location & transport

- **File:** JSON Lines, one JSON object per line, appended (never rewritten).
- **Default path:** `os.path.join(cfg.checkpoint_dir, "telemetry.jsonl")`,
  overridable via `cfg.telemetry_path`.
- **Enabled by default** for the native backend (`cfg.telemetry_enabled = True`).
  Because it is swallow-guarded and semantic-free, leaving it on is safe.
- **Every record** carries a top-level `"schema": "telemetry/v1"` and a `"type"`
  discriminator, so consumers can filter by type and future readers can detect
  schema drift. Numeric timestamps are `time.time()` (Unix epoch seconds);
  durations are `time.perf_counter()` seconds.

---

## 2. Record types

### 2.1 `type: "phase"` — one record per phase per iteration

Emitted at phase end. Required fields plus phase-specific counters.

```
{
  "schema": "telemetry/v1",
  "type": "phase",
  "t": 1735000000.123,
  "run_id": "ac67ffe7595f",
  "iteration": 42,
  "generation": 3,
  "phase": "selfplay" | "gather_apply_advance" | "training" | "arena" | "checkpoint",
  "backend": "native" | "python",
  "duration_s": 18.42
}
```

Phase-specific fields:

- **`selfplay`** (native): `games`, `examples`, `inference_calls`,
  `batch_min`, `batch_mean`, `batch_p50`, `batch_p90`, `batch_max`,
  `simulations` (total descent sims run this round), `sims_per_s`,
  `games_per_hour`, `round_seed`.
- **`gather_apply_advance`** (native, nested inside self-play):
  `gather_calls`, `apply_calls`, `advance_calls`, `gather_s`, `apply_s`,
  `advance_s` (cumulative within the self-play round), plus the same
  `batch_*` distribution (B per gather).
- **`training`**: `steps`, `batches`, `train_batch_size`, `policy_loss`,
  `value_loss`, `entropy`, `optimizer_steps`.
- **`checkpoint`**: `snapshot` (bool: versioned hardlink taken), `bytes`
  (approx size of the written `latest.pt`), `reason`.
- **`arena`**: `arena_games`, `arena_sims`, `wins`, `draws`, `losses`,
  `score`, `accepted`, `opening_seed`, `opening_pairs`,
  `opening_suite_hash`, `candidate_moves` / `champion_moves` (optional totals).

### 2.2 `type: "resource"` — one record per iteration (or per cadence)

```
{
  "schema": "telemetry/v1",
  "type": "resource",
  "t": 1735000000.123,
  "run_id": "ac67ffe7595f",
  "iteration": 42,
  "cpu_percent": 62.5,          # psutil.cpu_percent(interval=None) overall
  "cpu_count": 16,              # logical cores
  "ram_used_mb": 8123,
  "ram_total_mb": 15360,
  "ram_percent": 52.9,
  "swap_used_mb": 0,
  "swap_total_mb": 2048,
  "gpu_util_percent": 42.0,     # torch.cuda.utilization() 0..100
  "vram_used_mb": 4123,         # torch.cuda.mem_get_info(): total - free
  "vram_total_mb": 11264,
  "torch_alloc_mb": 2304,       # torch.cuda.memory_allocated() (model/buffers)
  "torch_reserved_mb": 2816     # torch.cuda.memory_reserved()
}
```

No new dependencies: VRAM/GPU come from `torch.cuda` (`mem_get_info`,
`utilization`, `memory_allocated`, `memory_reserved`); CPU/RAM/swap come from
`psutil` (already installed). Both are imported lazily and every field degrades
to `null` (or is omitted) if the source is unavailable — **never raises**.

### 2.3 `type: "diversity"` — replay diversity + trajectory hashes (per cadence)

```
{
  "schema": "telemetry/v1",
  "type": "diversity",
  "t": 1735000000.123,
  "run_id": "ac67ffe7595f",
  "iteration": 42,
  "source": "replay_buffer" | "selfplay_round",
  "replay_size": 50000,
  "unique_packed_states": 12345,
  "unique_exact_examples": 45678,
  "unique_state_fraction": 0.2469,
  "unique_example_fraction": 0.9136,
  "unique_trajectory_hashes": 89,
  "most_repeated_trajectory_count": 3,
  "trajectory_hashes": ["b40e…16-hex…", "…"]     # bounded sample, <= 32 entries
}
```

- **`source: "replay_buffer"`**: computed with the **same BLAKE2
  (digest_size=16) hashing as `scripts/audit_replay.py`** (packed state +
  state extras + sparse policy indices/probs + z) but on the **live**
  `ReplayBuffer` (not a checkpoint), so no `torch.load` round-trip. This is the
  port of `audit_replay.audit_replay` minus the checkpoint I/O.
- **`source: "selfplay_round"`**: one BLAKE2 digest per finished self-play
  game, over the game's ordered `(state, pi, z)` examples; emitted by
  `NativeSelfPlay.run`. Captures *generation-time* trajectory identity, cheap
  (12 games/iter), and is what makes a collapsed-to-a-few-trajectories
  regression visible immediately rather than at audit time.

---

## 3. Instrumentation map (exact locations)

### 3.1 `train.py` — aggregation & phase boundaries

| Location | Instrument |
|---|---|
| `_epoch_train` (line 777) | wrap body in `PhaseTimer("training")`; on exit emit `phase` with `steps/batches/losses/entropy/optimizer_steps` |
| `_arena_gate` (line 850) | wrap the `play_match(...)` call in `PhaseTimer("arena")`; emit `phase` with arena fields (W/D/L/score/accepted/hash) |
| `_save_latest` + `_snapshot_checkpoint` (lines 702, and the `_save_latest` used at 1556/1591) | wrap in `PhaseTimer("checkpoint")`; emit `phase` with `snapshot`/`bytes`/`reason` |
| `run_native` loop (lines 1505–1567) | wrap step 1 self-play `sp.run()` in `PhaseTimer("selfplay")` (emit at 1523 after `buffer.extend`); end of iteration (after `_log_metrics`, line 1566) emit `resource` and (on cadence) `diversity` |
| `_log_metrics` (line 107) | leave unchanged — it already demonstrates the required swallow-guard pattern |

`run()` and `run_parallel()` (lines 962 and 1158) are instrumented with the
same four phase timers for uniformity, but the native backend
(`run_native`) is the primary target.

### 3.2 `native_selfplay.py` — gather-apply-advance + trajectory hashes

`NativeSelfPlay.run` (lines 98–112) is the only place the gather/apply/advance
loop exists. Instrument:

- count `gather_calls`, `apply_calls`, `advance_calls` (already structurally
  one of each per loop iteration; just count them);
- record `B = len(tokens)` per gather into a bounded list for the `batch_*`
  distribution, and `inference_calls += 1` per non-empty gather;
- accumulate `gather_s` / `apply_s` / `advance_s` around the three calls;
- `simulations += num_simulations` per completed search (advance), enabling
  `sims_per_s = simulations / duration_s` and
  `games_per_hour = games * 3600 / duration_s`;
- after `finished_games()` (line 109), hash each game's ordered examples into
  `trajectory_hashes` and attach to the emitted record.

Because the inference fn is injected (a fake in tests), **all** of the above is
computed in `native_selfplay` from data it already holds — it does not depend on
the GPU runtime. This keeps the stats meaningful in CPU tests.

### 3.3 `gpu_runtime.py` — canonical inference-call counter

`InferenceRuntime.evaluate` (lines 404–413) is the single choke point for
production inference. Add per-instance counters only (no tensor/RNG/order
change):

- `self.call_count += 1`;
- append `B` to `self.batch_b` (a `collections.deque(maxlen=10_000)`);
- accumulate `self.total_forward_s` (perf_counter delta around the whole
  evaluate) and optionally split `prepare`/`forward_device`/`copy_back`.

Expose `def stats(self) -> dict` that returns `{"calls", "batch_min",
"batch_mean", "batch_p50", "batch_p90", "batch_max", "total_forward_s"}`
(percentiles computed from the deque at read time; cheap at 10k elements).

`train.py` reads `inference_fn.runtime.stats()` once per iteration to merge the
canonical GPU-side numbers into the `resource`/`selfplay` records (the
`native_selfplay` numbers and the `gpu_runtime` numbers should agree; the test
asserts this relationship, not exact equality across backends).

### 3.4 `arena.py` (and `native_arena.py`) — arena timing

`play_match` (arena.py:128) and `native_arena.play_match` wrap their bodies in
`PhaseTimer("arena")` and, on completion, call `telemetry.emit(...)` with
`duration_s`, `backend`, `arena_games`, `arena_sims`. The **return contract
`{"a","b","draws"}` is unchanged** (existing tests assert
`set(result) == {"a","b","draws"}`) — timing is a pure side effect.

---

## 4. Helper module — `telemetry.py` (new, repo root)

```
# telemetry.py  (contract)
class PhaseTimer:
    def __init__(self, name: str): ...
    def __enter__(self) -> "PhaseTimer": ...   # records perf_counter start
    def __exit__(self, *exc) -> None: ...      # computes duration_s; swallows
    @property
    def duration_s(self) -> float: ...

def emit(cfg, record: dict) -> None:
    """Append one JSON line to cfg.telemetry_path. NEVER raises."""
    # json.dumps + newline append, inside try/except Exception: pass

def sample_resources() -> dict:
    """psutil + torch.cuda snapshot; every field degrades to None. NEVER raises."""

def replay_diversity(buffer) -> dict:
    """Port of scripts/audit_replay.audit_replay hashing onto the live buffer."""

def game_trajectory_hash(examples) -> str:
    """BLAKE2(digest_size=16) over the ordered (state, pi, z) of one game."""
```

`emit` writes to a file handle opened/closed per call (append mode), so a
full-disk or permission failure is caught and training continues. No buffering
state is kept in the module that could be corrupted.

---

## 5. Cadence & overhead budget

| Signal | Cadence | Budget |
|---|---|---|
| `phase` records | every iteration, every phase | ~0 cost (a few appends) |
| `resource` | every iteration (or `cfg.telemetry_resource_every`, default 1) | < 5 ms (psutil + torch.cuda are cheap reads) |
| `diversity` (replay_buffer) | every `cfg.telemetry_diversity_every` iterations (default = `arena_every` = 20) | ~0.3 s (BLAKE2 over ~50k × 6.6 KB packed states); **off the hot path** |
| `diversity` (selfplay_round) | every iteration | ~0 (12 games × few KB) |
| batch distribution | accumulated per round; percentiles at read time | ~0 |

Total steady-state overhead target: **< 1% of wall time**, dominated by the
periodic replay-diversity audit (0.3 s per 20 iterations ≈ 0.015 s/iter).

---

## 6. Verification (must pass)

1. **Swallow-guard**: a test monkeypatches `telemetry.emit` to raise and
   asserts `train.run_native` still completes an iteration (no exception
   escapes into the loop).
2. **Schema conformance**: every emitted line parses as JSON, carries
   `schema == "telemetry/v1"` and a valid `type`; required fields present per
   §2.
3. **Semantic neutrality**: run
   `tests/test_native_train_e2e.py::test_native_selfplay_is_deterministic_given_seed`
   and `tests/test_selfplay_seed.py` with telemetry enabled and disabled;
   assert byte-identical replay examples / transcripts (telemetry changes
   nothing).
4. **Diversity parity**: `telemetry.replay_diversity` on a small synthetic
   buffer returns the same `unique_*` counts as
   `scripts/audit_replay.audit_replay` computed on a checkpoint saved from that
   buffer.
5. **Resource grace**: with CUDA unavailable (CI), `sample_resources()` returns
   GPU fields as `null` without raising; with psutil present, CPU/RAM fields
   are populated.

Commands (CPU-only):

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_telemetry.py -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
    tests/test_native_train_e2e.py tests/test_selfplay_seed.py \
    tests/test_observability.py tests/test_milestones.py -q
```

---

## 7. Explicit non-goals

- No changing the existing `training.jsonl` metric records (they remain the
  canonical training event stream; telemetry is additive and separate).
- No distributed/streaming transport — local JSONL append only.
- No change to checkpoint format, replay encoding, or arena scoring.
- No per-leaf/per-ply telemetry (that would add hot-path overhead); the
  `gather_apply_advance` and `selfplay` phases aggregate at the round level.
