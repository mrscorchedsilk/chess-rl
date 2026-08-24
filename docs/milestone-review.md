# Milestone Review — Arena Champion Capture & Checkpoint Retention

**Verdict: GO** (non-blocking; two correctness notes to fold into a follow-up ticket).

Reviewed against the `optimize/native-arena` worktree at commit `bad22bf`
("chore: snapshot pre-optimization state — milestone scaffolding + warm-start
train support"). The `train.py` milestone changes described as "recent
uncommitted" are now committed as part of that snapshot.

---

## 1. Scope reviewed

| Artifact | Lines | Role |
|---|---|---|
| `train.py::_save_milestone` (line 718) + 3 call sites (1077, 1320, 1538) | 59 lines added | persist accepted champion on arena promotion |
| `scripts/backfill_milestones.py` | 155 | retroactively extract champions from versioned snapshots |
| `scripts/prune_checkpoints.py` | 202 | retention policy for `ckpt-iter*.pt` (dry-run default) |
| `tests/test_milestones.py` | 150 | CPU-only behaviour tests for milestone capture |

---

## 2. What the change does

1. On every **arena acceptance**, the trainer writes a **weights-only
   milestone** `milestones/best-genNNNN-iterNNNN-<run_id>.pt` (~8 MB) plus a
   JSON sidecar recording arena provenance (W/D/L, score, threshold) and
   architecture identity (`architecture_id`, `policy_size`, `num_input_planes`,
   `board_size`). This is wired identically into all three loops: serial
   `run()`, `run_parallel()`, and native `run_native()`.
2. `backfill_milestones.py` recreates historical milestones from the existing
   ~99 MB versioned snapshots, keyed off `event: "arena"` + `accepted: true`
   records in `training.jsonl`.
3. `prune_checkpoints.py` deletes ordinary per-iteration snapshots while
   **always** keeping `latest.pt` / `best.pt` / the two meta files / everything
   under `milestones/`, plus a keep-set of last-N, a sparse ladder, and a
   neighborhood around each promotion.

The combined effect: arena champions survive snapshot pruning as small,
self-describing, re-loadable artifacts (consumed by the warm-start path), and
the ~99 MB rollback snapshots can be pruned without losing strength history.

---

## 3. Verification performed

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_milestones.py -q
  -> 3 passed in 2.36s

python -m py_compile scripts/backfill_milestones.py scripts/prune_checkpoints.py \
    train.py arena.py native_selfplay.py gpu_runtime.py
  -> OK

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
    tests/test_arena_openings.py tests/test_native_mcts.py \
    tests/test_native_actor.py tests/test_native_train_e2e.py -q
  -> 29 passed in 5.92s
```

The milestone tests verify: acceptance writes exactly one immutable milestone
whose weights are byte-identical to `best.pt`; rejection writes none; and two
successive promotions produce two distinct, never-overwritten milestones with
incrementing generations.

---

## 4. Strengths (findings that support GO)

1. **Atomicity.** Both the `.pt` and `.json` are written via
   `tmp + os.replace`, so a crash can never leave a torn milestone/sidecar pair.
   `backfill_milestones.py` and `prune_checkpoints.py` follow the same pattern.
2. **Correct capture point.** `_save_milestone` is called *after*
   `best_net.load_state_dict(net.state_dict())` and `_save_best_atomic`, so the
   milestone captures the **accepted champion**, never the losing candidate.
   Identical placement in all three run loops.
3. **Self-describing sidecar.** The sidecar carries everything needed to
   reload/validate the weights (`architecture_id` + tensor dims) and to explain
   *why* the milestone exists (arena W/D/L, score, threshold, `saved_at`).
4. **Immutable & idempotent.** Filenames embed `(generation, iteration,
   run_id)` and are never overwritten; `backfill` skips already-present
   milestones and both scripts default to dry-run where deletion is involved.
5. **Pruning safety is layered.** `prune_checkpoints.py` treats
   `milestones/` as untouchable, keeps active files unconditionally, dedupes
   duplicate per-iteration snapshots, and emits a human-readable manifest +
   freed-space estimate before deleting anything (only under `--apply`).
6. **Warm-start integration is consistent.** The milestone weights-only format
   matches the warm-start loader's expectations, so a milestone is directly
   re-loadable as a new lineage root (validated indirectly by the warm-start
   suite).

---

## 5. Findings / risks (non-blocking)

### F1 — `backfill_milestones.py` selects snapshots by mtime, ignoring `run_id` (medium)

`find_snapshot()` (`scripts/backfill_milestones.py:47`) picks the newest
`ckpt-iterNNNN-*.pt` for an iteration and never checks that the snapshot belongs
to the accepted event's `run_id`. If a `checkpoint_dir` ever holds snapshots
from **more than one lineage** (which is exactly what warm-starts produce, each
with a fresh `run_id`), the backfill could load best weights from the *wrong*
run while labelling the milestone with the accepted event's `run_id`.

**Recommendation:** filter candidates by `run_id` (via `checkpoint_meta.json`
or by rejecting snapshots whose provenance disagrees), or at minimum emit a
loud warning when the chosen snapshot's lineage cannot be confirmed to match
`ev["run_id"]`. Ticket this as part of a milestone-scripts hardening task.

### F2 — `backfill_milestones.py` hardcodes `checkpoint_format: "schema-v3"` (low)

`read_config`/payload extraction sets `"checkpoint_format": "schema-v3"`
unconditionally (`backfill_milestones.py:115`), while reading
`"schema_version"` from the source payload. If a pre-v3 snapshot is ever
backfilled, it would be mislabelled schema-v3. All production snapshots are
schema-v3 today, so this is latent only. Prefer deriving the label from the
source payload's own `checkpoint_format` field, falling back to `"schema-v3"`.

### F3 — `schema_version` duality in sidecars (low, documentation only)

`_save_milestone` (and `_publish_best`) record both
`"schema_version": 2` and `"checkpoint_format": "schema-v3"`. This is the
intended on-disk convention (numeric `schema_version` stays 2; the v3 feature
layer rides on `checkpoint_format`), but a future reader must check
`checkpoint_format`, not `schema_version`, to know the payload is v3. Add a
one-line comment to `_save_milestone` so the distinction is explicit.

### F4 — `backfill_milestones.py` has a side effect in dry-run (low)

`os.makedirs(milestones_dir, exist_ok=True)` (`backfill_milestones.py:81`) runs
before the dry-run branch, so `--dry-run` still creates an empty `milestones/`
directory. Cosmetic; move the `makedirs` into the write path or gate it on
`not args.dry_run`.

### F5 — No direct unit tests for the two scripts (medium, coverage gap)

`tests/test_milestones.py` covers the `train.py` path only. Neither
`backfill_milestones.py` nor `prune_checkpoints.py` has a unit test, so F1/F2/F4
are currently unguarded. Recommend `tests/test_checkpoint_retention.py` with
synthetic fixture directories: backfill idempotency, run_id-mismatch handling,
prune keep-set rules, dry-run-no-delete, and archive pruning. This is a natural
addition to the milestone-scripts hardening ticket.

### F6 — Coupling of snapshot cadence and arena cadence (low, by design)

`backfill` needs `ckpt-iterNNNN-*.pt` to exist at the promotion iteration.
Production has `arena_every = checkpoint_every_iterations = 20`, so both fire at
the same iterations and the coupling holds. If the two ever diverge, backfill
degrades to `SKIP (no snapshot)` — safe (never wrong weights), only incomplete.
Worth a note, not a change.

---

## 6. GO / NO-GO

**GO.** The milestone-capture feature is atomic, self-describing, immutable,
wired correctly into all three run loops, and its behaviour tests pass. Nothing
found blocks proceeding to the telemetry / native-arena work.

**Conditions attached to GO** (all non-blocking, fold into one follow-up
ticket):
- F1 (run_id-aware snapshot selection in backfill) — implement before relying
  on backfill across multiple warm-start lineages.
- F5 (unit tests for both scripts) — add with the hardening pass.

The `arena.py` paired-opening / scoring semantics remain untouched by this
milestone work, so the native-arena design in
`docs/native-arena-design.md` builds on a clean base.
