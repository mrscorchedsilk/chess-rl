# v4-20x256: launching a new lineage

This document is the operational half of the pipeline overhaul on
`feat/gpu-pipeline-overhaul`. It covers what changed, how to start a fresh
`v4-20x256` lineage, what to watch, and how to get back.

**Nothing here has been run as a long training job.** The commands below are
documented and smoke-tested; the decision to start a real run is yours.

---

## 1. Why v4 only makes sense after the pipeline fixes

The measured reason the GPU sat at 6% was not that the model was too small on
its own — it was that the learner asked for almost no work:

| | before | after |
|---|---|---|
| optimizer steps / iteration | 9 | 96 |
| sample reuse (presentations per generated position) | 0.96 | 10.24 |
| inference batch ceiling | 256 | 4096 |
| batch at 96 concurrent games | 192 (fell as games rose) | 1152 |
| CPU/GPU overlap | none | sharded |

A 24M-parameter body under the old loop would have trained *more slowly* and
played *worse*, and the natural conclusion would have been that the bigger
body was the problem. That is why `v4-20x256` is the last change, not the
first.

## 2. Measured cost of the larger body

Uncontended RTX 2080 Ti, 96 concurrent games, 60-ply cap, `leaves_per_game=12`,
`max_batch=4096`. Reproduce with `benchmarks/overlap_bench.py`.

| body | shards | games/hour | GPU busy | nvidia-smi | power |
|---|---|---|---|---|---|
| v2-6x128 | 1 | 3770 | 11.4% | 10.9% | 67 W |
| v2-6x128 | 2 | 3721 | 16.1% | 11.6% | 72 W |
| v2-6x128 | 4 | 3562 | 26.2% | 11.6% | 73 W |
| v4-20x256 | 1 | 2292 | 46.0% | 42.0% | 152 W |
| v4-20x256 | 2 | 3033 | 62.7% | 60.0% | 199 W |
| v4-20x256 | 4 | 3188 | 67.7% | 58.4% | 200 W |

Two things follow, and they are the whole argument for v4:

1. **Sharding only pays for a large body.** With v2 it is neutral-to-negative
   (0.94–0.99×) because there is barely any GPU time to hide. With v4 it is
   1.32–1.39×.
2. **The 11× larger network costs ~15% of throughput**, not 5×: 3188 vs 3770
   games/hour. The GPU had that much headroom.

### Architecture comparison

Same pipeline, 64 concurrent games, 2 shards, 60-ply cap, one process per body
so the VRAM peaks are clean (`benchmarks/arch_compare.py`).

| body | params | ms/step | train samples/s | peak VRAM | games/hour | GPU busy | nvidia-smi | power |
|---|---|---|---|---|---|---|---|---|
| v2-6x128 | 2.17 M | 15.78 | 15,839 | 0.24 GB | 3440 | 17.6% | 10.3% | 67 W |
| v3-10x192 | 7.24 M | 31.51 | 7,933 | 0.49 GB | 3438 | 31.4% | 26.3% | 95 W |
| v4-20x256 | 24.42 M | 75.13 | 3,327 | 1.25 GB | 3022 | 54.6% | 50.7% | 188 W |

`v3-10x192` is **3.3x the parameters for no measurable throughput cost at all**
(3438 vs 3440 games/hour) — the GPU simply had that much headroom.
`v4-20x256` is 11.25x the parameters for 0.88x throughput.

VRAM is nowhere near the limit: 1.25 GB of 11 GB at the training batch. The
constraint on going larger is wall-clock per game, not memory.

If you want the safest step rather than the largest, `v3-10x192` is free on
this measurement and `v4-20x256` costs 12%. Which one is strongest per
wall-clock hour is a training experiment; neither has been trained.

## 3. The launch command

`v4-20x256` weights are incompatible with every earlier lineage by
construction — `architecture_id` is the guard that prevents cross-body tensor
loads — so it **must** start from scratch in its own directory.

```bash
cd ~/chess-rl && .venv/bin/python train.py \
  --selfplay-backend native \
  --architecture v4-20x256 \
  --checkpoint-dir "$HOME/chess-rl/checkpoints/v4" \
  --games-in-flight 96 \
  --shards 4 \
  --games-per-iteration 20 \
  --train-epoch-size 8192 \
  --replay-size 500000 \
  --num-simulations 100 \
  --arena-every 50 \
  --arena-games 100 \
  --arena-simulations 100 \
  --num-iterations 100000
```

Deliberately **not** in that command:

* `--resume` — there is nothing to resume; a fresh lineage starts empty.
* `--moves-left-head` and `--resign` — both change behaviour in ways worth
  introducing one at a time, and neither can be toggled mid-lineage
  (the head changes the `state_dict`; resignation changes the value target).
  Add them at the start of a lineage if you want them, not partway.

What each non-obvious flag buys:

| flag | why |
|---|---|
| `--games-in-flight 96` | concurrency, which sets the GPU batch. Independent of `--games-per-iteration`, which is the training cadence. |
| `--shards 4` | CPU/GPU overlap; worth 1.39× for this body, ~0 for v2. |
| `--train-epoch-size 8192` | 96 optimizer steps per iteration instead of 9. The single largest lever on learning rate. |
| `--replay-size 500000` | ~2.07 GB RAM and ~0.85 GB per checkpoint (measured). Preflighted at startup. |
| `--arena-games 100` | the promotion gate is now a lower confidence bound, so the game count sets the smallest improvement that can ever be confirmed. Measured, at threshold 0.55 / 95%: 20 games confirm a true score of 0.695+, 50 → 0.642, 100 → 0.615, 200 → 0.598, 400 → 0.584. |

## 4. What to watch

Telemetry lands in `<checkpoint_dir>/telemetry.jsonl`.

| field | meaning | act when |
|---|---|---|
| `steps_per_iteration` | gradient steps | not ~96 → `train_epoch_size` is not applying |
| `sample_reuse` | presentations per generated position | drifts far from ~10 → retune `train_epoch_size` |
| `gpu_busy_fraction` | share of the round with a batch on the GPU | well below 0.45 → shards or batch are wrong |
| `batch_mean` | merged leaf batch | far below `games_in_flight * 12` → the budget is clamping |
| `false_resignation_rate` | fraction of suppressed resignations contradicted by the real result | above ~0.05 → threshold too aggressive. `None` means *no evidence yet*, not zero |
| `score_ci_low` | arena lower bound | promotion requires this ≥ threshold |

## 5. Rollback

Every change is additive and behind a default, so rollback is graduated
rather than all-or-nothing.

1. **Config only, no code change.** The pre-existing behaviour is reachable
   from defaults: `selfplay_shards = 1`, `selfplay_leaves_per_game = 0`,
   `selfplay_max_batch = 256`, `train_epoch_size = 768`,
   `optimizer = "adam"`, `lr_schedule = "none"`, `augment_colour_flip = 0.0`,
   `arena_require_lower_bound = False`, `train_channels_last = False`,
   `train_prefetch = 0`.
2. **Abandon the branch.** `feat/gpu-pipeline-overhaul` is a worktree at
   `~/chess-rl-v4`; `~/chess-rl` is untouched and still on
   `fix/native-selfplay-diversity`. The v2 lineage and its checkpoints were
   never written to.
3. **Native module.** The worktree builds into `_pkg/` and shadows the shared
   venv via `PYTHONPATH`; the venv's installed `chess_rl_native` is byte-for-byte
   what it was (verified: mtime unchanged). Nothing needs undoing.

## 6. Resuming the existing v2 lineage under this branch

The v2 lineage was trained with `torch.optim.Adam`. This branch defaults to
`AdamW`, and the two share a `state_dict` layout, so a resume would load
cleanly and silently switch coupled L2 for decoupled decay partway through a
2,300-iteration run. That is refused; pass the optimizer it was trained with:

```bash
cd ~/chess-rl && .venv/bin/python train.py \
  --selfplay-backend native --resume \
  --checkpoint-dir "$HOME/chess-rl/checkpoints/v2" \
  --optimizer adam --lr-schedule none --train-epoch-size 768
```

`--train-epoch-size 768` reproduces the old 9-steps-per-iteration behaviour.
Dropping it is a real change to that lineage's training dynamics — a
reasonable thing to want, but it should be a decision, not a side effect of
switching branches.

## 7. Known limitations

* **Self-play throughput is 1.40×, not ≥2×.** Measured against the original
  defaults, same body. The round is ~85–90% CPU tree descent on an 8-core CPU;
  overlap can only hide the GPU's share.
* **>60% GPU utilisation holds only for v4.** 67.7% in-driver / 58.4–60.0%
  sampled at shards 2–4. For v2 it stays near 11–26%.
* **No strength claim.** Every number here is throughput. Which body is
  strongest per wall-clock hour is a training experiment that has not been run.
* **`evaluate.py` computes its baseline Wilson intervals differently** from
  `stats.py` (fractional successes over `n` games vs doubled trials). Both are
  defensible; having two conventions in one repo is not. Worth reconciling.
* **The moves-left head and resignation are untested at scale** — unit-tested,
  but never run for a long training job.
