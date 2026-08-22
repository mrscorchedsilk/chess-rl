# Migration from ChessNet v1 to v2

## Why this is a clean restart

The v1 network used 18 input planes and a 4,096 `from_square × to_square` policy. That action encoding collapses queen, rook, bishop, and knight promotions onto one index, and the state omits repetition/halfmove history. The v2 network uses eight positions of history and a 73-plane spatial action map (4,672 actions), so v1 tensors cannot be loaded safely into v2.

## Preserved v1 artifacts

The stopped prototype is archived at:

`backups/2026-08-22-iter56-prototype/`

The archive contains:

- `latest.pt` saved at iteration 56
- the iteration-56 versioned snapshot
- the v1 `best.pt`
- metrics, controller log, stats, and configuration
- SHA-256 checksums
- verified Git bundles for the original chess learner, Light Chess UI, and training dashboard

The dashboard advanced beyond iteration 56 before shutdown, but only iteration 56 was durable. No v1 artifact is deleted or overwritten by v2.

## Namespace boundary

- v1 checkpoints: `checkpoints/`
- v2 checkpoints: `checkpoints/v2/`

The v2 loader requires checkpoint schema version 2 and refuses legacy snapshots with a clear incompatibility error. Never copy or rename a v1 `latest.pt` into the v2 directory.

## Fresh v2 start

After all quality gates pass:

```bash
cd ~/chess-rl
.venv/bin/python train.py --workers 8
```

Resume only a v2 run:

```bash
.venv/bin/python train.py --workers 8 --resume
```

Before the first real run, verify that `checkpoints/v2/` is empty and that all Python and Node tests pass.

## Comparison policy

The v1 iteration-56 candidate may be used only as an engineering comparison. Its architecture and policy space differ, so direct weight transfer is invalid. Strength comparisons must use paired-color games through the external evaluation harness rather than tensor or loss comparisons.
