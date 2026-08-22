# Sprint A–D acceptance contract

This document defines the conditions required before a fresh v2 training run may start.

## Sprint A — reliability and observability

- Training is stopped and the v1 iteration-56 checkpoint is archived with SHA-256 hashes.
- Source and UIs have Git baselines and dedicated implementation branches.
- `latest.pt` is written atomically after every completed iteration and on graceful shutdown.
- Resume restores candidate, accepted best, optimizer, RNGs, replay data, run ID, generation, iteration, and config snapshot.
- Worker or inference-server failure becomes a bounded error; no unbounded `Queue.get()` hang.
- Dashboard distinguishes live iteration from saved iteration and associates metrics with a run ID.
- Default controller binding is localhost; remote exposure is explicit.

## Sprint B — chess-state and action correctness

- Policy space is 73 planes × 64 origins = 4,672 actions.
- Every legal move round-trips uniquely, including all four promotions.
- Policy targets preserve total probability mass.
- Network input contains eight positions of piece history plus side, castling, en-passant, halfmove, and repetition context.
- Batch and scalar encoders are equivalent.
- Search and self-play agree on claimable draw terminal states.

## Sprint C — defensible learning and evaluation

- Training performs real epochs over a bounded shuffled sample, not three isolated minibatches.
- Policy loss, value loss, entropy, optimizer steps, replay size, run ID, and generation are logged separately.
- Self-play uses the accepted model; rejected challengers do not silently become the next teacher.
- Arena has no root noise, uses paired colors, scores draws as 0.5, and records event-only metrics.
- Fixed random/material baselines and tactical competence tests emit reproducible JSON results.

## Sprint D — safe performance and product integration

- Default worker count is 8 for the current 8-core/16-thread host.
- Replay storage is compressed and policy targets are sparse at rest.
- IPC failures are explicit and queues/processes close cleanly.
- Model server exposes a validated local `/move` API and can hot-reload accepted weights.
- Light Chess can select an AI side and simulation budget while preserving two-human play.
- Dashboard and both browser surfaces render without console errors.

## Final gate

A fresh v2 run may start only after:

1. All Python and Node tests pass.
2. CPU smoke tests pass from a clean v2 checkpoint directory.
3. A tiny end-to-end parallel training run saves and resumes exactly.
4. Independent review reports no unresolved critical or high-severity correctness issue.
5. Git working trees are clean and all implementation commits are recorded.
