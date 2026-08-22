# ChessNet v2 model card

## Status

**Architecture implementation and validation in progress. No v2 model has been trained or assigned an Elo.**

The archived v1 candidate is retained only as an engineering baseline. Its accepted `best.pt` was verified to equal the deterministic random initialization, so it is not a trained production model.

## Intended architecture

- Rules: standard chess via `python-chess`
- Learner: AlphaZero-style policy/value residual network
- Search: PUCT Monte Carlo Tree Search
- Actions: 73 spatial move planes × 64 origin squares (4,672 actions)
- State: eight positions of piece history plus rule-state planes
- Teacher: self-play only unless a future model card explicitly declares external distillation data

## Claims policy

A loss curve is not a strength claim. A v2 checkpoint may be described as trained only after it:

1. Passes legal-move, tactical, promotion, repetition, and draw-state suites.
2. Beats fixed random and material-greedy baselines with reported confidence intervals.
3. Has a reproducible run ID, configuration snapshot, generation, replay provenance, and exact resume test.
4. Reports paired-color results against a calibrated external opponent before any Elo estimate is published.

## Known migration boundary

The v1 4,096-action/18-plane checkpoints are structurally incompatible with the v2 policy and state encodings. They remain archived and must never be auto-loaded into v2.

## Hardware target

Initial v2 validation targets the RTX 2080 Ti 11 GB with eight CPU self-play workers. GPU utilization is not an optimization objective; accepted-model strength gained per unit wall time is the primary metric.
