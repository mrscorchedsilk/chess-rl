# Chess-RL v2 — AlphaZero-style self-play chess learner

A rules-only chess learner built around a residual policy/value network, PUCT Monte Carlo Tree Search, self-play, accepted-model arena gating, compressed replay, resumable checkpoints, fixed evaluation baselines, and live browser tooling.

> **Current status:** v2 implementation is validated but no production v2 model has been trained. The archived v1 experiment is preserved under `backups/2026-08-22-iter56-prototype/` and is structurally incompatible with v2.

## Architecture

```text
8 CPU self-play workers
        │  encoded positions + legal actions
        ▼
shared batched GPU inference server
        │  sparse legal-action policy + value
        ▼
compressed replay buffer
        │
        ▼
GPU policy/value optimization
        │
        ▼
deterministic arena: challenger vs accepted best
        │
        ├─ accepted → generation increments
        └─ rejected → candidate/optimizer reset to accepted best
```

### Neural representation

- **State:** 104 × 8 × 8 planes
  - eight positions of 12 piece planes
  - side to move
  - four castling planes
  - en-passant plane
  - normalized halfmove plane
  - repetition plane
- **Policy:** 73 spatial action planes × 64 origin squares = **4,672 actions**
  - 56 queen-like direction/distance planes
  - 8 knight planes
  - 9 underpromotion planes
- **Policy head:** spatial 1×1 convolution to 73 planes, flattened in square-major order
- **Value head:** scalar side-to-move estimate in `[-1, 1]`

The v2 action map uniquely represents queen, rook, bishop, and knight promotions. It does not share the v1 promotion-collision defect.

## Files

| File | Purpose |
|---|---|
| `config.py` | V2 model, search, training, arena, checkpoint and worker settings |
| `encoding.py` | 104-plane state encoder and 4,672-action move mapping |
| `model.py` | Residual policy/value network with spatial policy head |
| `mcts.py` | Batched PUCT search, root-noise control and claim-draw handling |
| `selfplay.py` | Rules-only game generation and training examples |
| `arena.py` | Deterministic paired-color challenger evaluation |
| `replay.py` | Compressed state and sparse policy replay storage |
| `parallel.py` | CPU workers, bounded IPC and shared inference server |
| `train.py` | V2 self-play, training, gating, metrics and exact resume |
| `evaluate.py` | Reproducible random/material/network baselines and tactics suite |
| `train_server.py` | Local training controller and truthful telemetry API on `:8792` |
| `serve.py` | Self-play viewer and local `/move` model API on `:8790` |
| `viewer.html` | Live network self-play viewer |
| `tests/` | Core, search, replay, parallel, training, controller and evaluation tests |

## Checkpoint safety

V1 files remain in `checkpoints/`. V2 uses:

```text
checkpoints/v2/
```

A v2 `latest.pt` contains:

- schema version
- candidate and accepted-best weights
- optimizer state
- iteration and generation
- run ID and optimizer-step count
- Python, NumPy, Torch and CUDA RNG states
- compressed replay buffer
- configuration snapshot

Every durable save also writes a lightweight `checkpoint_meta.json` for the dashboard. Writes use a temporary file plus atomic replacement. Legacy checkpoints are rejected rather than guessed or partially loaded.

## Verification

```bash
cd ~/chess-rl

# Complete v2 Python suite
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests -q

# Core end-to-end smoke
CUDA_VISIBLE_DEVICES='' .venv/bin/python smoke_test.py

# Tiny parallel train/checkpoint integration
CUDA_VISIBLE_DEVICES='' .venv/bin/python test_parallel_train.py

# Light Chess rules + API integration
node ~/light-chess/test.js

# Dashboard static contract
node ~/chess-training-dashboard/test_dashboard.js
```

## Start a fresh v2 run

Do this only after all verification commands pass and `checkpoints/v2/` is intentionally empty:

```bash
cd ~/chess-rl
.venv/bin/python train.py --workers 8
```

Resume an existing v2 run:

```bash
.venv/bin/python train.py --workers 8 --resume
```

Eight workers are the validated default for the Ryzen 7 3700X host. More workers increase RAM and process contention and are not assumed to improve learning throughput.

## Training dashboard

```bash
cd ~/chess-rl
.venv/bin/python train_server.py
```

Open `http://127.0.0.1:8792/`.

The controller binds to loopback by default. Explicit remote binding is available through `CHESS_TRAIN_HOST`, but exposes training controls and should only be used behind a trusted tunnel or authentication layer.

The dashboard distinguishes live from saved iteration and shows run ID, generation, separate losses, replay size, optimizer steps, arena events, checkpoint state, resources, and stale/error status.

## Model API and Light Chess

```bash
cd ~/chess-rl
.venv/bin/python serve.py
```

The local API accepts:

```http
POST /move
Content-Type: application/json

{"fen":"<six-field FEN>","sims":100}
```

It returns a legal UCI move, value estimate, top search moves, simulation count, model source and generation. Light Chess can select the AI side and simulation budget while preserving two-human mode.

## Evaluation

```bash
cd ~/chess-rl
CUDA_VISIBLE_DEVICES='' .venv/bin/python evaluate.py --games 20 --seed 42
```

Evaluation uses deterministic random, one-ply material-greedy and network+MCTS players, paired colors, standard draw scoring, Wilson intervals and tactical competence positions. A loss curve alone is never treated as a playing-strength claim.

## Migration and model claims

See:

- `MIGRATION_V1_TO_V2.md`
- `MODEL_CARD_V2.md`
- `SPRINT_A_D_ACCEPTANCE.md`

Do not publish a v2 Elo until the checkpoint has reproducible fixed-baseline and calibrated-opponent results with confidence intervals.
