# Chess-RL Project — Complete Brief (for Claude / new context)

Generated: 2026-08-24, from the live repo at `~/chess-rl` (Linux host "BraveAI", RTX 2080 Ti 11 GB, Ryzen 7 3700X 8C/16T).

---

## 1. What this project is

An AlphaZero-style chess learner: a residual policy/value neural network (CNN ResNet) trained purely by **self-play + PUCT Monte Carlo Tree Search**, with no chess knowledge beyond the rules. Codebase is Python (PyTorch) with a **C++ native engine** (pybind11) that accelerates self-play MCTS and inference batching, plus an arena gating loop, compressed replay, resumable checkpoints, a training controller API/dashboard, and a local `/move` model API.

Repo: `https://github.com/mrscorchedsilk/chess-rl.git` — Local path: `~/chess-rl`. Companion repos: `~/light-chess` (browser chess UI that calls the model API), `~/chess-training-dashboard` (training dashboard frontend). Venv: `~/chess-rl/.venv` (Python 3.11, torch 2.12.0+cu130, PEP 668).

**Honesty policy (established, non-negotiable):** a loss curve is not a strength claim. No Elo may be published until a checkpoint beats fixed baselines and calibrated opponents with confidence intervals. Tests verify code; primary sources verify research.

---

## 2. History — everything built over time (in order)

### Phase 0 — v1 prototype (before 2026-08-22)
- First AlphaZero prototype: 18 input planes, 4,096-action `from→to` policy (`from_square × to_square`).
- **Known defect**: promotion encoding collapses queen/rook/bishop/knight promotions onto one index; state omits repetition/halfmove history. Structurally incompatible with v2.
- Stopped at iteration 56; archived with SHA-256 checksums and git bundles at `backups/2026-08-22-iter56-prototype/` (latest.pt, best.pt, metrics, config, 3 git bundles: chess-rl baseline, dashboard baseline, light-chess baseline). The v1 `best.pt` was verified to equal the deterministic random init — **not a trained model**.
- Commit: `ef7e25a baseline: archive AlphaZero chess prototype before Sprint A-D rewrite` (Aug 22 13:32).

### Phase 1 — ChessNet v2 "Sprint A–D" rewrite (2026-08-22)
- Full clean rewrite to the AlphaZero 73-plane / 4,672-action policy map and 104-plane 8-history encoder. Commit `5f345a3 [verified] implement ChessNet v2 Sprint A-D` (Aug 22 14:41).
- Sprint A (reliability/observability): atomic checkpoints, exact resume, bounded worker failures, run IDs, dashboard truthfulness, loopback controller default.
- Sprint B (chess correctness): 4,672 action map with unique promotions, legal-move round-trip, probability mass preservation, batch/scalar encoder equivalence, claim-draw agreement between search and self-play.
- Sprint C (defensible learning): real epochs over bounded shuffled samples, separate loss logging, self-play uses the **accepted** model only, arena without root noise + paired colors + draws=0.5, reproducible baselines.
- Sprint D (performance/integration): 8 workers default, compressed/sparse replay, `/move` API + hot reload, Light Chess integration, dashboard without console errors.
- Acceptance contract lives in `SPRINT_A_D_ACCEPTANCE.md`. Migration rationale in `MIGRATION_V1_TO_V2.md`, v2 claims policy in `MODEL_CARD_V2.md`.

### Phase 2 — Native C++ foundation (2026-08-22, evening)
Pinned native toolchain + differential validation against python-chess, in small gated commits (all on `main` then branched):
- `b5cacb2` pinned native chess extension toolchain (C++17 required, pybind11, vendored `Disservin/chess-library` pinned at commit 53e6a84 with PROVENANCE.json).
- `9b80dda` machine-readable acceptance gate runner (`scripts/gate_runner.py`).
- `026c7f8` native position adapter + verified perft; `8c86795` promoted perft to a required acceptance gate.
- `449ec9b` native history + draw semantics preserved; `4d1e379` history semantics test in clean wheel.
- `c2def89` 100k native legal differential harness (`benchmarks/native_legal_differential.py`); `774b793` EP semantics aligned.
- `8b6d388` native policy map + history encoder (73-plane map in C++, `native/policy.{h,cpp}`).

### Phase 3 — Schema-v3 checkpoints + AMP + architecture versioning (T8/T9, Aug 22 21:09)
- `8295dc0` — checkpoint format schema-v3: `architecture_id` stamped in every checkpoint; loader validates body shape before `load_state_dict` and rejects cross-body loads (`IncompatibleCheckpointError`). Registered architectures: v2-6x128 (default, 2.17M params), v3-10x128 (3.35M), v3-10x192 (7.24M, preferred next), v3-10x256 (12.6M). See `MODEL_CARD_V3.md`.
- AMP: mixed precision (autocast + GradScaler, FP32 master weights, scaler state checkpointed/restored). Pinned prefetch loader stages minibatches async H2D.

### Phase 4 — Native MCTS core (T5) → multi-game Actor (T6/T7) → train-loop integration (T10) (Aug 22 21:36–21:58)
- `b591dac` native C++ MCTS core with pinned two-phase sparse API: `gather_leaves(max_batch)` → batched leaves (encoded planes + CSR legal moves) → `apply_evaluations(tokens, legal_offsets, legal_logits, values)`. SoA node pool, no per-node heap allocation, single mutable Position per search, root-only Dirichlet noise, virtual loss.
- `57d7c1e` native multi-game Actor (one C++ object owning N games, each with its own MCTS/Position/RNG; merged leaf batches; temperature sampling; z-value finalization mirroring selfplay.py) + persistent GPU inference runtime.
- `5d28e63` wired native actor + GPU runtime into `train.py` as `--selfplay-backend native` (replaces the 8-process Python worker pool).
- `47ad927` capped actor merged batch at total budget; canary script + native backend wired into the controller (start/stop API).
- `f744929` threaded native Actor (per-game `std::thread`) + checkpoint cadence 20.
- Benchmarks: Python-process baseline measured ~232 games/hour (4 games, 487 examples, `benchmarks/results/parallel-python-baseline.json`). Perf/throughput profiling files under `benchmarks/`.

### Phase 5 — Self-play diversity fixes (the big bug) (Aug 23)
- **Discovered failure mode**: self-play had a fixed-seed bug producing tens of thousands of replay examples collapsing onto a handful of repeated trajectories. `scripts/audit_replay.py` built to diagnose (read-only BLAKE2 diversity audit of schema-v3 checkpoints).
- `4863dd6 fix(selfplay): per-iteration seeds + paired-opening arena + warm start + replay audit` (Aug 23 21:35):
  - per-iteration round seeds (SplitMix64-style, stable across resume),
  - **paired-opening arena**: arena games start from `arena_games // 2` distinct deterministic openings (seed 424242, 8 plies), each played twice with colors swapped; temperature 0, no root noise,
  - warm-start support,
  - replay audit.
- `0f7132a` `--checkpoint-dir` CLI flag (rule-4-compliant rollout — never touch the live checkpoint dir during experiments); `9c67751` controller warm-start + checkpoint-dir start actions; `2206c17` checkpoint telemetry follows active run's checkpoint_dir.
- Validation runs: `checkpoints/v2-fixed/` (run `ac67ffe7595f`, iters 1–519, gen 3 — acceptances at iter 40 @0.575, 300 @0.575, 340 @0.55); `checkpoints/v2-canary/` (30-min production canary, 13 iterations, `scripts/canary_train.py`); `scripts/diversity_canary.py` + `resume_canary_check.py` proved stop/save/resume with run-id continuity.
- `bad22bf chore: snapshot pre-optimization state` (Aug 23 23:10) — tag `snapshot-pre-optimization-20260823` on branch `fix/native-selfplay-diversity`.

### Phase 6 — Current: native-arena optimization (dsh team, in progress, NOT merged)
Branch `optimize/native-arena` lives in a separate worktree `/home/iamsilverboss/dsh-team/work` (DeepSeek Harness team), 6 commits ahead of the snapshot. Planning docs: `docs/native-arena-design.md`, `docs/telemetry-design.md`, `docs/tickets.md`, `docs/milestone-review.md`. Tickets (implemented in order, each revertible):
- **Ticket A** `415a01b` — permanent phase/resource/diversity telemetry (`telemetry.py`, PhaseTimer, swallow-guarded, <1% overhead).
- **Ticket B** `b772b86` — `native_arena.py` adapter behind `arena_backend` flag (default `"python"`; python path untouched).
- **Ticket C** `be01c26` — native-arena correctness tests (determinism, paired openings/color swap, result contract, python-MCTS parity on fixed seed, terminal adjudication).
- **Ticket D** `58724a2` — `benchmarks/arena_bench.py` native-vs-python arena benchmark. Python arena baseline: ~104–105 s per 20-game/40-sim suite (telemetry jsonl; tickets cite median ~125 s, range 112–141 s). **Target: native arena ≤ 60 s (≥2× speedup)**.
- `ed10e32` — gitignore arena benchmark telemetry artifacts.
- Also `95d17e5` docs: Pro planning artifacts.
- Red lines from tickets (do NOT touch): checkpoint frequency, batch 512/1024 defaults, `games_per_iteration` without benchmark, actor sharding, training+inference overlap, simulation count bumps for GPU utilization, v2 architecture.

**Not yet merged into the working branch `fix/native-selfplay-diversity`** — the running trainer does NOT use the native arena yet.

---

## 3. Architecture

### 3.1 Data flow (native backend — the one that's running)

```
8 CPU self-play games (native C++ Actor, threaded per game)
        │  gather_leaves() → merged sparse leaf batch (planes + CSR legal moves)
        ▼
GPU InferenceRuntime (persistent, batch-bucketed 32/64/128/256)
        │  FP16 autocast body+policy; FP32 value head; on-GPU CSR legal-logit gather
        │  returns ONLY legal_logits[K] + values[B,1] — dense 4672-rows never leave GPU
        ▼
Actor.apply_evaluations() → backprop, virtual loss, Dirichlet root noise
        │  advance() → temp-sampled moves (temp=1 first 30 plies, else 0)
        ▼
finished games → (state, pi, z) examples → compressed replay buffer (50k cap)
        ▼
GPU training: 3 epochs/iteration over bounded sample, batch 256, AMP GradScaler
        ▼
arena gate every 20 iters: challenger vs accepted best, 20 games, 10 paired
openings × color-swap, 40 sims, temp 0, no noise; accept if score ≥ 0.55
        ├─ accepted → generation++, weights become the new teacher, milestone saved
        └─ rejected → candidate/optimizer reset to accepted best
checkpoint every 20 iters → schema-v3 latest.pt (atomic write) + checkpoint_meta.json
```

(Alternative Python path: `parallel.py` — 8 CPU worker processes + shared batched inference server + bounded IPC; kept as default until native proven — now superseded by native in the live run.)

### 3.2 Neural network (ChessNet, `model.py`)
- ResNet encoder: conv-in (3×3, BN, ReLU) + `num_res_blocks` residual blocks (two 3×3 convs each).
- **Policy head**: true spatial 1×1 conv over the board grid, 73 output channels, flattened in NHWC order → flat logit index = `from_square * 73 + plane` (4,672 actions).
- **Value head**: 1×1 conv → 32 ch → ReLU → flatten → linear → ReLU → linear → tanh ∈ [−1, 1] (side-to-move estimate).
- Registered sizes (verified by `tests/test_model_v3.py`):

| architecture_id | res blocks | filters | params | FP32 size |
|---|---:|---:|---:|---:|
| v2-6x128 (default) | 6 | 128 | 2,170,218 | 8.68 MiB |
| v3-10x128 | 10 | 128 | 3,352,938 | 13.41 MiB |
| v3-10x192 (preferred next) | 10 | 192 | 7,241,194 | 28.96 MiB |
| v3-10x256 | 10 | 256 | 12,604,010 | 50.42 MiB |

- v3 bodies also drop redundant conv biases (`remove_conv_bias`); v2 bodies never silently mutated.

### 3.3 State encoding (`encoding.py`)
- 104 planes × 8×8, White orientation, `rank*8+file` order: 8-position history × 12 piece planes + 8 meta planes (side to move, 4 castling, en-passant, normalized halfmove, repetition).
- Batch and scalar encoders must be equivalent (tested).

### 3.4 Action encoding (73-plane AlphaZero map, 4,672 actions)
- 56 queen-like direction/distance planes + 8 knight planes + 9 underpromotion planes.
- Every legal move round-trips uniquely **including all four promotions** (queen promo rides the queen-like plane; board-aware inversion in `Position::index_to_move`).

### 3.5 MCTS
- Python (`mcts.py`): batched PUCT, virtual loss 3.0, root Dirichlet noise (α=0.3, ε=0.25), temperature 1.0 first 30 plies, c_puct 1.25, 100 sims self-play / 40 sims arena.
- Native (`native/mcts.cpp` + `node_pool.cpp`): semantics-for-semantics equivalent; single mutable Position, SoA node pool (no heap per node), two-phase sparse batched API, exact same selection formula `-W/N + c_puct·P·sqrt(N_parent)/(1+N)`.

### 3.6 GPU inference runtime (`gpu_runtime.py`)
- Persistent runtime: `channels_last` model, eval mode, FP32 master, gradients off, `torch.compile(reduce-overhead)` with clean eager fallback.
- Fixed batch buckets (32/64/128/256) with preallocated pinned host + device buffers; padded rows explicitly masked.
- FP16 autocast body+policy, FP32 value head + CSR gather; non-blocking H2D/D2H on copy streams fenced by CUDA events.
- **Sparse policy output**: only legal-action logits are returned (CSR gather on device) — dense rows never leave GPU. `MAX_LEGAL_PER_ROW = 256`.
- Determinism: cudnn deterministic on for runtime lifetime. NOT thread-safe.

### 3.7 Training loop (`train.py`, 1682 lines)
- Entry points: `run()` (serial), `run_parallel()` (worker pool), `run_native()` (native actor + GPU runtime). `build_parser()`/`main()` CLI: `--workers`, `--resume`, `--checkpoint-dir`, `--selfplay-backend {python,native}`, `--num-simulations`, `--games-per-iteration`, `--num-iterations`, `--arena-every`.
- Checkpoint machinery: `_save_latest` (schema-v3, atomic temp+rename), `_save_best_atomic`, `_publish_best`, `_save_meta`, `_save_milestone` (accepted champions), `_archive_best/_archive_latest`, `_validate_checkpoint_compat` (rejects legacy/cross-body), `_load_latest_v2`, `_load_warm_start`, `_restore_rng`, `_reconcile_best`.
- Metrics: policy loss, value loss, entropy, optimizer steps, replay size, games, round seed logged to `training.jsonl`; arena events separately (`_log_arena_event`).

### 3.8 Arena gating
- Every `arena_every` iterations (currently 20): challenger vs accepted best, `arena_games=20` (10 paired openings × color swap), 40 sims, temp 0, no root noise, draw = 0.5, accept if score ≥ 0.55 (`_arena_gate`).
- Accepted → generation++ + milestone; rejected → candidate/optimizer reset to accepted best.

### 3.9 Replay buffer (`replay.py`)
- Compressed states, sparse policy targets at rest, 50,000 cap, pinned prefetch loader for async H2D minibatch staging.

### 3.10 Checkpoint format (schema-v3)
`latest.pt` contains: schema version, architecture_id, candidate + accepted-best weights, optimizer state, iteration, generation, run ID, optimizer-step count, Python/NumPy/Torch/CUDA RNG states, compressed replay buffer, config snapshot. Sidecars: `checkpoint_meta.json` / `best_meta.json` for the dashboard. Atomic writes. Milestones dir holds weights-only champions (~8.7 MB each vs ~99 MB full snapshots).

### 3.11 Servers / UIs
- `train_server.py` (port **8792**, systemd user service `chess-train-server.service`, auto-restart): training controller — dashboard HTML + `/api/status`, `/api/history`, `/api/control start|stop|warm-start` (with checkpoint-dir). NOTE: systemd unit sets `CHESS_TRAIN_HOST=0.0.0.0` → **listening on all interfaces** (README default is loopback; exposure is deliberate but should sit behind auth/tailnet).
- `serve.py` (port **8790**, loopback default): visual self-play viewer (`viewer.html`) + `POST /move {fen, sims}` → `{move, value, top_moves, sims, time_ms, model{source,generation}}`, `POST /control reload|new|pause|resume`; hot-reloads `best.pt` between games.
- `~/chess-training-dashboard` — dashboard frontend (contract-tested by `test_dashboard.js`).
- `~/light-chess` — browser chess UI that can set the AI side and sim budget via the `/move` API.

---

## 4. Program files (complete inventory)

### Top-level Python (`~/chess-rl`)
| File | Lines | Purpose |
|---|---:|---|
| `config.py` | 107 | All hyperparams + architecture registry (`ARCHITECTURES`), checkpoint dir, arena/telemetry knobs |
| `encoding.py` | 224 | 104-plane state encoder + 4,672-action move map |
| `model.py` | 210 | ChessNet ResNet + spatial policy head + value head; architecture-id resolution/inference |
| `mcts.py` | 278 | Python batched PUCT search, root noise, draw claims |
| `selfplay.py` | 93 | Rules-only game generation, training examples |
| `arena.py` | 176 | Deterministic paired-opening challenger evaluation |
| `replay.py` | 394 | Compressed replay storage + sparse policy targets |
| `parallel.py` | 474 | 8-process CPU self-play pool + shared inference server (python backend) |
| `train.py` | 1682 | Self-play, training, gating, metrics, resume, CLI (main) |
| `native_selfplay.py` | 134 | Python↔C++ bridge: drives native Actor with GPU runtime |
| `gpu_runtime.py` | 457 | Persistent batched GPU inference, sparse CSR legal-logit gather |
| `evaluate.py` | 484 | Reproducible random/material/network baselines + tactics suite |
| `serve.py` | 743 | Self-play viewer + `/move` model API (:8790) |
| `train_server.py` | 704 | Training controller + dashboard API (:8792) |
| `bench_mcts.py` / `bench_parallel.py` / `profile_mcts.py` | — | Benchmarks/profiling |
| `smoke_test.py`, `test_parallel_train.py`, `test_checkpoint_helpers.py`, `monitor_run.py` | — | Integration/smoke tests |
| `viewer.html` | — | Live network self-play viewer |
| `CMakeLists.txt` | — | Native C++ build (core lib + perft/history test binaries + pybind11 module) |
| `pyproject.toml`, `requirements.txt` | — | Packaging/deps |

### Native C++ (`native/`)
| File | Lines | Purpose |
|---|---:|---:|
| `position.h/.cpp` | 66/459 | Board wrapper (vendored Disservin/chess-library), FEN/history, legal moves, 104-plane encoder, action-map index↔UCI (board-aware), perft |
| `policy.h/.cpp` | 37/156 | 73-plane action-map constants, move↔index mapping (board-free + exact impl) |
| `mcts.h/.cpp` | 122/386 | PUCT MCTS core, sparse two-phase gather/apply API, node pool, Dirichlet, virtual loss |
| `node_pool.h/.cpp` | 73/45 | SoA compact node pool |
| `actor.h/.cpp` | 150/370 | Multi-game self-play Actor (threaded), example capture, z finalization |
| `chess_rl_native.cpp` | 302 | pybind11 module exposing Position, MCTS, Actor, perft, policy maps, build_info |
| `tests/native_perft_test.cpp`, `native_history_test.cpp` | — | C++ acceptance tests |
| `third_party/chess-library/` | — | Pinned chess library (commit 53e6a84, PROVENANCE.json + SHA-256) |

### Scripts (`scripts/`)
| File | Purpose |
|---|---|
| `gate_runner.py` | Machine-readable acceptance gate runner (ordered gates, evidence, fail-closed, no shell pipelines) |
| `native_foundation_gate.py` | Clean native build/perft acceptance gate |
| `canary_train.py` | 30-min production canary for native trainer (health gates, resume check) |
| `diversity_canary.py` | Controlled diversity canary: warm-starts new lineage from champion into separate checkpoint dir |
| `resume_canary_check.py` | Proves stop/save/resume (run-id continuity + iteration advance) |
| `audit_replay.py` | Read-only replay diversity audit (BLAKE2; exposes fixed-seed collapse) |
| `backfill_milestones.py` | Extracts accepted champions from full snapshots into weights-only milestones |
| `compare_champions.py` | Head-to-head comparison of accepted champions with corrected paired-opening arena |
| `prune_checkpoints.py` | Retention-policy pruning (dry-run default: active files, milestones, last-N, safety ladder, arena-neighbor) |
| `dashboard_smoke_test.py` | API contract + derived-metrics verification of :8792 |

### Tests (`tests/`, 452 collected)
Core/search (`test_core_v2`, `test_search_v2`, `test_review_regressions_v2`, `test_bench_v2`), model (`test_model_v3`), replay (`test_replay_precision_v2`, `test_pinned_replay_loader`, `test_replay_audit`), parallel (`test_parallel_pipeline_diagnostics`), training (`test_training_v2`, `test_train_loop_v2`, `test_selfplay_seed`, `test_warm_start`, `test_native_train_e2e`), controller/HTTP security (`test_controller_v2`, `test_http_security_v2`, `test_move_http_v2`, `test_serve_main_v2`, `test_observability`), arena (`test_arena_openings`, `test_milestones`), native (`test_native_import`, `test_native_perft`, `test_native_history`, `test_native_policy_encoding`, `test_native_legal_differential`, `test_native_mcts`, `test_native_actor`), evaluation (`test_evaluation_v2`), gates (`test_gate_runner`).

### Benchmarks (`benchmarks/`)
`parallel_pipeline.py`, `train_step.py`, `native_legal_differential.py` (100k differential), `native_mcts.py`, `native_policy_encoder_parity.py`, `gpu_runtime.py`, `_profile_transfer.py` + `results/` (python baseline: ~232 games/hr, 4 games/487 examples, cuda 13.0, torch 2.12.0+cu130, RTX 2080 Ti 10,818 MiB).

### Docs / evidence
`README.md`, `MIGRATION_V1_TO_V2.md`, `MODEL_CARD_V2.md`, `MODEL_CARD_V3.md`, `SPRINT_A_D_ACCEPTANCE.md`, `LICENSE` (MIT). dsh worktree docs: `native-arena-design.md`, `telemetry-design.md`, `tickets.md`, `milestone-review.md`.

### Checkpoint data (sizes)
| Dir | Contents | Size |
|---|---|---:|
| `checkpoints/v2/` | **Live production lineage** (run 8e11b89e667d): latest.pt, best.pt, iter-200→2300 snapshots, `milestones/` (gen1–4 champions) | 2.1 GB |
| `checkpoints/v2-fixed/` | Post-diversity-fix validation lineage (run ac67ffe7595f, iter 500, gen 3) | 2.2 GB |
| `checkpoints/v2-canary/` | 30-min canary (13 iterations) | 608 MB |
| `checkpoints/` | v1 legacy checkpoints (incompatible, never auto-load) | 360 MB |
| `backups/` | v1 prototype archive, pre-reset states, rollback-20260823, canary-precheck | 614 MB |

---

## 5. Live state right now (2026-08-24 ~17:30 IST)

- **Service**: `chess-train-server.service` active ~19 h, auto-restart, CPU ~17.5 h, RSS 2.5 G (peak 4.5 G). Dashboard at `http://127.0.0.1:8792` (bound 0.0.0.0).
- **Running trainer** (child of the service):
  `.venv/bin/python train.py --selfplay-backend native --resume --num-simulations 100 --games-per-iteration 20 --num-iterations 100000 --arena-every 20`
- **Run**: `8e11b89e667d` — resumed Aug 24 16:51; currently **iter 2301, generation 4, optimizer steps 9, replay 50,000/50,000 (full), 46,020 self-play games** since run start (Aug 22 23:19). Latest losses: policy 2.18, value 0.22, entropy 1.83. Native module import verified; 452 tests collected.
- **Arena history (live run, 19 recorded events, iter 1940→2300)**: rejections at 1940/1960 (0.25), 1980 (0.50); **acceptances: iter 2000 gen1, 2040 gen2, 2260 gen3, 2280 gen4 — all score 0.75 (10W 10D 0L)**, then rejects at 2060–2240 and latest iter 2300 **rejected 0.475 (3W 13D 4L)**. Champions: `checkpoints/v2/milestones/best-gen0001-iter2000 … gen0004-iter2280`.
- Earlier fixed-lineage run `ac67ffe7595f` (v2-fixed): 519 iters, gen 3, acceptances at 40/300/340.
- `best.pt` = gen 4 champion (8.7 MB weights-only, saved iter 2301). `latest.pt` = 103 MB full resumable snapshot.
- **Branch state**: working tree clean on `fix/native-selfplay-diversity` @ `bad22bf` (tag `snapshot-pre-optimization-20260823`). `optimize/native-arena` (telemetry + native arena, Tickets A–D) lives in worktree `/home/iamsilverboss/dsh-team/work` — **not merged**; running trainer still uses Python arena.

---

## 6. How to run / verify (canonical commands)

```bash
cd ~/chess-rl
# Full Python test suite (CPU)
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests -q
# Native import + perft/history gates
.venv/bin/python -m pytest tests/test_native_*.py -q
# Fresh run (only after tests pass and checkpoints/v2 intentionally emptied)
.venv/bin/python train.py --workers 8
# Resume / live-style run
.venv/bin/python train.py --selfplay-backend native --resume --arena-every 20
# Controller + dashboard (systemd: chess-train-server.service; :8792)
.venv/bin/python train_server.py
# Model API + viewer (:8790)
.venv/bin/python serve.py
# Evaluation baselines
CUDA_VISIBLE_DEVICES='' .venv/bin/python evaluate.py --games 20 --seed 42
# Rebuild native (IMPORTANT: must be Release — Debug is ~40x slower perft)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
.venv/bin/pip install --force-reinstall dist/chess_rl_native-0.1.0-*.whl
# External gates
node ~/light-chess/test.js && node ~/chess-training-dashboard/test_dashboard.js
```

---

## 7. Known issues / open items / conventions

1. **Native-arena optimization unmerged** — Tickets A–D complete on `optimize/native-arena`; needs review/merge. Target ≥2× arena speedup (python arena ~104–125 s per 20-game suite; native target ≤60 s). Red lines in tickets must be respected.
2. **Dashboard exposure**: `CHESS_TRAIN_HOST=0.0.0.0` in systemd unit — control API reachable beyond loopback; fine behind tailnet/auth, otherwise tighten.
3. **v1 ↔ v2 incompatibility** is intentional and enforced; never copy v1 checkpoints into v2 dirs.
4. **No trained production model yet** — gen 4 champion beats its own generations but has no published Elo; fixed-baseline eval with CIs is the gate.
5. **Checkpoint hygiene**: use `--checkpoint-dir` for experiments (canary rule #4); prune with `scripts/prune_checkpoints.py` (dry-run default); keep milestones (they're the only weights-only champions after pruning).
6. Conventions: loss curves ≠ strength; tests before claims; primary sources verify research; commits atomic + separately revertible; every behavior change behind a flag with python path as default until correctness gates pass.
7. GPU is the RTX 2080 Ti 11 GB — v3-10x192 is the preferred next body if it fits the VRAM/throughput gates; throughput (strength per wall-time) is the metric, not GPU utilization.
