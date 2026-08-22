# Chess-RL — AlphaZero-style self-play chess learner (Option 1: CNN ResNet + MCTS)

A from-scratch reinforcement-learning chess agent that is given **only the rules**
and learns to play entirely against itself — the AlphaZero paradigm. This is the
"option 1" architecture: a convolutional ResNet that outputs a **policy** (which
move to play) and a **value** (how good the position is), combined with Monte
Carlo Tree Search (MCTS).

Runs on a single NVIDIA GPU (built and smoke-tested on an RTX 2080 Ti / 11 GB).

## How it learns (the loop)

```
self-play games ──► (position, MCTS move-probs, result)  replay buffer
                          │
                          ▼
              train ResNet:  loss = (z − v)² − πᵀ log p + c·‖θ‖²
                          │
                          ▼
        arena: new net vs best net — accept if win-rate ≥ 55%
```

- **The rules are the only teacher.** `python-chess` supplies legal-move
  generation (`board.legal_moves`) — that is literally "what it can't do." No
  openings, no human games, no evaluation heuristics are fed in.
- **MCTS turns win/loss/draw into a per-move training signal.** The policy head
  proposes moves; the search improves on it; the improved distribution π becomes
  the target for the next round of self-play.

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (`Config`) + `get_device()` |
| `encoding.py` | Board → 18×8×8 planes; move ↔ flat 4096 index; legal-move mask |
| `model.py` | `ChessNet` — ResNet encoder + policy/value heads |
| `mcts.py` | `MCTS` — PUCT search with Dirichlet exploration |
| `selfplay.py` | `play_game()` — generate one self-play game |
| `arena.py` | `play_match()` — head-to-head evaluation |
| `train.py` | `run()` — the full training loop (CLI entry point) |
| `smoke_test.py` | Correctness checks (encoding, model, MCTS, self-play) |

## Quickstart

```bash
cd ~/chess-rl
# dependencies (torch is pre-installed here; see requirements.txt for a fresh env)
.venv/bin/python smoke_test.py     # verify everything works
.venv/bin/python train.py          # start training (defaults sized for 11 GB GPU)
```

Watch for: loss decreasing, and the arena reporting the new net beating the old
one as training progresses.

## Key hyperparameters (config.py)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_res_blocks` / `num_filters` | 6 / 128 | AlphaZero used 20 / 256 — bump when you have more GPU time |
| `num_simulations` | 100 | MCTS rollouts per move during self-play (the #1 speed/strength dial) |
| `c_puct` | 1.25 | exploration constant |
| `dirichlet_alpha` / `epsilon` | 0.3 / 0.25 | root move exploration noise |
| `games_per_iteration` | 20 | self-play games before each training burst |
| `arena_accept_threshold` | 0.55 | keep the new net only if it beats the best |

## Scaling up (2080 Ti → DGX Spark)

The current defaults are deliberately small so you can *watch* learning happen.
To get real strength, raise these in order of impact:

1. **Parallel / batched self-play** — MCTS currently evaluates one leaf at a
   time; running many games concurrently and batching their network calls is
   the single biggest speedup.
2. **More simulations + bigger net** — `num_simulations` → 400–800,
   `num_res_blocks`/`num_filters` → 15/192 or 20/256. The 128 GB unified-memory
   DGX Spark handles the full AlphaZero-size net comfortably.
3. **Swap the CNN for a Transformer encoder** (option 2) — drop-in replacement
   for the body; keep the same policy/value heads and MCTS.
4. **Accelerate with Stockfish (later)** — seed policy/value via distillation
   from Stockfish moves/evals, then keep self-playing. This is the fastest path
   to strength once the self-play loop is proven.

## Environment note

`torch` lives in the Hermes agent's own venv on this machine; the project venv
bridges it via `.venv/lib/python3.11/site-packages/_hermes_torch.pth` rather
than re-downloading the CUDA wheel. On a clean machine, install torch first
(see `requirements.txt`).
