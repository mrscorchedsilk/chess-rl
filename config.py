"""Configuration for the AlphaZero-style chess learner (option 1: CNN ResNet + MCTS).

Defaults are sized to be runnable and *observable* on an 11 GB GPU
(RTX 2080 Ti), not to reach superhuman strength. See README for scaling notes.

Sprint B: the policy is now the AlphaZero 73-plane action map (4672 = 64 x 73
from_square x plane) and the board encoder is an 8-position history stack of
104 planes. Checkpoints from the old 4096 from->to prototype live in
``checkpoints/`` and are NOT loadable by this config; the default
``checkpoint_dir`` therefore points at the versioned ``checkpoints/v2/``
subdirectory so stale ``latest.pt`` snapshots can never be resumed by accident.
"""
import os
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
#  Architecture registry (Task 9)                                             #
# --------------------------------------------------------------------------- #
# Canonical body identities: (num_res_blocks, num_filters).  Checkpoints carry
# an ``architecture_id`` so tensors can NEVER be loaded into a different body.
# The v2 body keeps its legacy id; the v3 candidates are selectable but the
# DEFAULT stays v2-6x128 until benchmarks pick a winner (the final 10x192
# default switch is a later step, not this one).
ARCHITECTURES = {
    "v2-6x128": (6, 128),      # current default; 2,170,218 params
    "v3-10x128": (10, 128),    # 3,352,938 params
    "v3-10x192": (10, 192),    # 7,241,194 params
    "v3-10x256": (10, 256),    # 12,604,010 params
}


class Config:
    # ---- model (ResNet encoder + spatial policy/value heads) ----
    board_size = 8
    history_steps = 8                 # positions of board history stacked (current first)
    num_input_planes = 12 * history_steps + 8   # 104: 8 history x 12 piece planes + 8 meta
    num_res_blocks = 6                # AlphaZero used 20; 6 keeps training fast/observable
    num_filters = 128                 # AlphaZero used 256
    # Task 9: body identity.  Default is v2-6x128 (unchanged).  Selectable:
    #   v3-10x128 / v3-10x192 / v3-10x256.  Explicit num_res_blocks/num_filters
    # overrides (as used by the test suite) still win over this label; the
    # model derives a canonical id from the actual body in that case.
    architecture_id = "v2-6x128"
    # v3-only: drop the redundant conv biases (conv+BN makes them redundant).
    # Gated behind an explicit v3 architecture id — v2 bodies are NEVER
    # silently mutated.
    remove_conv_bias = False
    policy_planes = 73                # 56 queen-like (8 dirs x 7 dist) + 8 knight + 9 underpromo
    policy_size = policy_planes * 64  # 4672: flat index = from_square * 73 + plane

    # ---- MCTS ----
    num_simulations = 100       # self-play: MCTS rollouts per move
    c_puct = 1.25
    dirichlet_alpha = 0.3
    dirichlet_epsilon = 0.25
    temperature = 1.0
    temperature_threshold = 30  # first N plies use temperature > 0
    batch_size = 32             # MCTS leaf evaluations batched per forward pass
    virtual_loss = 3.0          # PUCT virtual loss for batched parallel search

    # ---- training ----
    learning_rate = 0.001
    weight_decay = 1e-4
    train_batch_size = 256       # samples per gradient step
    training_epochs = 3          # full passes over the bounded per-iteration sample
    epochs_per_iteration = 3     # backward-compatible alias
    # Positions drawn from the replay buffer per epoch.  Previously this was
    # only ever read via ``getattr(cfg, "train_epoch_size", 0)`` and was never
    # defined, so the bound silently collapsed to
    # ``train_batch_size * training_epochs`` == 768 -> 3 batches/epoch ->
    # 9 optimizer steps per iteration, i.e. roughly ONE gradient step per 260
    # freshly generated positions.  It is now explicit.
    #
    # 8192 with train_batch_size 256 and 3 epochs gives 32 batches/epoch and
    # 96 optimizer steps/iteration.  Against ~2.4k positions generated per
    # iteration that is a sample-reuse ratio near 10; `training_rate`
    # telemetry reports the realised ratio every iteration so it can be tuned
    # against overfitting to a stale buffer rather than guessed.
    train_epoch_size = 8192
    games_per_iteration = 20
    replay_buffer_size = 50_000
    num_iterations = 200
    amp = True                   # mixed precision (autocast + GradScaler) on CUDA

    # ---- arena acceptance gating (new net vs current best) ----
    arena_every = 10
    arena_simulations = 40
    arena_games = 20
    arena_accept_threshold = 0.55
    arena_root_noise = False    # arena matches use NO Dirichlet root exploration
    # Paired-opening diversity: arena games start from `arena_games // 2`
    # distinct deterministic openings (generated from `arena_seed`), each
    # played twice with colors swapped.  Independent of weights / global RNG /
    # self-play seed; bounded shallow depth keeps arena games non-trivial but
    # varied.  Temperature stays 0 and root noise stays off.
    arena_seed = 424242
    arena_opening_plies = 8
    # Search backend for the arena gate: "python" (arena.py + mcts.py,
    # unchanged default) or "native" (native_arena.py + chess_rl_native.MCTS;
    # see docs/native-arena-design.md).  Game semantics (paired openings,
    # color swap, temp 0, no root noise, score/threshold) are identical.
    arena_backend = "python"

    # ---- permanent phase telemetry (Ticket A; see docs/telemetry-design.md) ----
    # Swallow-guarded + semantic-free, so leaving it on is safe: no extra
    # forward passes, no RNG draws, no data mutation — replay examples, move
    # choices, checkpoints and scores are bit-identical with it on vs off.
    telemetry_enabled = True          # global on/off for the JSONL emitter
    telemetry_path = None             # None -> <checkpoint_dir>/telemetry.jsonl
    telemetry_resource_every = 1      # emit a `resource` record every N iterations
    telemetry_diversity_every = arena_every  # replay-buffer diversity audit cadence

    # ---- misc ----
    max_game_length = 400       # plies hard cap -> draw
    checkpoint_every_iterations = 20    # completed iterations between snapshots
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "checkpoints", "v2")   # versioned; legacy checkpoints/ untouched
    seed = 42

    # ---- parallel self-play (see parallel.py) ----
    selfplay_workers = 8        # 0/1 -> serial loop; >=2 -> process pool
    result_timeout_seconds = 60 # upper bound for a worker's game to land in the queue
    inference_max_batch = 4096  # server coalescing cap (positions per forward)
    inference_min_batch = 256   # below this the server waits a beat for more workers

    device = get_device()

    def __repr__(self):
        return "\n".join(f"{k} = {v}" for k, v in vars(self).items())
