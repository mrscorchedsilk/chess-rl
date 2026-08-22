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


class Config:
    # ---- model (ResNet encoder + spatial policy/value heads) ----
    board_size = 8
    history_steps = 8                 # positions of board history stacked (current first)
    num_input_planes = 12 * history_steps + 8   # 104: 8 history x 12 piece planes + 8 meta
    num_res_blocks = 6                # AlphaZero used 20; 6 keeps training fast/observable
    num_filters = 128                 # AlphaZero used 256
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
    games_per_iteration = 20
    replay_buffer_size = 50_000
    num_iterations = 200

    # ---- arena acceptance gating (new net vs current best) ----
    arena_every = 10
    arena_simulations = 40
    arena_games = 20
    arena_accept_threshold = 0.55
    arena_root_noise = False    # arena matches use NO Dirichlet root exploration

    # ---- misc ----
    max_game_length = 400       # plies hard cap -> draw
    checkpoint_every_iterations = 1    # completed iterations between snapshots
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
