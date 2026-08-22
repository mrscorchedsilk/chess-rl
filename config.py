"""Configuration for the AlphaZero-style chess learner (option 1: CNN ResNet + MCTS).

Defaults are sized to be runnable and *observable* on an 11 GB GPU
(RTX 2080 Ti), not to reach superhuman strength. See README for scaling notes.
"""
import os
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    # ---- model (ResNet encoder + policy/value heads) ----
    board_size = 8
    num_input_planes = 18
    num_res_blocks = 6          # AlphaZero used 20; 6 keeps training fast/observable
    num_filters = 128           # AlphaZero used 256
    policy_size = 4096          # 64 x 64 from->to

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
    epochs_per_iteration = 3
    games_per_iteration = 20
    replay_buffer_size = 200_000
    num_iterations = 200

    # ---- arena acceptance gating (new net vs current best) ----
    arena_every = 10
    arena_simulations = 40
    arena_games = 20
    arena_accept_threshold = 0.55

    # ---- misc ----
    max_game_length = 400       # plies hard cap -> draw
    checkpoint_interval_minutes = 10   # save a periodic snapshot at least this often
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    seed = 42

    # ---- parallel self-play (see parallel.py) ----
    selfplay_workers = 8        # 0/1 -> serial loop; >=2 -> process pool (8 = sweet spot on 16 cores)
    inference_max_batch = 4096  # server coalescing cap (positions per forward)
    inference_min_batch = 256   # below this the server waits a beat for more workers

    device = get_device()

    def __repr__(self):
        return "\n".join(f"{k} = {v}" for k, v in vars(self).items())
