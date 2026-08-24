"""Concurrency must ENLARGE the GPU batch, not thin it.

`Actor::gather_leaves` used to split a fixed total budget equally across
in-play games (`per_game = max_batch / in_play`), and `InferenceRuntime`
rejected anything above 256.  Together those meant that raising the number of
concurrent games made each game's slice SMALLER while the batch stayed pinned
near 240 — so the GPU never saw a bigger forward pass no matter how much
self-play concurrency was configured.

With a fixed per-game leaf target the merged batch is
``games_in_flight * leaves_per_game`` (still capped by the total budget), so
concurrency and batch size scale together.

Run:  .venv/bin/python -m pytest tests/test_batch_scaling.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess_rl_native as native  # noqa: E402
from config import Config         # noqa: E402
import native_selfplay            # noqa: E402


def _null_eval(inputs, offsets, indices):
    return (np.zeros(len(indices), dtype=np.float32),
            np.zeros((len(offsets) - 1, 1), dtype=np.float32))


def _actor(games, sims=64, seed=7):
    a = native.Actor(games=games, c_puct=1.25, virtual_loss=3.0,
                     num_simulations=sims, temperature=1.0,
                     temperature_threshold=30, max_game_length=200,
                     seed=seed, num_threads=min(games, os.cpu_count() or 1))
    a.set_teacher(0, 0)
    return a


def steady_batch(games, max_batch, leaves_per_game, rounds=6):
    """Largest merged batch over the first few post-root gather rounds.

    The very first gather always yields exactly one leaf per game (root
    expansion), so a single gather says nothing about steady-state batching.
    """
    actor = _actor(games)
    sizes = []
    for _ in range(rounds):
        tokens, inputs, offsets, indices = actor.gather_leaves(
            max_batch, leaves_per_game)
        if len(tokens) == 0:
            actor.advance()
            continue
        sizes.append(len(tokens))
        logits, values = _null_eval(inputs, offsets, indices)
        actor.apply_evaluations(tokens, offsets, logits, values)
        actor.advance()
    return max(sizes) if sizes else 0


# --------------------------------------------------------------------------- #
#  the defect                                                                 #
# --------------------------------------------------------------------------- #

def test_legacy_equal_share_does_not_grow_the_batch():
    """Pins the OLD behaviour, still reachable with leaves_per_game=0."""
    small = steady_batch(24, max_batch=256, leaves_per_game=0)
    large = steady_batch(96, max_batch=256, leaves_per_game=0)
    assert large <= small, (
        "legacy split should not grow the batch with more games; "
        f"got {small} -> {large}"
    )
    assert large <= 256


def test_fixed_leaf_target_scales_the_batch_with_concurrency():
    sizes = {g: steady_batch(g, max_batch=4096, leaves_per_game=12)
             for g in (24, 48, 96)}
    assert sizes[48] > sizes[24]
    assert sizes[96] > sizes[48]
    # linear in games, up to the per-round variation of finished searches
    assert sizes[96] >= 3 * sizes[24] * 0.8


@pytest.mark.parametrize("games,lpg", [(8, 4), (16, 8), (32, 12)])
def test_batch_equals_games_times_leaf_target(games, lpg):
    got = steady_batch(games, max_batch=4096, leaves_per_game=lpg)
    assert got == games * lpg


# --------------------------------------------------------------------------- #
#  the budget is still respected                                              #
# --------------------------------------------------------------------------- #

def test_merged_batch_never_exceeds_max_batch():
    """leaves_per_game is a target; the total budget is a hard cap."""
    got = steady_batch(64, max_batch=128, leaves_per_game=32)
    assert got <= 128


def test_leaves_per_game_is_clamped_by_the_equal_share():
    """64 games x 32 leaves would be 2048; a 256 budget clamps to 4 each."""
    got = steady_batch(64, max_batch=256, leaves_per_game=32)
    assert got == 64 * (256 // 64)


def test_negative_leaf_target_is_rejected():
    actor = _actor(4)
    with pytest.raises(ValueError):
        actor.gather_leaves(256, -1)


def test_zero_max_batch_is_rejected():
    actor = _actor(4)
    with pytest.raises(ValueError):
        actor.gather_leaves(0, 4)


# --------------------------------------------------------------------------- #
#  config decoupling                                                          #
# --------------------------------------------------------------------------- #

def test_games_in_flight_defaults_to_games_per_iteration():
    cfg = Config()
    cfg.selfplay_games_in_flight = None
    cfg.games_per_iteration = 20
    sp = native_selfplay.NativeSelfPlay(cfg, _null_eval, seed=1)
    assert sp.games == 20


def test_games_in_flight_is_independent_of_training_cadence():
    cfg = Config()
    cfg.games_per_iteration = 20
    cfg.selfplay_games_in_flight = 64
    sp = native_selfplay.NativeSelfPlay(cfg, _null_eval, seed=1)
    assert sp.games == 64
    assert cfg.games_per_iteration == 20


def test_explicit_games_argument_still_wins():
    cfg = Config()
    cfg.selfplay_games_in_flight = 64
    sp = native_selfplay.NativeSelfPlay(cfg, _null_eval, games=6, seed=1)
    assert sp.games == 6


def test_actor_threads_are_clamped_to_hardware():
    cfg = Config()
    cfg.selfplay_games_in_flight = (os.cpu_count() or 1) * 4
    sp = native_selfplay.NativeSelfPlay(cfg, _null_eval, seed=1)
    assert sp.actor.num_threads <= (os.cpu_count() or 1)


def test_config_defaults_are_coherent():
    cfg = Config()
    from gpu_runtime import BUCKETS
    assert cfg.selfplay_max_batch <= max(BUCKETS), (
        "selfplay_max_batch must fit the largest inference bucket"
    )
    assert cfg.selfplay_leaves_per_game > 0
    assert max(BUCKETS) >= 4096
