"""Persistent worker pool: identical results regardless of pool size, and a
pool that is sized to the machine rather than to the game count.

Before this change `Actor::parallel_for` constructed and joined a fresh
std::vector<std::thread> on EVERY gather_leaves and EVERY advance — thousands
of thread creations per game — and `num_threads` defaulted to the game count,
so 20 concurrent games oversubscribed a 16-thread CPU.

The invariant that matters is that the pool is an execution detail: for a
given seed the finished games must be byte-identical whether the work runs on
one worker or sixteen.  The merged CSR ordering is established serially by
the caller, so nothing about the output may depend on scheduling.

Run:  .venv/bin/python -m pytest tests/test_native_thread_pool.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess_rl_native  # noqa: E402

from test_native_actor import (  # noqa: E402
    drive, fake_evaluator, new_actor, serialize,
)


HW = os.cpu_count() or 1


# --------------------------------------------------------------------------- #
#  pool sizing                                                                #
# --------------------------------------------------------------------------- #

def test_pool_never_exceeds_hardware_threads():
    """Asking for more workers than the CPU has must clamp, not oversubscribe."""
    actor = new_actor(games=HW * 4, num_simulations=1, max_game_length=2,
                      seed=1, num_threads=HW * 4)
    assert actor.num_threads <= HW


def test_pool_never_exceeds_game_count():
    """Workers beyond the game count would idle forever."""
    actor = new_actor(games=3, num_simulations=1, max_game_length=2,
                      seed=1, num_threads=64)
    assert actor.num_threads == 3


def test_pool_defaults_are_clamped_too():
    """The default (num_threads omitted) used to be the raw game count."""
    actor = new_actor(games=HW * 2, num_simulations=1, max_game_length=2,
                      seed=1)
    assert 1 <= actor.num_threads <= HW


def test_single_thread_request_is_honoured():
    actor = new_actor(games=8, num_simulations=1, max_game_length=2,
                      seed=1, num_threads=1)
    assert actor.num_threads == 1


# --------------------------------------------------------------------------- #
#  the invariant: pool size must not change results                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("threads", [1, 2, 3, 8, 16])
def test_results_identical_across_pool_sizes(threads):
    """Same seed, different worker counts -> byte-identical finished games."""
    ref = new_actor(games=6, num_simulations=8, max_game_length=12,
                    seed=20260824, num_threads=1)
    ref.set_teacher(3, 1)
    drive(ref)
    expected = serialize(ref.finished_games())

    actor = new_actor(games=6, num_simulations=8, max_game_length=12,
                      seed=20260824, num_threads=threads)
    actor.set_teacher(3, 1)
    drive(actor)
    assert serialize(actor.finished_games()) == expected


def test_repeated_runs_on_the_same_pool_are_stable():
    """A reused pool must not carry state between actors."""
    outs = []
    for _ in range(3):
        actor = new_actor(games=4, num_simulations=6, max_game_length=10,
                          seed=777, num_threads=HW)
        actor.set_teacher(0, 0)
        drive(actor)
        outs.append(serialize(actor.finished_games()))
    assert outs[0] == outs[1] == outs[2]


def test_merged_gather_csr_is_ordered_by_game_regardless_of_threads():
    """Leaf blocks stay contiguous and in game order; the merge is serial."""
    def first_gather(threads):
        actor = new_actor(games=8, num_simulations=16, max_game_length=40,
                          seed=99, num_threads=threads)
        actor.set_teacher(0, 0)
        tokens, inputs, offsets, indices = actor.gather_leaves(256)
        return (list(tokens), np.asarray(inputs).tobytes(),
                np.asarray(offsets).tolist(), np.asarray(indices).tolist())

    base = first_gather(1)
    for threads in (2, 5, HW):
        assert first_gather(threads) == base


def test_offsets_remain_monotonic_and_terminated():
    actor = new_actor(games=8, num_simulations=16, max_game_length=40,
                      seed=1234, num_threads=HW)
    actor.set_teacher(0, 0)
    tokens, inputs, offsets, indices = actor.gather_leaves(256)
    offsets = np.asarray(offsets)
    assert offsets[0] == 0
    assert offsets[-1] == len(indices)
    assert np.all(np.diff(offsets) >= 0)
    assert len(offsets) == len(tokens) + 1


# --------------------------------------------------------------------------- #
#  lifetime                                                                   #
# --------------------------------------------------------------------------- #

def test_actor_destruction_joins_workers():
    """Creating and dropping many actors must not leak or hang."""
    for i in range(25):
        actor = new_actor(games=4, num_simulations=2, max_game_length=4,
                          seed=i, num_threads=HW)
        actor.set_teacher(0, 0)
        drive(actor)
        del actor


def test_many_gathers_reuse_one_pool():
    """The whole point: thousands of parallel_for calls, one pool."""
    actor = new_actor(games=8, num_simulations=32, max_game_length=60,
                      seed=4242, num_threads=HW)
    actor.set_teacher(0, 0)
    rounds = 0
    while not actor.is_done() and rounds < 5000:
        tokens, inputs, offsets, indices = actor.gather_leaves(128)
        if len(tokens) > 0:
            logits, values = fake_evaluator(inputs, offsets, indices)
            actor.apply_evaluations(tokens, offsets, logits, values)
        actor.advance()
        rounds += 1
    assert actor.is_done()
    assert rounds > 50, "expected many gather rounds to exercise pool reuse"
