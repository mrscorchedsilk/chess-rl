"""Sharded self-play: overlap without changing what is played.

The serial loop leaves the GPU idle during tree descent and every CPU thread
idle during the forward pass.  ShardedSelfPlay splits the games into
independent actors driven by separate Python threads, serialising only GPU
access.

The properties that must hold regardless of scheduling:

  * shards == 1 reproduces NativeSelfPlay byte-for-byte (the seed is used
    unchanged, not derived);
  * a sharded round is reproducible across runs, even though the threads
    finish in arbitrary order — examples are concatenated in shard order;
  * every requested game is played exactly once;
  * a failure inside one shard propagates to the caller rather than hanging.

CPU-only: a deterministic in-process evaluator stands in for the GPU.

Run:  .venv/bin/python -m pytest tests/test_sharded_selfplay.py -q
"""
import os
import sys
import threading

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config          # noqa: E402
import native_selfplay as ns       # noqa: E402


def _cfg(**over):
    cfg = Config()
    cfg.num_simulations = 16
    cfg.max_game_length = 12
    cfg.telemetry_enabled = False
    cfg.selfplay_leaves_per_game = 8
    cfg.selfplay_max_batch = 1024
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _evaluator(calls=None, lock=None):
    """Deterministic position-dependent logits; optionally records call order."""
    def evaluate(inputs, offsets, indices):
        arr = ns.expand_planes(inputs)
        base = (arr[:, 0, 0, :].sum(axis=1) * 7.0).astype(np.int64)
        logits = np.zeros(len(indices), dtype=np.float32)
        for i in range(arr.shape[0]):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if e > s:
                logits[s:e] = ((np.arange(s, e) + int(base[i])) % 13).astype(np.float32)
        if calls is not None:
            with lock:
                calls.append(arr.shape[0])
        return logits, np.zeros((arr.shape[0], 1), dtype=np.float32)
    return evaluate


def _key(examples):
    return [(e[0].tobytes(), e[1].tobytes(), float(e[2])) for e in examples]


# --------------------------------------------------------------------------- #
#  equivalence with the serial path                                           #
# --------------------------------------------------------------------------- #

def test_single_shard_is_byte_identical_to_native_selfplay():
    cfg = _cfg()
    serial = ns.NativeSelfPlay(cfg, _evaluator(), games=6, seed=1234).run()
    sharded = ns.ShardedSelfPlay(cfg, _evaluator(), games=6, shards=1,
                                 seed=1234).run()
    assert _key(serial) == _key(sharded)


def test_single_shard_uses_the_round_seed_unchanged():
    """A derived seed here would silently change the whole v2 lineage."""
    cfg = _cfg()
    sp = ns.ShardedSelfPlay(cfg, _evaluator(), games=4, shards=1, seed=777)
    assert sp.round_seed == 777


# --------------------------------------------------------------------------- #
#  determinism under concurrency                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shards", [2, 3, 4])
def test_sharded_rounds_are_reproducible(shards):
    cfg = _cfg()
    a = ns.ShardedSelfPlay(cfg, _evaluator(), games=8, shards=shards,
                           seed=99).run()
    b = ns.ShardedSelfPlay(cfg, _evaluator(), games=8, shards=shards,
                           seed=99).run()
    assert _key(a) == _key(b)


def test_thread_completion_order_does_not_leak_into_the_examples():
    """The real risk: examples ordered by which thread finished first.

    Injecting asymmetric delays into the evaluator forces a different
    completion order between the two runs.  If the merge were driven by
    completion rather than shard index, these would differ.
    """
    import time as _time

    def jittery(delays):
        state = {"n": 0}
        lock = threading.Lock()

        def evaluate(inputs, offsets, indices):
            with lock:
                i = state["n"]
                state["n"] += 1
            _time.sleep(delays[i % len(delays)])
            arr = ns.expand_planes(inputs)
            base = (arr[:, 0, 0, :].sum(axis=1) * 7.0).astype(np.int64)
            logits = np.zeros(len(indices), dtype=np.float32)
            for k in range(arr.shape[0]):
                s, e = int(offsets[k]), int(offsets[k + 1])
                if e > s:
                    logits[s:e] = (
                        (np.arange(s, e) + int(base[k])) % 13
                    ).astype(np.float32)
            return logits, np.zeros((arr.shape[0], 1), dtype=np.float32)
        return evaluate

    cfg = _cfg()
    a = ns.ShardedSelfPlay(cfg, jittery([0.0, 0.004]), games=8, shards=4,
                           seed=5).run()
    b = ns.ShardedSelfPlay(cfg, jittery([0.004, 0.0]), games=8, shards=4,
                           seed=5).run()
    assert _key(a) == _key(b)


# --------------------------------------------------------------------------- #
#  game accounting                                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("games,shards,expected", [
    (8, 4, [2, 2, 2, 2]),
    (10, 4, [3, 3, 2, 2]),
    (5, 2, [3, 2]),
    (3, 8, [1, 1, 1]),      # shards clamped to games
])
def test_games_are_split_across_shards_without_loss(games, shards, expected):
    cfg = _cfg()
    sp = ns.ShardedSelfPlay(cfg, _evaluator(), games=games, shards=shards,
                            seed=1)
    assert sp._shard_games == expected
    assert sum(sp._shard_games) == games
    assert sp.shards == len(expected)


def test_every_game_is_played_exactly_once():
    cfg = _cfg()
    sp = ns.ShardedSelfPlay(cfg, _evaluator(), games=9, shards=3, seed=2)
    sp.run()
    assert len(sp.trajectory_hashes) == 9


def test_actor_threads_are_divided_across_shards_not_duplicated():
    cfg = _cfg(selfplay_actor_threads=8)
    sp = ns.ShardedSelfPlay(cfg, _evaluator(), games=16, shards=4, seed=1)
    assert sum(a.num_threads for a in sp.actors) <= 8 + len(sp.actors)


# --------------------------------------------------------------------------- #
#  concurrency behaviour                                                      #
# --------------------------------------------------------------------------- #

def test_gpu_access_is_serialised():
    """The runtime is not thread-safe; only one shard may be inside it."""
    inside = []
    peak = [0]
    lock = threading.Lock()

    def evaluate(inputs, offsets, indices):
        with lock:
            inside.append(1)
            peak[0] = max(peak[0], len(inside))
        try:
            arr = ns.expand_planes(inputs)
            return (np.zeros(len(indices), dtype=np.float32),
                    np.zeros((arr.shape[0], 1), dtype=np.float32))
        finally:
            with lock:
                inside.pop()

    cfg = _cfg()
    ns.ShardedSelfPlay(cfg, evaluate, games=8, shards=4, seed=3).run()
    assert peak[0] == 1, f"{peak[0]} shards were inside the runtime at once"


def test_shard_failure_propagates_and_does_not_hang():
    class Boom(RuntimeError):
        pass

    def explode(inputs, offsets, indices):
        raise Boom("evaluator failed")

    cfg = _cfg()
    sp = ns.ShardedSelfPlay(cfg, explode, games=8, shards=4, seed=4)
    with pytest.raises(Boom):
        sp.run()


def test_telemetry_fields_are_populated():
    cfg = _cfg()
    sp = ns.ShardedSelfPlay(cfg, _evaluator(), games=8, shards=2, seed=6)
    sp.run()
    assert sp.round_duration_s > 0
    assert sp.inference_calls > 0
    assert sp.gather_calls > 0
    assert sp.gpu_busy_s > 0
    assert 0.0 <= sp.gpu_busy_fraction <= float(sp.shards)
    assert sp.batch_stats


def test_shards_are_clamped_to_at_least_one():
    cfg = _cfg()
    sp = ns.ShardedSelfPlay(cfg, _evaluator(), games=4, shards=0, seed=1)
    assert sp.shards == 1


def test_zero_games_is_rejected():
    cfg = _cfg()
    with pytest.raises(ValueError):
        ns.ShardedSelfPlay(cfg, _evaluator(), games=0, shards=2, seed=1)
