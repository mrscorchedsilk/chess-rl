"""Async CUDA path of PinnedReplayLoader.

Every pre-existing test for this class ran with ``device="cpu"``, which takes
the synchronous passthrough branch — the async producer/consumer path had no
coverage at all, and shipped with two deadlock/crash bugs:

  1. the consumer popped from ``_ready`` without notifying the producer, which
     parks on ``len(_ready) >= num_prefetch``.  With num_prefetch=1 the
     producer slept forever after the first batch and both threads deadlocked
     on batch 2;
  2. pinned buffers were allocated ``empty_like`` the FIRST slice, so the
     ragged final slice of an epoch (8192 % 256, or any sample_size that is
     not a whole multiple of batch_size) raised a shape mismatch in copy_.

The class was wired to nothing, so neither bug was ever reachable.  It is now
the training loop's staging path, so both are pinned here.

Run:  .venv/bin/python -m pytest tests/test_pinned_loader_async.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config                          # noqa: E402
from replay import PinnedReplayLoader, ReplayBuffer  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="async path requires CUDA"
)


def make_buffer(n, capacity=None, seed=0):
    cfg = Config()
    buf = ReplayBuffer(capacity or n, cfg.policy_size,
                       cfg.num_input_planes, cfg.board_size)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        state = (rng.random((cfg.num_input_planes, cfg.board_size,
                             cfg.board_size)) < 0.2).astype(np.float32)
        pi = np.zeros(cfg.policy_size, dtype=np.float32)
        idx = rng.choice(cfg.policy_size, size=6, replace=False)
        pi[idx] = 1.0 / 6
        rows.append((state, pi, float(rng.choice([-1.0, 0.0, 1.0]))))
    buf.extend(rows)
    return buf


# --------------------------------------------------------------------------- #
#  regression: the deadlock                                                   #
# --------------------------------------------------------------------------- #


def test_many_batches_with_prefetch_one_do_not_deadlock():
    """Producer parks once the queue fills; the consumer must wake it."""
    buf = make_buffer(2048)
    loader = PinnedReplayLoader(buf, batch_size=128, device="cuda",
                                num_prefetch=1)
    rows = np.arange(len(buf))
    got = sum(1 for _ in loader.batches(rows))
    assert got == int(np.ceil(len(rows) / 128))


def test_repeated_calls_reuse_the_loader_without_leaking_buffers():
    buf = make_buffer(1024)
    loader = PinnedReplayLoader(buf, batch_size=256, device="cuda",
                                num_prefetch=1)
    rows = np.arange(len(buf))
    for _ in range(4):
        assert sum(1 for _ in loader.batches(rows)) == 4
        assert len(loader._free) == loader.num_buffers
        assert len(loader._ready) == 0
    assert loader.pin_sets_allocated <= loader.num_buffers


# --------------------------------------------------------------------------- #
#  regression: the ragged tail                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n,batch", [(1000, 256), (8192 % 700 + 700, 256),
                                     (513, 128), (100, 128)])
def test_ragged_final_batch_is_staged_correctly(n, batch):
    buf = make_buffer(n)
    loader = PinnedReplayLoader(buf, batch_size=batch, device="cuda",
                                num_prefetch=1)
    rows = np.arange(len(buf))
    sizes = [s.shape[0] for s, _, _ in loader.batches(rows)]
    assert sum(sizes) == n
    assert sizes[:-1] == [batch] * (len(sizes) - 1)
    assert sizes[-1] == n - batch * (len(sizes) - 1)


# --------------------------------------------------------------------------- #
#  the data must be the same data                                             #
# --------------------------------------------------------------------------- #

def test_async_batches_match_synchronous_sample_indices():
    buf = make_buffer(600)
    rows = np.arange(len(buf))
    loader = PinnedReplayLoader(buf, batch_size=256, device="cuda",
                                num_prefetch=1)
    for i, (states, pis, zs) in enumerate(loader.batches(rows)):
        rb = rows[i * 256:(i + 1) * 256]
        exp_s, exp_p, exp_z = buf.sample_indices(rb, "cuda")
        assert torch.equal(states, exp_s)
        assert torch.equal(pis, exp_p)
        assert torch.equal(zs, exp_z)


def test_buffer_contents_are_not_corrupted_by_pinned_reuse():
    """Reused pinned buffers must not bleed batch i into batch i+1."""
    buf = make_buffer(1024)
    rows = np.arange(len(buf))
    loader = PinnedReplayLoader(buf, batch_size=128, device="cuda",
                                num_prefetch=1)
    collected = [s.clone() for s, _, _ in loader.batches(rows)]
    for i, states in enumerate(collected):
        exp, _, _ = buf.sample_indices(rows[i * 128:(i + 1) * 128], "cuda")
        assert torch.equal(states, exp), f"batch {i} corrupted"


def test_worker_errors_surface_to_the_caller():
    buf = make_buffer(256)
    loader = PinnedReplayLoader(buf, batch_size=64, device="cuda",
                                num_prefetch=1)

    class Boom(RuntimeError):
        pass

    def explode(*a, **k):
        raise Boom("staging failed")

    loader.buffer = type("B", (), {"sample_indices": staticmethod(explode)})()
    with pytest.raises(Boom):
        list(loader.batches(np.arange(256)))
