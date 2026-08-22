"""Task 8: pinned minibatch staging (PinnedReplayLoader).

Strict TDD for the pinned prefetch loader that stages replay minibatches on
pinned host memory for async H2D transfer.  CPU-only (pinned allocs are
CPU-side; no CUDA required — the loader is a synchronous passthrough on CPU).

Run: .venv/bin/python -m pytest tests/test_pinned_replay_loader.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay import PinnedReplayLoader, ReplayBuffer  # noqa: E402


def _make_buffer(n=32, capacity=64, policy_size=4672, planes=104, board=8, seed=0):
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(capacity, policy_size, planes, board)
    for _ in range(n):
        state = rng.random((planes, board, board), dtype=np.float32)
        pi = np.zeros(policy_size, dtype=np.float32)
        idx = rng.choice(policy_size, 16, replace=False)
        pi[idx] = 1.0
        pi /= pi.sum()
        buf.add(state, pi, float(rng.choice([-1.0, 0.0, 1.0])))
    return buf


def test_loader_reconstructs_dense_batches_matching_sampler():
    buf = _make_buffer(seed=1)
    loader = PinnedReplayLoader(buf, batch_size=8, device="cpu", num_prefetch=1)
    rows = np.arange(len(buf))
    # The loader is a generator over consecutive slices; the first slice must
    # exactly equal a direct dense reconstruction of the same rows.
    expected_states, expected_pis, expected_zs = buf.sample_indices(rows[:8])
    batches = list(loader.batches(rows))
    assert len(batches) == 4, f"32 rows / batch 8 = 4 slices, got {len(batches)}"
    got_states, got_pis, got_zs = batches[0]
    assert got_states.shape == expected_states.shape
    assert np.allclose(got_states.numpy(), expected_states.numpy())
    assert np.allclose(got_pis.numpy(), expected_pis.numpy())
    assert np.allclose(got_zs.numpy(), expected_zs.numpy())


def test_loader_batch_count_matches_slices():
    buf = _make_buffer(seed=2)
    loader = PinnedReplayLoader(buf, batch_size=8, device="cpu")
    list(loader.batches(np.arange(16)))
    assert loader.batch_count == 2


def test_loader_empty_rows_yields_nothing():
    buf = _make_buffer(n=0, seed=3)
    loader = PinnedReplayLoader(buf, batch_size=8, device="cpu")
    assert list(loader.batches(np.array([], dtype=np.int64))) == []


def test_loader_no_pinned_allocation_on_cpu():
    buf = _make_buffer(seed=4)
    loader = PinnedReplayLoader(buf, batch_size=8, device="cpu")
    list(loader.batches(np.arange(8)))
    assert loader.pin_sets_allocated == 0  # CPU passthrough never pins
    assert loader.prefetch_calls == 0      # no async worker on CPU
