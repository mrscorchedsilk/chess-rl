"""Replay compression must preserve non-binary rule-state planes."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fractional_state_values_survive_replay_roundtrip():
    from replay import ReplayBuffer

    state = np.zeros((104, 8, 8), dtype=np.float32)
    state[0, 0, 0] = 1.0
    state[102, :, :] = np.float32(0.37)  # normalized halfmove-clock plane
    policy = np.zeros(4672, dtype=np.float32)
    policy[[0, 77]] = [0.25, 0.75]

    buf = ReplayBuffer(4, 4672, 104, 8)
    buf.add(state, policy, -1.0)
    restored_state, restored_policy, restored_z = buf.sample_indices([0])

    assert np.array_equal(restored_state[0].numpy(), state)
    assert np.array_equal(restored_policy[0].numpy(), policy)
    assert float(restored_z[0, 0]) == -1.0

    snapshot = buf.state_dict()
    assert "state_extra_idx" in snapshot
    assert "state_extra_values" in snapshot
    assert snapshot["positions"].nbytes < state.nbytes / 4
