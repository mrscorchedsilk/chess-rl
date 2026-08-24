"""Replay capacity preflight.

The replay buffer is resident for the whole run AND embedded in every
resumable snapshot.  A capacity that does not fit therefore fails hours into
training, during a checkpoint write, leaving a partial snapshot and no obvious
cause.  The preflight turns that into an explicit startup error naming the
number to change.

Per-example costs are measured, not guessed (see
REPLAY_BYTES_PER_EXAMPLE_RAM / _DISK in replay.py).

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_replay_capacity.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config   # noqa: E402
import replay               # noqa: E402


GB = 10 ** 9


def _cfg(capacity, **over):
    cfg = Config()
    cfg.replay_buffer_size = capacity
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
#  estimates                                                                  #
# --------------------------------------------------------------------------- #

def test_estimate_scales_linearly_with_capacity():
    a = replay.estimate_replay_footprint(100_000)
    b = replay.estimate_replay_footprint(500_000)
    assert b["ram_bytes"] == 5 * a["ram_bytes"]
    assert b["checkpoint_bytes"] == 5 * a["checkpoint_bytes"]


def test_500k_projection_matches_the_measured_constants():
    est = replay.estimate_replay_footprint(500_000)
    assert 1.9 < est["ram_gb"] < 2.3
    assert 0.7 < est["checkpoint_gb"] < 1.0


# --------------------------------------------------------------------------- #
#  RAM budget                                                                 #
# --------------------------------------------------------------------------- #

def test_500k_passes_on_a_machine_with_room():
    cfg = _cfg(500_000)
    est = replay.preflight_replay_capacity(
        cfg, available_ram_bytes=32 * GB, free_disk_bytes=100 * GB)
    assert est["capacity"] == 500_000


def test_capacity_beyond_the_ram_budget_is_refused():
    cfg = _cfg(5_000_000)
    with pytest.raises(replay.ReplayCapacityError, match="resident"):
        replay.preflight_replay_capacity(
            cfg, available_ram_bytes=8 * GB, free_disk_bytes=1000 * GB)


def test_ram_error_names_a_capacity_that_would_fit():
    cfg = _cfg(5_000_000)
    with pytest.raises(replay.ReplayCapacityError) as ei:
        replay.preflight_replay_capacity(
            cfg, available_ram_bytes=8 * GB, free_disk_bytes=1000 * GB)
    msg = str(ei.value)
    assert "Lower replay_buffer_size to about" in msg
    suggested = int(msg.split("to about")[1].split(",")[0].strip()
                    + msg.split("to about")[1].split(",")[1][:3])
    assert suggested < 5_000_000


def test_budget_fraction_is_configurable():
    cfg = _cfg(1_000_000, replay_ram_budget_fraction=0.9)
    replay.preflight_replay_capacity(
        cfg, available_ram_bytes=8 * GB, free_disk_bytes=1000 * GB)
    cfg_tight = _cfg(1_000_000, replay_ram_budget_fraction=0.1)
    with pytest.raises(replay.ReplayCapacityError):
        replay.preflight_replay_capacity(
            cfg_tight, available_ram_bytes=8 * GB, free_disk_bytes=1000 * GB)


# --------------------------------------------------------------------------- #
#  disk budget                                                                #
# --------------------------------------------------------------------------- #

def test_insufficient_disk_for_snapshots_is_refused():
    cfg = _cfg(500_000, replay_preflight_snapshots=3)
    with pytest.raises(replay.ReplayCapacityError, match="checkpoint"):
        replay.preflight_replay_capacity(
            cfg, available_ram_bytes=64 * GB, free_disk_bytes=1 * GB)


def test_disk_error_reports_both_needed_and_free():
    cfg = _cfg(500_000)
    with pytest.raises(replay.ReplayCapacityError) as ei:
        replay.preflight_replay_capacity(
            cfg, available_ram_bytes=64 * GB, free_disk_bytes=1 * GB)
    msg = str(ei.value)
    assert "is free under" in msg
    assert "snapshots need" in msg


def test_snapshot_retention_scales_the_disk_requirement():
    cfg_one = _cfg(500_000, replay_preflight_snapshots=1)
    replay.preflight_replay_capacity(
        cfg_one, available_ram_bytes=64 * GB, free_disk_bytes=1 * GB)
    cfg_many = _cfg(500_000, replay_preflight_snapshots=10)
    with pytest.raises(replay.ReplayCapacityError):
        replay.preflight_replay_capacity(
            cfg_many, available_ram_bytes=64 * GB, free_disk_bytes=1 * GB)


# --------------------------------------------------------------------------- #
#  escape hatch                                                               #
# --------------------------------------------------------------------------- #

def test_preflight_can_be_disabled():
    cfg = _cfg(50_000_000, replay_preflight=False)
    est = replay.preflight_replay_capacity(
        cfg, available_ram_bytes=1 * GB, free_disk_bytes=1 * GB)
    assert est["capacity"] == 50_000_000


def test_unknown_ram_does_not_block_startup():
    """A machine that cannot report memory must not fail the run."""
    cfg = _cfg(500_000)
    est = replay.preflight_replay_capacity(
        cfg, available_ram_bytes=0, free_disk_bytes=100 * GB)
    assert est["capacity"] == 500_000


# --------------------------------------------------------------------------- #
#  the buffer really does hold what it says                                   #
# --------------------------------------------------------------------------- #

def test_large_capacity_buffer_evicts_at_capacity_not_earlier():
    cfg = Config()
    cap = 500
    buf = replay.ReplayBuffer(cap, cfg.policy_size, cfg.num_input_planes,
                              cfg.board_size)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(cap + 137):
        state = (rng.random((cfg.num_input_planes, 8, 8)) < 0.2).astype(np.float32)
        pi = np.zeros(cfg.policy_size, dtype=np.float32)
        pi[rng.integers(0, cfg.policy_size, 4)] = 0.25
        rows.append((state, pi, 0.0))
    buf.extend(rows)
    assert len(buf) == cap
