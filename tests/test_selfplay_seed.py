"""TDD tests for Phase 1: per-iteration self-play round seeding (native backend).

These tests are written BEFORE the implementation. They pin the required
behaviour:

  A. pure seed stability across calls (same iteration -> same seed)
  B. adjacent iterations produce different seeds
  C. seed fits an unsigned 64-bit integer (native Actor uint64)
  D. same model + same round seed -> identical finished-game hashes
  E. same model + adjacent iteration seeds -> different game trajectories
  F. resume consistency: seed depends only on (base_seed, iteration), not on
     optimizer_steps / global RNG state
  G. constructing NativeSelfPlay with an explicit seed leaves cfg.seed unchanged

Run:  .venv/bin/python -m pytest tests/test_selfplay_seed.py -q
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from config import Config
import native_selfplay


def fake_evaluator(inputs, legal_offsets, legal_indices):
    """Deterministic hash-based logits, zero values (mirrors test_native_actor.py)."""
    inputs = np.asarray(inputs, dtype=np.float32)
    offsets = np.asarray(legal_offsets, dtype=np.int32)
    indices = np.asarray(legal_indices, dtype=np.int32)
    logits = np.zeros(indices.shape[0], dtype=np.float32)
    for i in range(inputs.shape[0]):
        h = int.from_bytes(
            hashlib.sha256(inputs[i].tobytes()).digest()[:8], "little"
        )
        s, e = int(offsets[i]), int(offsets[i + 1])
        for k in range(s, e):
            logits[k] = float((h >> ((k - s) % 32)) & 0x1F)
    return logits, np.zeros(inputs.shape[0], dtype=np.float32)


def small_cfg():
    cfg = Config()
    cfg.seed = 42
    cfg.num_simulations = 8
    cfg.games_per_iteration = 4
    cfg.max_game_length = 40
    cfg.temperature = 1.0
    cfg.temperature_threshold = 30
    return cfg


def round_hash(examples):
    """Stable digest of a finished self-play round's example trajectories."""
    h = hashlib.blake2b(digest_size=16)
    for state, pi, z in examples:
        h.update(np.asarray(state, dtype=np.float32).tobytes())
        h.update(np.asarray(pi, dtype=np.float32).tobytes())
        h.update(np.float32(z).tobytes())
    return h.hexdigest()


def run_round(seed):
    cfg = small_cfg()
    sp = native_selfplay.NativeSelfPlay(
        cfg, fake_evaluator, games=cfg.games_per_iteration,
        weight_version=0, generation=0, seed=seed,
    )
    return round_hash(sp.run())


# --------------------------------------------------------------------------- #
#  A. pure seed stability                                                     #
# --------------------------------------------------------------------------- #

def test_selfplay_seed_is_stable_for_same_iteration():
    assert native_selfplay.derive_selfplay_seed(42, 2260) == \
        native_selfplay.derive_selfplay_seed(42, 2260)
    assert native_selfplay.derive_selfplay_seed(0, 0) == \
        native_selfplay.derive_selfplay_seed(0, 0)
    assert native_selfplay.derive_selfplay_seed(2 ** 63 - 1, 999) == \
        native_selfplay.derive_selfplay_seed(2 ** 63 - 1, 999)


# --------------------------------------------------------------------------- #
#  B. adjacent iteration diversity                                            #
# --------------------------------------------------------------------------- #

def test_selfplay_seed_changes_between_iterations():
    base = 42
    for n in range(1, 200):
        assert native_selfplay.derive_selfplay_seed(base, n) != \
            native_selfplay.derive_selfplay_seed(base, n + 1), \
            f"iterations {n} and {n + 1} reused the same seed"


# --------------------------------------------------------------------------- #
#  C. unsigned 64-bit range                                                   #
# --------------------------------------------------------------------------- #

def test_selfplay_seed_fits_native_uint64():
    for base, it in [(42, 2260), (0, 0), (2 ** 64 - 1, 2 ** 32),
                     (2 ** 63 - 1, 999), (123456789, 987654321)]:
        s = native_selfplay.derive_selfplay_seed(base, it)
        assert isinstance(s, int)
        assert 0 <= s < 2 ** 64, f"seed {s} outside uint64 range"


# --------------------------------------------------------------------------- #
#  D. native reproducibility: same round seed -> identical games              #
# --------------------------------------------------------------------------- #

def test_native_same_round_seed_identical_trajectories():
    seed = native_selfplay.derive_selfplay_seed(42, 2260)
    assert run_round(seed) == run_round(seed)


# --------------------------------------------------------------------------- #
#  E. native round diversity: adjacent iteration seeds -> different games     #
# --------------------------------------------------------------------------- #

def test_native_adjacent_iteration_seeds_differ():
    s2260 = native_selfplay.derive_selfplay_seed(42, 2260)
    s2261 = native_selfplay.derive_selfplay_seed(42, 2261)
    assert s2260 != s2261
    assert run_round(s2260) != run_round(s2261)


# --------------------------------------------------------------------------- #
#  F. resume consistency: pure function of (base_seed, iteration)             #
# --------------------------------------------------------------------------- #

def test_selfplay_seed_resume_consistency():
    # Uninterrupted run reaching iteration N and a resumed run starting at the
    # same completed-iteration boundary N must derive the SAME seed. Because
    # the derivation is a pure function of (base_seed, iteration), perturbing
    # global RNG state (which resume restores differently) must not change it.
    import random
    base = 42
    expected = native_selfplay.derive_selfplay_seed(base, 128)
    random.seed(12345)
    np.random.seed(54321)
    assert native_selfplay.derive_selfplay_seed(base, 128) == expected
    # distinct iterations remain distinct even after RNG perturbation
    assert native_selfplay.derive_selfplay_seed(base, 128) != \
        native_selfplay.derive_selfplay_seed(base, 129)


# --------------------------------------------------------------------------- #
#  G. no cfg mutation                                                         #
# --------------------------------------------------------------------------- #

def test_native_selfplay_explicit_seed_does_not_mutate_cfg():
    cfg = small_cfg()
    before = cfg.seed
    sp = native_selfplay.NativeSelfPlay(
        cfg, fake_evaluator, games=2, weight_version=0, generation=0,
        seed=123456789,
    )
    assert cfg.seed == before, "NativeSelfPlay mutated cfg.seed"
    # the constructed actor must actually run to completion (sanity)
    assert len(sp.run()) > 0
