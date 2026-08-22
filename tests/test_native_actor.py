"""Native multi-game self-play Actor tests (Task 6).

Exercises the C++ Actor (native/actor.h, native/actor.cpp) through the pinned
Python contract:

    actor = chess_rl_native.Actor(games=8, c_puct=1.25, virtual_loss=3.0,
                                  num_simulations=100, temperature=1.0,
                                  temperature_threshold=30, max_game_length=400,
                                  seed=42)
    actor.set_teacher(weight_version, generation)
    tokens, inputs, legal_offsets, legal_indices = actor.gather_leaves(max_batch)
    actor.apply_evaluations(tokens, legal_offsets, legal_logits, values)
    actor.advance()
    games = actor.finished_games()   # [{"generation", "weight_version",
                                     #   "termination", "examples": [...]]]
    actor.is_done()
    actor.games_remaining()

All tests use the same deterministic fake evaluator as test_native_mcts.py
(hash-based logits, zero values), so every run is reproducible.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

import chess_rl_native

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLANES, ROWS, COLS = 104, 8, 8
POLICY_SIZE = 4672


def fake_evaluator(inputs, legal_offsets, legal_indices):
    """Deterministic logits (hash of the encoded position), zero values.

    Identical pattern to tests/test_native_mcts.py::fake_evaluator.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    offsets = np.asarray(legal_offsets, dtype=np.int32)
    indices = np.asarray(legal_indices, dtype=np.int32)
    logits = np.zeros(indices.shape[0], dtype=np.float32)
    for i in range(inputs.shape[0]):
        row_hash = int.from_bytes(
            hashlib.sha256(inputs[i].tobytes()).digest()[:8], "little"
        )
        s, e = int(offsets[i]), int(offsets[i + 1])
        for k in range(s, e):
            logits[k] = float((row_hash >> ((k - s) % 32)) & 0x1F)
    values = np.zeros(inputs.shape[0], dtype=np.float32)
    return logits, values


def drive(actor, max_batch=32):
    """Drive gather/apply/advance until every game has finished."""
    while not actor.is_done():
        tokens, inputs, offsets, indices = actor.gather_leaves(max_batch)
        if len(tokens) > 0:
            logits, values = fake_evaluator(inputs, offsets, indices)
            actor.apply_evaluations(tokens, offsets, logits, values)
        actor.advance()


def new_actor(games, num_simulations, max_game_length, seed, **kw):
    kw.setdefault("temperature", 1.0)
    kw.setdefault("temperature_threshold", 30)
    return chess_rl_native.Actor(
        games=games, num_simulations=num_simulations,
        max_game_length=max_game_length, seed=seed, **kw,
    )


def serialize(games):
    """Flatten finished_games() into plain Python for exact comparison."""
    out = []
    for game in games:
        examples = [
            (state.tolist(), pi.tolist(), float(z))
            for state, pi, z in game["examples"]
        ]
        out.append((game["generation"], game["weight_version"],
                    game["termination"], examples))
    return out


# ---------------------------------------------------------------------------
# 1. Full run: gather/apply/advance until done; examples are well-formed
# ---------------------------------------------------------------------------


def test_full_run_produces_valid_examples():
    actor = new_actor(games=2, num_simulations=8, max_game_length=20, seed=42)
    actor.set_teacher(weight_version=3, generation=7)

    assert actor.games_remaining() == 2
    assert not actor.is_done()

    drive(actor)

    assert actor.is_done()
    assert actor.games_remaining() == 0

    games = actor.finished_games()
    assert len(games) == 2
    for game in games:
        assert game["generation"] == 7
        assert game["weight_version"] == 3
        assert isinstance(game["termination"], str) and game["termination"]
        assert len(game["examples"]) >= 1
        for state, pi, z in game["examples"]:
            assert state.shape == (PLANES, ROWS, COLS)
            assert state.dtype == np.float32
            assert pi.shape == (POLICY_SIZE,)
            assert pi.dtype == np.float32
            assert np.all(np.isfinite(state))
            assert np.all(pi >= 0.0)
            assert pi.sum() == pytest.approx(1.0, abs=1e-3)  # pi mass ~1.0
            assert z in (-1.0, 0.0, 1.0)

    # finished_games() drains: a second call returns nothing new.
    assert actor.finished_games() == []


# ---------------------------------------------------------------------------
# 2. Determinism: same seed -> identical finished games
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_finished_games():
    def run(seed):
        actor = new_actor(games=2, num_simulations=8, max_game_length=20, seed=seed)
        actor.set_teacher(weight_version=1, generation=2)
        drive(actor)
        return serialize(actor.finished_games())

    assert run(42) == run(42)
    assert run(1234) == run(1234)
    # Different seeds play different games (temperature=1.0 for all 20 plies
    # makes the sampled moves genuinely stochastic).
    assert run(42) != run(1234)


# ---------------------------------------------------------------------------
# 3. gather_leaves merges leaves across games into one batch
# ---------------------------------------------------------------------------


def test_gather_leaves_merges_across_games():
    actor = new_actor(games=4, num_simulations=8, max_game_length=20, seed=5)

    tokens, inputs, offsets, indices = actor.gather_leaves(32)
    B = len(tokens)

    # Every one of the 4 games contributes its root-expansion leaf.
    assert B > 1
    assert tokens == list(range(B))
    assert inputs.shape == (B, PLANES, ROWS, COLS)
    assert inputs.dtype == np.float32
    assert inputs.flags["C_CONTIGUOUS"]
    assert offsets.dtype == np.int32
    assert indices.dtype == np.int32
    assert offsets.shape == (B + 1,)
    assert offsets[0] == 0
    assert int(offsets[-1]) == len(indices)
    for i in range(B):
        row = indices[int(offsets[i]):int(offsets[i + 1])]
        assert len(row) >= 1
        assert np.all(np.diff(row) >= 0)  # sorted ascending within the row
        assert np.all(row >= 0) and np.all(row < POLICY_SIZE)

    # Route the merged batch back and confirm the search still progresses.
    logits, values = fake_evaluator(inputs, offsets, indices)
    actor.apply_evaluations(tokens, offsets, logits, values)
    while not actor.is_done():
        tokens, inputs, offsets, indices = actor.gather_leaves(32)
        if len(tokens) > 0:
            logits, values = fake_evaluator(inputs, offsets, indices)
            actor.apply_evaluations(tokens, offsets, logits, values)
        actor.advance()
    assert len(actor.finished_games()) == 4


# ---------------------------------------------------------------------------
# 4. apply_evaluations input validation
# ---------------------------------------------------------------------------


def test_apply_evaluations_rejects_wrong_sized_arrays():
    actor = new_actor(games=2, num_simulations=8, max_game_length=20, seed=7)
    tokens, inputs, offsets, indices = actor.gather_leaves(32)
    logits, values = fake_evaluator(inputs, offsets, indices)
    B, K = len(tokens), len(indices)

    with pytest.raises(ValueError):
        actor.apply_evaluations(tokens, offsets, logits, values[:-1])  # values too short
    with pytest.raises(ValueError):
        actor.apply_evaluations(tokens + [0], offsets, logits, values)  # tokens too long
    with pytest.raises(ValueError):
        actor.apply_evaluations(tokens, offsets[:-1], logits, values)  # offsets too short
    with pytest.raises(ValueError):
        actor.apply_evaluations(tokens, offsets, logits[:-1], values)  # logits too short
    with pytest.raises(ValueError):
        actor.apply_evaluations([B], offsets, logits, values)  # token out of range
    with pytest.raises(ValueError):
        actor.apply_evaluations([-1], offsets, logits, values)  # token out of range

    # Rejected calls leave the actor intact: the valid call still succeeds.
    assert len(logits) == K and len(values) == B
    actor.apply_evaluations(tokens, offsets, logits, values)
    actor.advance()


def test_apply_evaluations_without_gather_raises():
    actor = new_actor(games=2, num_simulations=8, max_game_length=20, seed=8)
    with pytest.raises(ValueError):
        actor.apply_evaluations(
            [],
            np.zeros(1, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# 5. games_remaining decreases monotonically and reaches 0 at is_done()
# ---------------------------------------------------------------------------


def test_games_remaining_decreases_monotonically_to_zero():
    actor = new_actor(games=3, num_simulations=6, max_game_length=12, seed=11)

    remaining = [actor.games_remaining()]
    assert remaining[-1] == 3
    assert not actor.is_done()

    while not actor.is_done():
        tokens, inputs, offsets, indices = actor.gather_leaves(16)
        if len(tokens) > 0:
            logits, values = fake_evaluator(inputs, offsets, indices)
            actor.apply_evaluations(tokens, offsets, logits, values)
        actor.advance()
        remaining.append(actor.games_remaining())
        assert remaining[-1] <= remaining[-2]

    assert remaining[-1] == 0
    assert len(actor.finished_games()) == 3
