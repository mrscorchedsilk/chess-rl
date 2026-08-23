"""TDD tests for Phase 4: read-only replay-diversity audit utility.

Written BEFORE the implementation.  The audit tool
(scripts/audit_replay.py) must, given a schema-v3 checkpoint, return
machine-readable JSON covering replay diversity without ever mutating the
checkpoint.  Tests use small synthetic replay buffers.

Run:  .venv/bin/python -m pytest tests/test_replay_audit.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from replay import ReplayBuffer  # noqa: E402

# The audit script lives in scripts/ and is importable via a small shim.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import audit_replay  # noqa: E402

POLICY_SIZE = 4672
PLANES, BOARD = 104, 8


def _state(game_start, tag=0):
    """A (104,8,8) binary state.  ``game_start`` zeroes the 7 future-history
    steps (planes 12:96), which is exactly the encoder's signature of ply 0."""
    state = np.zeros((PLANES, BOARD, BOARD), dtype=np.float32)
    # place a marker unique per tag in the current-position plane
    r, c = (tag // BOARD) % BOARD, tag % BOARD
    state[0, r, c] = 1.0
    if not game_start:
        state[12, 0, 0] = 1.0       # history step 1 non-zero -> not a start
    state[96].fill(1.0)             # side-to-move plane
    return state


def _example(game_start, tag):
    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    pi[int(tag) % POLICY_SIZE] = 1.0
    z = float((tag % 3) - 1)
    return (_state(game_start, tag), pi, z)


def _save_checkpoint(path, examples, games_per_iteration=12,
                     iteration=5, generation=2, run_id="run-test"):
    buf = ReplayBuffer(capacity=max(1, len(examples)) * 4,
                       policy_size=POLICY_SIZE, num_input_planes=PLANES,
                       board_size=BOARD)
    for e in examples:
        buf.add(*e)
    payload = {
        "schema_version": 2,
        "checkpoint_format": "schema-v3",
        "architecture_id": "v2-6x128",
        "run_id": run_id,
        "iteration": iteration,
        "generation": generation,
        "policy_size": POLICY_SIZE,
        "num_input_planes": PLANES,
        "board_size": BOARD,
        "config": {"games_per_iteration": games_per_iteration},
        "replay": buf.state_dict(),
        "best": {},  # audit must not need model tensors
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


# --------------------------------------------------------------------------- #
#  1. all unique examples                                                     #
# --------------------------------------------------------------------------- #

def test_audit_all_unique_examples(tmp_path):
    n = 20
    examples = [_example(game_start=False, tag=i) for i in range(n)]
    path = _save_checkpoint(str(tmp_path / "latest.pt"), examples,
                            games_per_iteration=4)
    r = audit_replay.audit_replay(path)
    assert r["replay_example_count"] == n
    assert r["unique_packed_states"] == n
    assert r["unique_exact_examples"] == n
    assert r["unique_state_fraction"] == pytest.approx(1.0)
    assert r["unique_example_fraction"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#  2. duplicated examples                                                     #
# --------------------------------------------------------------------------- #

def test_audit_duplicated_examples(tmp_path):
    ex = _example(game_start=False, tag=7)
    examples = [ex] * 10
    path = _save_checkpoint(str(tmp_path / "latest.pt"), examples,
                            games_per_iteration=4)
    r = audit_replay.audit_replay(path)
    assert r["replay_example_count"] == 10
    assert r["unique_packed_states"] == 1
    assert r["unique_exact_examples"] == 1
    assert r["unique_state_fraction"] == pytest.approx(0.1)
    assert r["unique_example_fraction"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
#  3. repeated full-game blocks                                               #
# --------------------------------------------------------------------------- #

def _make_game(tag, plies=3):
    return [_example(game_start=True, tag=tag)] + \
           [_example(game_start=False, tag=tag + j + 1) for j in range(plies - 1)]


def test_audit_repeated_full_game_blocks(tmp_path):
    games_per_iteration = 2
    # one 2-game block, repeated 5 times
    block = _make_game(100) + _make_game(200)
    examples = []
    for _ in range(5):
        examples.extend(block)
    path = _save_checkpoint(str(tmp_path / "latest.pt"), examples,
                            games_per_iteration=games_per_iteration)
    r = audit_replay.audit_replay(path)
    assert r["replay_example_count"] == 5 * len(block)
    assert r["inferred_game_start_count"] == 5 * games_per_iteration
    # two distinct trajectories (game 100 and game 200), each repeated 5x
    assert r["unique_full_game_trajectory_hashes"] == 2
    assert r["most_repeated_trajectory_count"] == 5
    # one distinct 2-game block, repeated 5x
    assert r["most_repeated_12_game_block_count"] is not None
    # games_per_iteration=2 so the field is for 2-game blocks
    assert r["unique_12_game_blocks"] == 1
    assert r["most_repeated_12_game_block_count"] == 5


# --------------------------------------------------------------------------- #
#  4. empty replay                                                            #
# --------------------------------------------------------------------------- #

def test_audit_empty_replay(tmp_path):
    path = _save_checkpoint(str(tmp_path / "latest.pt"), [],
                            games_per_iteration=12)
    r = audit_replay.audit_replay(path)
    assert r["replay_example_count"] == 0
    assert r["unique_packed_states"] == 0
    assert r["unique_exact_examples"] == 0
    assert r["inferred_game_start_count"] == 0
    assert r["inferred_complete_game_count"] == 0
    assert r["unique_full_game_trajectory_hashes"] == 0
    assert r["most_repeated_trajectory_count"] == 0


# --------------------------------------------------------------------------- #
#  5. partially cut ring-buffer boundary                                      #
# --------------------------------------------------------------------------- #

def test_audit_ring_buffer_boundary(tmp_path):
    # The buffer begins mid-game (a non-start tail), then a fresh game starts.
    examples = [_example(game_start=False, tag=1),
                _example(game_start=False, tag=2)]      # partial tail
    examples += _make_game(300)                         # one complete game
    path = _save_checkpoint(str(tmp_path / "latest.pt"), examples,
                            games_per_iteration=4)
    r = audit_replay.audit_replay(path)
    # Only the fresh game's start is detectable; the leading tail is not.
    assert r["inferred_game_start_count"] == 1
    assert r["replay_example_count"] == len(examples)


# --------------------------------------------------------------------------- #
#  6. read-only: source checkpoint never modified                             #
# --------------------------------------------------------------------------- #

def test_audit_is_read_only(tmp_path):
    import hashlib
    examples = _make_game(1) + _make_game(2)
    path = _save_checkpoint(str(tmp_path / "latest.pt"), examples,
                            games_per_iteration=2)
    with open(path, "rb") as f:
        before = hashlib.sha256(f.read()).hexdigest()
    audit_replay.audit_replay(path)
    with open(path, "rb") as f:
        after = hashlib.sha256(f.read()).hexdigest()
    assert before == after, "audit must not mutate the checkpoint"


# --------------------------------------------------------------------------- #
#  7. exact examples include policy and z                                     #
# --------------------------------------------------------------------------- #

def test_audit_exact_example_hash_includes_policy_and_z(tmp_path):
    base = _example(game_start=False, tag=5)
    state, pi, z = base
    # same state, different policy -> different exact example
    other_pi = np.zeros_like(pi); other_pi[999] = 1.0
    examples = [base, (state, other_pi, z)]
    path = _save_checkpoint(str(tmp_path / "latest.pt"), examples,
                            games_per_iteration=2)
    r = audit_replay.audit_replay(path)
    assert r["unique_packed_states"] == 1          # same state
    assert r["unique_exact_examples"] == 2         # different policy -> distinct
