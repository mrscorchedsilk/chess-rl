"""Native MCTS core tests (Task 5).

Exercises the C++ MCTS (native/mcts.h, native/mcts.cpp) through the pinned
Python contract:

    mcts = chess_rl_native.MCTS(c_puct=1.25, virtual_loss=3.0,
                                num_simulations=100, dirichlet_alpha=0.3,
                                dirichlet_epsilon=0.25, seed=42)
    mcts.set_root(start_fen, history_moves)
    tokens, inputs, legal_offsets, legal_indices = mcts.gather_leaves(max_batch)
    mcts.apply_evaluations(tokens, legal_offsets, legal_logits, values)
    done = mcts.is_complete()
    policy = mcts.policy(temperature)   # list[(uci, prob)] sorted by uci

All tests use a deterministic fake evaluator (hash-based logits, zero values)
and no root noise (dirichlet_epsilon=0), so every search is reproducible.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

import chess
import chess_rl_native

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# White Ra1 + Kg6 vs black Kh8: the ONLY legal mate is a1a8# (the rook covers
# g8 along rank 8, the king covers h7/g7). Verified against python-chess.
# (The FEN in the task brief, "7k/8/8/8/8/8/8/R6K", has NO mate in 1: after
# a1a8+ black escapes to h7/g7 — hence the king is placed on g6 here.)
MATE_IN_1_FEN = "7k/8/6K1/8/8/8/8/R7 w - - 0 1"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"

PLANES, ROWS, COLS = 104, 8, 8
POLICY_SIZE = 4672


def fake_evaluator(inputs, legal_offsets, legal_indices):
    """Deterministic logits (hash of the encoded position), zero values."""
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


def run_search(mcts, max_batch=32):
    """Drive gather/apply until is_complete(); returns the final gather."""
    last = None
    while not mcts.is_complete():
        last = mcts.gather_leaves(max_batch)
        tokens, inputs, offsets, indices = last
        if len(tokens) == 0:
            break
        logits, values = fake_evaluator(inputs, offsets, indices)
        mcts.apply_evaluations(tokens, offsets, logits, values)
    return last


def legal_uci(fen):
    return {m.uci() for m in chess.Board(fen).legal_moves}


def _new_mcts(num_simulations, seed=42, **kw):
    kw.setdefault("dirichlet_epsilon", 0.0)
    return chess_rl_native.MCTS(num_simulations=num_simulations, seed=seed, **kw)


# ---------------------------------------------------------------------------
# 1. Start position: loop runs to completion; policy and visit counts sane
# ---------------------------------------------------------------------------


def test_start_position_search_runs_to_completion_and_policy_is_sane():
    mcts = _new_mcts(num_simulations=20, seed=1)
    mcts.set_root(START_FEN, [])

    # First gather_leaves returns the ROOT as the single leaf (leaf 0).
    tokens, inputs, offsets, indices = mcts.gather_leaves(32)
    assert tokens == [0]
    assert inputs.shape == (1, PLANES, ROWS, COLS)
    assert inputs.dtype == np.float32
    assert inputs.flags["C_CONTIGUOUS"]
    assert offsets.shape == (2,)
    assert offsets[0] == 0
    assert len(indices) == offsets[-1] == 20  # 20 legal moves from the start

    logits, values = fake_evaluator(inputs, offsets, indices)
    mcts.apply_evaluations(tokens, offsets, logits, values)

    assert not mcts.is_complete()
    run_search(mcts, max_batch=32)
    assert mcts.is_complete()

    legal = legal_uci(START_FEN)
    policy = mcts.policy(1.0)
    ucis = [uci for uci, _ in policy]
    assert ucis == sorted(ucis), "policy must be sorted by UCI"
    assert set(ucis) == legal
    assert sum(prob for _, prob in policy) == pytest.approx(1.0, abs=1e-5)
    # With non-uniform priors and only `num_simulations` == 20 sims, a couple
    # of the lowest-prior moves legitimately receive zero visits (mcts.py does
    # the same), so we only require non-negativity here, not strict positivity.
    assert all(prob >= 0.0 for _, prob in policy)

    counts = mcts.root_visit_counts()
    assert [uci for uci, _ in counts] == sorted(uci for uci, _ in counts)
    assert {uci for uci, _ in counts} == legal
    visit_ints = [c for _, c in counts]
    assert all(isinstance(c, int) and c >= 0 for c in visit_ints)
    assert sum(visit_ints) == 20  # one root-child visit per simulation

    # temperature == 0 -> one-hot on the most visited move.
    best_uci = max(counts, key=lambda p: p[1])[0]
    onehot = mcts.policy(0.0)
    probs = dict(onehot)
    assert probs[best_uci] == pytest.approx(1.0)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert sum(1 for p in probs.values() if p > 0.0) == 1


# ---------------------------------------------------------------------------
# 2. Mate in 1: policy(0) must put all mass on the mating move
# ---------------------------------------------------------------------------


def test_mate_in_one_is_found_with_full_mass():
    mcts = _new_mcts(num_simulations=40, seed=2)
    mcts.set_root(MATE_IN_1_FEN, [])
    run_search(mcts, max_batch=32)
    assert mcts.is_complete()

    board = chess.Board(MATE_IN_1_FEN)
    mates = []
    for move in board.legal_moves:
        probe = board.copy()
        probe.push(move)
        if probe.is_checkmate():
            mates.append(move.uci())
    assert len(mates) == 1  # the fixture is a unique mate-in-1

    policy = mcts.policy(0.0)
    probs = dict(policy)
    assert probs[mates[0]] == pytest.approx(1.0)
    assert all(p == 0.0 for m, p in policy if m != mates[0])
    assert sum(probs.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Terminal root: empty policy, immediate completion
# ---------------------------------------------------------------------------


def test_terminal_root_stalemate_completes_immediately():
    mcts = _new_mcts(num_simulations=10, seed=3)
    mcts.set_root(STALEMATE_FEN, [])
    assert mcts.is_complete()

    tokens, inputs, offsets, indices = mcts.gather_leaves(8)
    assert tokens == []
    assert inputs.shape == (0, PLANES, ROWS, COLS)
    assert offsets.shape == (1,)
    assert offsets[0] == 0
    assert indices.shape == (0,)

    assert mcts.policy(0.0) == []
    assert mcts.policy(1.0) == []
    assert mcts.root_visit_counts() == []


# ---------------------------------------------------------------------------
# 4. Multi-leaf gather: tensor/CSR shape invariants across the whole loop
# ---------------------------------------------------------------------------


def test_multileaf_gather_tensor_invariants():
    mcts = _new_mcts(num_simulations=50, seed=4)
    mcts.set_root(START_FEN, [])
    run_search(mcts, max_batch=8)
    assert mcts.is_complete()

    # Re-run and inspect EVERY gather of a fresh, identical search.
    mcts2 = _new_mcts(num_simulations=50, seed=4)
    mcts2.set_root(START_FEN, [])
    gathers = []
    while not mcts2.is_complete():
        tokens, inputs, offsets, indices = mcts2.gather_leaves(8)
        if len(tokens) == 0:
            break
        logits, values = fake_evaluator(inputs, offsets, indices)
        mcts2.apply_evaluations(tokens, offsets, logits, values)
        gathers.append((tokens, inputs, offsets, indices))

    assert len(gathers) >= 3  # 1 root leaf + several multi-leaf batches
    for tokens, inputs, offsets, indices in gathers:
        B = len(tokens)
        assert tokens == list(range(B))
        assert inputs.shape == (B, PLANES, ROWS, COLS)
        assert inputs.dtype == np.float32
        assert inputs.flags["C_CONTIGUOUS"]
        assert offsets.dtype == np.int32
        assert indices.dtype == np.int32
        assert offsets.shape == (B + 1,)
        assert offsets[0] == 0
        assert int(offsets[-1]) == len(indices)
        assert int(offsets[-1]) >= B
        for i in range(B):
            row = indices[int(offsets[i]):int(offsets[i + 1])]
            assert len(row) >= 1
            assert np.all(np.diff(row) >= 0)  # sorted ascending within the row
            assert np.all(row >= 0) and np.all(row < POLICY_SIZE)

    # Every simulation visited exactly one root child.
    assert sum(c for _, c in mcts2.root_visit_counts()) == 50

    # Deterministic: identical parameters -> identical policies.
    assert mcts.policy(1.0) == mcts2.policy(1.0)


# ---------------------------------------------------------------------------
# 5. apply_evaluations input validation
# ---------------------------------------------------------------------------


def _fresh_gather(num_simulations=10, max_batch=8, seed=7):
    mcts = _new_mcts(num_simulations=num_simulations, seed=seed)
    mcts.set_root(START_FEN, [])
    tokens, inputs, offsets, indices = mcts.gather_leaves(max_batch)
    logits, values = fake_evaluator(inputs, offsets, indices)
    return mcts, tokens, inputs, offsets, indices, logits, values


def test_apply_evaluations_rejects_wrong_sized_arrays():
    mcts, tokens, inputs, offsets, indices, logits, values = _fresh_gather()
    B, K = len(tokens), len(indices)

    with pytest.raises(ValueError):
        mcts.apply_evaluations(tokens, offsets, logits, values[:-1])  # values too short
    with pytest.raises(ValueError):
        mcts.apply_evaluations(tokens + [0], offsets, logits, values)  # tokens too long
    with pytest.raises(ValueError):
        mcts.apply_evaluations(tokens, offsets[:-1], logits, values)  # offsets too short
    with pytest.raises(ValueError):
        mcts.apply_evaluations(tokens, offsets, logits[:-1], values)  # logits too short
    with pytest.raises(ValueError):
        mcts.apply_evaluations([B], offsets, logits, values)  # token out of range
    with pytest.raises(ValueError):
        mcts.apply_evaluations([-1], offsets, logits, values)  # token out of range

    assert len(logits) == K and len(values) == B  # the fixture itself is valid


def test_apply_evaluations_without_gather_or_twice_raises():
    mcts = _new_mcts(num_simulations=10, seed=8)
    mcts.set_root(START_FEN, [])
    with pytest.raises(ValueError):
        mcts.apply_evaluations(
            [],
            np.zeros(1, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    mcts, tokens, inputs, offsets, indices, logits, values = _fresh_gather(seed=9)
    mcts.apply_evaluations(tokens, offsets, logits, values)
    with pytest.raises(ValueError):  # pending list already consumed
        mcts.apply_evaluations(tokens, offsets, logits, values)


# ---------------------------------------------------------------------------
# 6. set_root resets state; UCI history is respected
# ---------------------------------------------------------------------------


def test_set_root_resets_search_state():
    mcts = _new_mcts(num_simulations=10, seed=10)
    mcts.set_root(START_FEN, [])
    run_search(mcts)
    assert mcts.is_complete()

    mcts.set_root(START_FEN, [])  # fresh search on the same root
    assert not mcts.is_complete()
    tokens, inputs, offsets, indices = mcts.gather_leaves(8)
    assert tokens == [0]
    run_search(mcts)
    assert mcts.is_complete()


def test_set_root_with_uci_history_resolves_side_to_move():
    # After 1. e4 the side to move is black; the policy must cover black's
    # legal moves at the root.
    mcts = _new_mcts(num_simulations=8, seed=11)
    mcts.set_root(START_FEN, ["e2e4"])
    run_search(mcts, max_batch=8)

    board = chess.Board(START_FEN)
    board.push_uci("e2e4")
    legal_black = {m.uci() for m in board.legal_moves}
    policy = dict(mcts.policy(1.0))
    assert set(policy) == legal_black
    assert sum(policy.values()) == pytest.approx(1.0)
