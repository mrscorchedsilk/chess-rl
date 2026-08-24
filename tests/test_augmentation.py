"""Exact colour-flip augmentation.

Chess has no dihedral symmetry, but mirroring the board vertically and
swapping colours is an exact symmetry: the position is legal, its value from
the mover's point of view is unchanged, and its moves are the mirror of the
original's.  Applying it doubles the usable replay for free.

Only if it is exact.  The input planes are in ABSOLUTE White orientation (not
side-to-move orientation), so this is a real transformation of every plane;
the policy index is ``from_square * 73 + plane`` and BOTH halves move; and
castling rights, en passant and side-to-move all have to follow.

The two checks that actually prove it are against independent references:
``flip_planes`` against the native encoder applied to ``board.mirror()``, and
``POLICY_MIRROR`` against python-chess's own move mirroring, over every legal
move of a corpus of awkward positions.

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_augmentation.py -q
"""
import os
import sys

import chess
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import augment                       # noqa: E402
import chess_rl_native as native     # noqa: E402
from encoding import POLICY_SIZE, move_to_index   # noqa: E402


FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 5 4",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "r3k2r/8/8/8/8/8/8/R3K2R w Qk - 13 40",
    "8/P6k/8/8/8/8/6Kp/8 b - - 3 60",
    "8/8/8/4k3/8/8/4K3/8 w - - 87 120",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
]


# --------------------------------------------------------------------------- #
#  the permutation itself                                                     #
# --------------------------------------------------------------------------- #

def test_policy_mirror_is_a_permutation():
    assert augment.POLICY_MIRROR.shape == (POLICY_SIZE,)
    assert len(np.unique(augment.POLICY_MIRROR)) == POLICY_SIZE


def test_policy_mirror_is_an_involution():
    """Mirroring twice must be the identity — it is its own inverse."""
    twice = augment.POLICY_MIRROR[augment.POLICY_MIRROR]
    assert np.array_equal(twice, np.arange(POLICY_SIZE))


def test_direction_tables_are_closed_under_mirroring():
    assert sorted(augment.QUEEN_DIR_MIRROR) == list(range(8))
    assert sorted(augment.KNIGHT_MIRROR) == list(range(8))


def test_underpromotion_planes_map_to_themselves():
    """A vertical mirror preserves FILE deltas, and the piece is unchanged."""
    base = 56 + 8
    for u in range(9):
        assert augment.PLANE_MIRROR[base + u] == base + u


# --------------------------------------------------------------------------- #
#  against python-chess                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fen", FENS)
def test_policy_mirror_matches_python_chess_move_mirroring(fen):
    board = chess.Board(fen)
    checked = 0
    for move in board.legal_moves:
        mirrored = chess.Move(chess.square_mirror(move.from_square),
                              chess.square_mirror(move.to_square),
                              promotion=move.promotion)
        assert int(augment.POLICY_MIRROR[move_to_index(move)]) == \
            move_to_index(mirrored), (fen, move.uci(), mirrored.uci())
        checked += 1
    assert checked > 0


def test_every_legal_move_of_the_corpus_is_covered():
    total = sum(len(list(chess.Board(f).legal_moves)) for f in FENS)
    assert total > 100, "corpus too small to be meaningful"


# --------------------------------------------------------------------------- #
#  against the encoder                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fen", FENS)
def test_flip_planes_equals_encoding_the_mirrored_board(fen):
    board = chess.Board(fen)
    original = np.asarray(native.encode_fen(fen)).astype(np.float32)
    expected = np.asarray(
        native.encode_fen(board.mirror().fen())).astype(np.float32)
    got = augment.flip_planes(original)
    assert np.array_equal(got, expected), (
        f"planes differing: "
        f"{np.where(np.any(got != expected, axis=(1, 2)))[0].tolist()}"
    )


def test_flip_planes_is_an_involution():
    state = np.asarray(native.encode_fen(FENS[1])).astype(np.float32)
    assert np.array_equal(augment.flip_planes(augment.flip_planes(state)), state)


def test_side_to_move_plane_flips():
    white = np.asarray(native.encode_fen(FENS[0])).astype(np.float32)
    m = augment.META_BASE
    assert white[m + augment.META_SIDE_TO_MOVE, 0, 0] == 1.0
    assert augment.flip_planes(white)[m + augment.META_SIDE_TO_MOVE, 0, 0] == 0.0


def test_castling_rights_follow_the_colour_swap():
    """White queenside + black kingside only; must become the reverse."""
    state = np.asarray(native.encode_fen(
        "r3k2r/8/8/8/8/8/8/R3K2R w Qk - 13 40")).astype(np.float32)
    m = augment.META_BASE
    assert state[m + augment.META_WQ, 0, 0] == 1.0
    assert state[m + augment.META_BK, 0, 0] == 1.0
    assert state[m + augment.META_WK, 0, 0] == 0.0
    flipped = augment.flip_planes(state)
    assert flipped[m + augment.META_WK, 0, 0] == 1.0
    assert flipped[m + augment.META_BQ, 0, 0] == 1.0
    assert flipped[m + augment.META_WQ, 0, 0] == 0.0


def test_halfmove_and_repetition_are_colour_agnostic():
    state = np.asarray(native.encode_fen(
        "8/8/8/4k3/8/8/4K3/8 w - - 87 120")).astype(np.float32)
    m = augment.META_BASE
    flipped = augment.flip_planes(state)
    assert np.array_equal(flipped[m + augment.META_HALFMOVE],
                          state[m + augment.META_HALFMOVE])
    assert np.array_equal(flipped[m + augment.META_REPETITION],
                          state[m + augment.META_REPETITION])


def test_en_passant_square_is_mirrored():
    state = np.asarray(native.encode_fen(
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    ).astype(np.float32)
    m = augment.META_BASE + augment.META_EP
    assert state[m].sum() == 1.0
    flipped = augment.flip_planes(state)
    assert flipped[m].sum() == 1.0
    r0, f0 = np.argwhere(state[m] == 1.0)[0]
    r1, f1 = np.argwhere(flipped[m] == 1.0)[0]
    assert f1 == f0 and r1 == 7 - r0


# --------------------------------------------------------------------------- #
#  policy and value targets                                                   #
# --------------------------------------------------------------------------- #

def test_flip_policy_preserves_mass_and_moves_probability_correctly():
    board = chess.Board(FENS[1])
    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    moves = list(board.legal_moves)
    for i, mv in enumerate(moves):
        pi[move_to_index(mv)] = 1.0 / len(moves)
    flipped = augment.flip_policy(pi)
    assert flipped.sum() == pytest.approx(pi.sum())
    for mv in moves:
        mirrored = chess.Move(chess.square_mirror(mv.from_square),
                              chess.square_mirror(mv.to_square),
                              promotion=mv.promotion)
        assert flipped[move_to_index(mirrored)] == pytest.approx(
            pi[move_to_index(mv)])


def test_flip_policy_is_an_involution():
    rng = np.random.default_rng(0)
    pi = rng.random(POLICY_SIZE).astype(np.float32)
    assert np.allclose(augment.flip_policy(augment.flip_policy(pi)), pi)


def test_value_is_unchanged_because_z_is_from_the_movers_point_of_view():
    state = np.asarray(native.encode_fen(FENS[0])).astype(np.float32)
    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    pi[0] = 1.0
    _, _, z = augment.flip_example(state, pi, 1.0)
    assert z == 1.0


def test_flip_example_returns_all_three_transformed_parts():
    state = np.asarray(native.encode_fen(FENS[1])).astype(np.float32)
    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    pi[move_to_index(next(iter(chess.Board(FENS[1]).legal_moves)))] = 1.0
    s2, p2, z2 = augment.flip_example(state, pi, -1.0)
    assert not np.array_equal(s2, state)
    assert not np.array_equal(p2, pi)
    assert z2 == -1.0


# --------------------------------------------------------------------------- #
#  shape validation                                                           #
# --------------------------------------------------------------------------- #

def test_wrong_plane_shape_is_rejected():
    with pytest.raises(ValueError, match="planes"):
        augment.flip_planes(np.zeros((12, 8, 8), dtype=np.float32))


def test_wrong_policy_length_is_rejected():
    with pytest.raises(ValueError, match="policy"):
        augment.flip_policy(np.zeros(100, dtype=np.float32))


def test_augment_examples_doubles_the_stream():
    state = np.asarray(native.encode_fen(FENS[0])).astype(np.float32)
    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    pi[0] = 1.0
    rows = [(state, pi, 0.0)] * 5
    assert len(list(augment.augment_examples(rows))) == 10
    assert len(list(augment.augment_examples(rows, include_original=False))) == 5


# --------------------------------------------------------------------------- #
#  batched torch path (what training actually uses)                           #
# --------------------------------------------------------------------------- #

def _torch():
    import torch
    return torch


def _state_batch():
    return np.stack([np.asarray(native.encode_fen(f)) for f in FENS]
                    ).astype(np.float32)


def test_torch_plane_flip_matches_the_numpy_reference_exactly():
    torch = _torch()
    arr = _state_batch()
    got = augment.flip_planes_batch(torch.from_numpy(arr)).numpy()
    want = np.stack([augment.flip_planes(a) for a in arr])
    assert np.array_equal(got, want)


def test_torch_policy_flip_matches_the_numpy_reference_exactly():
    torch = _torch()
    rng = np.random.default_rng(0)
    pi = rng.random((6, POLICY_SIZE)).astype(np.float32)
    got = augment.flip_policy_batch(torch.from_numpy(pi)).numpy()
    want = np.stack([augment.flip_policy(p) for p in pi])
    assert np.array_equal(got, want)


def test_torch_flip_is_an_involution():
    torch = _torch()
    arr = torch.from_numpy(_state_batch())
    assert torch.equal(augment.flip_planes_batch(
        augment.flip_planes_batch(arr)), arr)


def test_augment_batch_only_touches_masked_rows():
    torch = _torch()
    states = torch.from_numpy(_state_batch())
    pis = torch.from_numpy(
        np.random.default_rng(1).random((states.shape[0], POLICY_SIZE)
                                        ).astype(np.float32))
    mask = torch.zeros(states.shape[0], dtype=torch.bool)
    mask[1] = True
    out_s, out_p = augment.augment_batch(states, pis, mask)
    assert torch.equal(out_s[0], states[0])
    assert torch.equal(out_p[0], pis[0])
    assert torch.equal(out_s[1], augment.flip_planes_batch(states[1:2])[0])
    assert not torch.equal(out_s[1], states[1])


def test_augment_batch_leaves_inputs_untouched():
    torch = _torch()
    states = torch.from_numpy(_state_batch())
    pis = torch.zeros((states.shape[0], POLICY_SIZE))
    original = states.clone()
    mask = torch.ones(states.shape[0], dtype=torch.bool)
    augment.augment_batch(states, pis, mask)
    assert torch.equal(states, original)


def test_empty_mask_is_a_no_op_without_copying():
    torch = _torch()
    states = torch.from_numpy(_state_batch())
    pis = torch.zeros((states.shape[0], POLICY_SIZE))
    mask = torch.zeros(states.shape[0], dtype=torch.bool)
    out_s, out_p = augment.augment_batch(states, pis, mask)
    assert out_s is states and out_p is pis


def test_wrong_batch_shape_is_rejected():
    torch = _torch()
    with pytest.raises(ValueError, match="states"):
        augment.flip_planes_batch(torch.zeros(2, 12, 8, 8))
