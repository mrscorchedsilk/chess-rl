"""Exact colour-flip augmentation for chess training examples.

Chess has none of Go's eight-fold board symmetry, so AlphaGo-style dihedral
augmentation is unavailable.  One exact symmetry does exist: mirror the board
vertically (rank r -> 7-r) and swap the two colours.  The resulting position
is legal, has the same game-theoretic value from the mover's point of view,
and its move set is the mirror of the original's.  Applying it doubles the
effective size of the replay buffer for free.

"Free" only if it is exact.  Three things have to be transformed together and
any one of them being wrong silently teaches the network nonsense:

  * the 104 input planes (which are in ABSOLUTE White orientation, not
    side-to-move orientation, so the flip is a real transformation and not a
    no-op);
  * the 4672-entry policy target, whose index is ``from_square * 73 + plane``
    — both halves move;
  * the value target, which negates.

The policy permutation is built from the action-map tables rather than
hand-written, and ``tests/test_augmentation.py`` checks it against
python-chess's own ``Move`` mirroring across every legal move of a corpus of
positions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from encoding import (
    KNIGHT_OFFSETS,
    KNIGHT_PLANES,
    POLICY_PLANES,
    POLICY_SIZE,
    QUEEN_DIRECTIONS,
    QUEEN_PLANES,
    UNDERPROMOTION_PLANES,
)

HISTORY_STEPS = 8
PIECE_PLANES_PER_STEP = 12
META_BASE = PIECE_PLANES_PER_STEP * HISTORY_STEPS   # 96
NUM_PLANES = META_BASE + 8                          # 104

# Meta plane offsets (relative to META_BASE), mirroring position.cpp.
META_SIDE_TO_MOVE = 0
META_WK, META_WQ, META_BK, META_BQ = 1, 2, 3, 4
META_EP = 5
META_HALFMOVE = 6
META_REPETITION = 7


def _mirror_direction_table(table):
    """Index -> index of the vertically mirrored (dr, df) -> (-dr, df) delta."""
    lookup = {delta: i for i, delta in enumerate(table)}
    out = []
    for dr, df in table:
        mirrored = (-dr, df)
        if mirrored not in lookup:
            raise ValueError(
                f"direction table is not closed under vertical mirroring: "
                f"{(dr, df)} -> {mirrored} is absent"
            )
        out.append(lookup[mirrored])
    return tuple(out)


QUEEN_DIR_MIRROR = _mirror_direction_table(QUEEN_DIRECTIONS)
KNIGHT_MIRROR = _mirror_direction_table(KNIGHT_OFFSETS)


def _build_plane_mirror():
    """73-entry permutation of the policy planes under a vertical mirror."""
    planes = np.empty(POLICY_PLANES, dtype=np.int64)
    for d in range(len(QUEEN_DIRECTIONS)):
        for dist in range(7):
            planes[d * 7 + dist] = QUEEN_DIR_MIRROR[d] * 7 + dist
    for k in range(KNIGHT_PLANES):
        planes[QUEEN_PLANES + k] = QUEEN_PLANES + KNIGHT_MIRROR[k]
    # Underpromotion planes encode (piece, file delta).  A vertical mirror
    # leaves the FILE delta untouched and the promoting piece unchanged, so
    # these planes map to themselves; only the from-square moves.
    base = QUEEN_PLANES + KNIGHT_PLANES
    for u in range(UNDERPROMOTION_PLANES):
        planes[base + u] = base + u
    return planes


PLANE_MIRROR = _build_plane_mirror()

# square ^ 56 flips the rank and leaves the file: python-chess's square_mirror.
SQUARE_MIRROR = np.arange(64, dtype=np.int64) ^ 56

# Full 4672 permutation: index = from_square * 73 + plane.
POLICY_MIRROR = (
    SQUARE_MIRROR[:, None] * POLICY_PLANES + PLANE_MIRROR[None, :]
).reshape(POLICY_SIZE)


def flip_planes(state: np.ndarray) -> np.ndarray:
    """Colour-flip + vertical mirror of a (104, 8, 8) encoded position."""
    state = np.asarray(state)
    if state.shape != (NUM_PLANES, 8, 8):
        raise ValueError(
            f"expected ({NUM_PLANES}, 8, 8) planes, got {state.shape}"
        )
    out = np.empty_like(state)

    # ---- history piece planes: swap colours, then mirror ranks ---- #
    for step in range(HISTORY_STEPS):
        off = step * PIECE_PLANES_PER_STEP
        for pt in range(6):
            out[off + pt] = state[off + pt + 6][::-1, :]      # black -> white
            out[off + pt + 6] = state[off + pt][::-1, :]      # white -> black

    m = META_BASE
    # Side to move flips.
    out[m + META_SIDE_TO_MOVE] = 1.0 - state[m + META_SIDE_TO_MOVE]
    # Castling rights follow the colour swap; king/queen side is a FILE
    # property and a vertical mirror does not touch files.
    out[m + META_WK] = state[m + META_BK]
    out[m + META_WQ] = state[m + META_BQ]
    out[m + META_BK] = state[m + META_WK]
    out[m + META_BQ] = state[m + META_WQ]
    # En passant target is a square: mirror its rank.
    out[m + META_EP] = state[m + META_EP][::-1, :]
    # Halfmove clock and repetition are colour-agnostic scalars.
    out[m + META_HALFMOVE] = state[m + META_HALFMOVE]
    out[m + META_REPETITION] = state[m + META_REPETITION]
    return out


def flip_policy(pi: np.ndarray) -> np.ndarray:
    """Permute a 4672-entry policy target under the mirror."""
    pi = np.asarray(pi).reshape(-1)
    if pi.shape[0] != POLICY_SIZE:
        raise ValueError(f"expected {POLICY_SIZE} policy entries, got {pi.shape[0]}")
    out = np.zeros_like(pi)
    out[POLICY_MIRROR] = pi
    return out


def flip_example(state: np.ndarray, pi: np.ndarray, z: float
                 ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Colour-flip one (state, pi, z) training example.

    ``z`` is the outcome from the point of view of the side to move.  The flip
    swaps which colour is to move, but z stays FROM THE MOVER'S POINT OF VIEW
    — and the mover in the flipped position is the same player, just recoloured
    — so z is unchanged.  (Negating it here would teach the network that every
    position is exactly as good for one side as it is bad for the other side
    of the *mirrored* board, which is not what the label means.)
    """
    return flip_planes(state), flip_policy(pi), float(z)


def augment_examples(examples, include_original: bool = True):
    """Yield colour-flipped copies of ``(state, pi, z)`` triples."""
    for state, pi, z in examples:
        if include_original:
            yield state, pi, z
        yield flip_example(state, pi, z)
