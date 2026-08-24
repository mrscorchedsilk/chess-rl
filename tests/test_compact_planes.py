"""Compact uint8 board planes: exact-expansion contract.

The native actor emits board planes as uint8 rather than float32, which moves
a quarter of the bytes over PCIe and removes the CPU-side float cast that
dominated `InferenceRuntime.prepare`.  That is only safe because exactly one
of the 104 planes is non-binary:

  * 96 history piece planes and 7 meta planes are strictly 0/1;
  * the halfmove-clock plane is ``clock / 100.0`` in the float encoder, and
    stores the RAW clock in the compact encoder.

These tests pin that contract from both ends: that the binary planes really
are binary for awkward positions (castling rights, en passant, repetition,
promotions, black to move), and that dividing the halfmove plane by
HALFMOVE_SCALE reproduces the float encoder BIT-EXACTLY rather than
approximately.

Run:  .venv/bin/python -m pytest tests/test_compact_planes.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess_rl_native as native  # noqa: E402

HISTORY_STEPS = 8
META_BASE = 12 * HISTORY_STEPS
HM_PLANE = META_BASE + native.HALFMOVE_META_PLANE
SCALE = np.float32(native.HALFMOVE_SCALE)


def expand(u8: np.ndarray) -> np.ndarray:
    """The documented consumer-side expansion, in float32."""
    out = u8.astype(np.float32)
    out[HM_PLANE] = out[HM_PLANE] / SCALE
    return out


FENS = [
    # start position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # all castling rights, black to move, non-zero clock
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 5 4",
    # en passant target set
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    # no castling rights at all, high halfmove clock
    "8/8/8/4k3/8/8/4K3/8 w - - 87 120",
    # partial castling rights (white queenside + black kingside only)
    "r3k2r/8/8/8/8/8/8/R3K2R w Qk - 13 40",
    # promotion-adjacent material, black to move
    "8/P6k/8/8/8/8/6Kp/8 b - - 3 60",
    # near the 50-move boundary
    "8/8/4k3/8/8/4K3/8/8 w - - 99 150",
    # clock exactly at the 75-move (150 ply) rule limit
    "8/8/4k3/8/8/4K3/8/8 b - - 150 200",
]


# --------------------------------------------------------------------------- #
#  the halfmove plane                                                         #
# --------------------------------------------------------------------------- #

def test_halfmove_roundtrip_is_bit_exact_for_every_storable_clock():
    """float32(double(n)/100.0) must equal float32(float32(n)/float32(100)).

    This is the whole justification for storing the raw clock in a uint8: if
    the two divides disagreed anywhere in 0..255, the compact encoder would
    silently feed the network a different value from the float encoder.
    """
    mismatches = []
    for n in range(256):
        cpp = np.float32(np.float64(n) / np.float64(100.0))
        consumer = np.float32(np.float32(n) / SCALE)
        if cpp != consumer:
            mismatches.append(n)
    assert mismatches == []


def test_halfmove_plane_stores_the_raw_clock():
    u8 = native.encode_fen_u8("8/8/8/4k3/8/8/4K3/8 w - - 87 120")
    assert int(u8[HM_PLANE, 0, 0]) == 87
    assert np.all(u8[HM_PLANE] == 87), "the clock fills the whole plane"


def test_halfmove_plane_expands_to_the_float_encoding():
    fen = "8/8/8/4k3/8/8/4K3/8 w - - 87 120"
    f32 = native.encode_fen(fen)
    u8 = native.encode_fen_u8(fen)
    assert np.array_equal(expand(u8)[HM_PLANE], f32[HM_PLANE])
    assert f32[HM_PLANE, 0, 0] == np.float32(0.87)


def test_clock_above_255_wraps_upstream_and_both_encoders_agree():
    """Documents a PRE-EXISTING upstream behaviour, not a compact-plane bug.

    The vendored chess-library truncates the FEN halfmove-clock field modulo
    256 while parsing, so a FEN claiming 300 is already 44 by the time either
    encoder sees it — the float encoder reports 0.44 too.  The compact encoder
    must therefore agree with the float encoder here rather than "fix" it,
    because a divergence is what would actually hurt: it would feed the
    network a different feature from the one the reference path produces.

    Self-play never reaches this: the 75-move rule ends a game at 150 plies.
    HALFMOVE_CLAMP remains as a defensive guard should the upstream parser
    ever start returning the true value.
    """
    fen = "8/8/4k3/8/8/4K3/8/8 w - - 300 400"
    stored = int(native.encode_fen_u8(fen)[HM_PLANE, 0, 0])
    as_float = float(native.encode_fen(fen)[HM_PLANE, 0, 0])
    assert stored == 300 % 256 == 44
    assert as_float == pytest.approx(0.44)
    assert np.array_equal(expand(native.encode_fen_u8(fen)),
                          native.encode_fen(fen).astype(np.float32))


def test_clamp_constant_is_the_uint8_ceiling():
    """Whatever the parser does, the stored byte can never alias."""
    assert native.HALFMOVE_CLAMP == 255
    for fen in FENS:
        assert 0 <= int(native.encode_fen_u8(fen)[HM_PLANE, 0, 0]) <= 255


# --------------------------------------------------------------------------- #
#  every other plane                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fen", FENS)
def test_all_planes_except_halfmove_are_strictly_binary(fen):
    u8 = native.encode_fen_u8(fen)
    others = np.delete(u8, HM_PLANE, axis=0)
    assert set(np.unique(others).tolist()) <= {0, 1}


@pytest.mark.parametrize("fen", FENS)
def test_compact_expansion_equals_float_encoder_exactly(fen):
    f32 = native.encode_fen(fen).astype(np.float32)
    got = expand(native.encode_fen_u8(fen))
    assert np.array_equal(got, f32), (
        f"compact expansion differs from encode_fen for {fen}; "
        f"max abs diff {np.abs(got - f32).max()}"
    )


@pytest.mark.parametrize("fen", FENS)
def test_compact_encoding_is_four_times_smaller(fen):
    f32 = native.encode_fen(fen)
    u8 = native.encode_fen_u8(fen)
    assert u8.shape == f32.shape
    assert u8.dtype == np.uint8
    assert f32.nbytes == 4 * u8.nbytes


# --------------------------------------------------------------------------- #
#  history and repetition metadata survive the narrowing                      #
# --------------------------------------------------------------------------- #

def test_history_planes_match_after_real_moves():
    """History planes are populated only after moves; they must still match."""
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    pos = native.Position.from_uci_history(start, moves)
    f32 = np.asarray(pos.encode(HISTORY_STEPS)).astype(np.float32)
    u8 = np.asarray(pos.encode_u8(HISTORY_STEPS))
    assert np.array_equal(expand(u8), f32)
    # history really is populated, otherwise this test proves nothing
    assert f32[12:96].any()


def test_repetition_plane_matches_after_a_repeated_position():
    """The repetition meta plane is binary and must survive narrowing."""
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # knights out and back: the start position recurs
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"]
    pos = native.Position.from_uci_history(start, moves)
    f32 = np.asarray(pos.encode(HISTORY_STEPS)).astype(np.float32)
    u8 = np.asarray(pos.encode_u8(HISTORY_STEPS))
    rep = META_BASE + 7
    assert np.array_equal(expand(u8), f32)
    assert set(np.unique(u8[rep]).tolist()) <= {0, 1}
    assert f32[rep, 0, 0] == 1.0, "expected a detected repetition"


def test_actor_gather_emits_compact_planes():
    actor = native.Actor(games=2, c_puct=1.25, virtual_loss=3.0,
                         num_simulations=4, temperature=1.0,
                         temperature_threshold=30, max_game_length=8,
                         seed=5, num_threads=2)
    actor.set_teacher(0, 0)
    tokens, inputs, offsets, indices = actor.gather_leaves(32)
    assert inputs.dtype == np.uint8
    assert inputs.shape[1:] == (104, 8, 8)


# --------------------------------------------------------------------------- #
#  the GPU consumer                                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA required"
)
def test_gpu_runtime_uint8_and_float32_paths_agree_exactly():
    from config import Config
    from gpu_runtime import InferenceRuntime
    from model import ChessNet

    cfg = Config()
    cfg.num_res_blocks, cfg.num_filters = 2, 16
    rt = InferenceRuntime(cfg=cfg, model=ChessNet(cfg))
    assert rt.halfmove_plane == HM_PLANE

    fens = FENS[:4]
    f32 = np.stack([native.encode_fen(f) for f in fens]).astype(np.float32)
    u8 = np.stack([native.encode_fen_u8(f) for f in fens])
    per = 20
    off = np.arange(0, (len(fens) + 1) * per, per, dtype=np.int32)
    idx = np.sort(np.random.default_rng(0).integers(
        0, cfg.policy_size, size=len(fens) * per)).astype(np.int32)

    la, va = rt.evaluate(f32, off, idx)
    lb, vb = rt.evaluate(u8, off, idx)
    assert np.array_equal(la, lb)
    assert np.array_equal(va, vb)


def test_runtime_rejects_unsupported_dtypes():
    from config import Config
    from gpu_runtime import InferenceRuntime
    from model import ChessNet
    import torch

    cfg = Config()
    cfg.num_res_blocks, cfg.num_filters = 1, 8
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cpu":
        pytest.skip("CUDA required")
    rt = InferenceRuntime(cfg=cfg, model=ChessNet(cfg))
    bad = np.zeros((2, 104, 8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="float32 or uint8"):
        rt.evaluate(bad, np.array([0, 1, 2], dtype=np.int32),
                    np.array([0, 1], dtype=np.int32))
