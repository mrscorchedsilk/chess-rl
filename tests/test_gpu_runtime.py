"""Task 7: persistent GPU inference runtime (gpu_runtime.py) — strict TDD.

CUDA-only.  Xfails cleanly (run=False) when CUDA is unavailable so CPU CI
stays green.

Run: .venv/bin/python -m pytest tests/test_gpu_runtime.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from gpu_runtime import (  # noqa: E402
    BUCKETS,
    InferenceRuntime,
    evaluate as module_evaluate,
    set_default_runtime,
)
from model import ChessNet  # noqa: E402

CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.xfail(not CUDA_OK, reason="CUDA unavailable", run=False)

PLANES, SIZE = 104, 8
POLICY_SIZE = 4672


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #
def _tiny_cfg():
    cfg = Config()
    cfg.num_res_blocks = 1
    cfg.num_filters = 8
    cfg.architecture_id = "custom-1x8"
    return cfg


def _make_models():
    """(cfg, reference net, runtime net with copied weights), all CUDA fp32."""
    cfg = _tiny_cfg()
    ref = ChessNet(cfg).to("cuda").eval()
    rt_model = ChessNet(cfg).to("cuda").eval()
    rt_model.load_state_dict(ref.state_dict())
    return cfg, ref, rt_model


def _make_batch(B, seed, min_len=1, max_len=40, policy_size=POLICY_SIZE):
    """Random inputs + random-but-valid CSR legal indices (sorted per row)."""
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((B, PLANES, SIZE, SIZE)).astype(np.float32)
    lengths = rng.integers(min_len, max_len + 1, size=B)
    offsets = np.zeros(B + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths).astype(np.int32)
    idx = np.concatenate(
        [np.sort(rng.choice(policy_size, int(l), replace=False)) for l in lengths]
    ).astype(np.int32)
    return inputs, offsets, idx


def _dense_reference(cfg, ref, inputs, offsets, indices):
    """Plain FP32 eager forward of the SAME net, dense-gathered in numpy."""
    B = inputs.shape[0]
    with torch.no_grad():
        plogits, pval = ref(torch.from_numpy(inputs).to("cuda"))
    plogits = plogits.float().cpu().numpy()  # [B, 4672]
    pval = pval.float().cpu().numpy()  # [B, 1]
    lengths = offsets[1:] - offsets[:-1]
    row_ids = np.repeat(np.arange(B), lengths)
    exp_logits = plogits.reshape(-1)[row_ids * POLICY_SIZE + indices]
    return exp_logits, pval


# --------------------------------------------------------------------------- #
#  1. shapes / dtypes / value range                                            #
# --------------------------------------------------------------------------- #
def test_shapes_dtypes_and_value_range():
    cfg, ref, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, compile=False)
    try:
        inputs, offsets, indices = _make_batch(32, seed=1)
        K = len(indices)
        logits, values = rt.evaluate(inputs, offsets, indices)
        assert logits.shape == (K,), f"expected (K,)={K}, got {logits.shape}"
        assert values.shape == (32, 1), f"got {values.shape}"
        assert logits.dtype == np.float32 and values.dtype == np.float32
        assert np.all(np.isfinite(logits)) and np.all(np.isfinite(values))
        assert np.all(values >= -1.0 - 1e-5) and np.all(values <= 1.0 + 1e-5)
    finally:
        rt.close()


# --------------------------------------------------------------------------- #
#  2. legal logits == dense policy gathered at legal_indices (FP16 tolerance)  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("amp", [True, False])
def test_legal_logits_match_dense_gather(amp):
    cfg, ref, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, amp=amp, compile=False)
    try:
        inputs, offsets, indices = _make_batch(32, seed=2)
        logits, values = rt.evaluate(inputs, offsets, indices)
        exp_logits, exp_values = _dense_reference(cfg, ref, inputs, offsets, indices)
        if amp:  # FP16 body tolerance
            np.testing.assert_allclose(logits, exp_logits, rtol=2e-2, atol=2e-2)
            np.testing.assert_allclose(values, exp_values, rtol=1e-2, atol=1e-2)
        else:  # full FP32 path matches the eager reference to rounding
            np.testing.assert_allclose(logits, exp_logits, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(values, exp_values, rtol=1e-4, atol=1e-4)
    finally:
        rt.close()


# --------------------------------------------------------------------------- #
#  4. batch bucketing: B=1, 33, 100 (padding is masked, never returned)       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B", [1, 33, 100, 256])
def test_batch_bucketing_padding_is_masked(B):
    assert 256 in BUCKETS
    cfg, ref, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, compile=False)
    try:
        inputs, offsets, indices = _make_batch(B, seed=B)
        logits, values = rt.evaluate(inputs, offsets, indices)
        assert logits.shape == (len(indices),)
        assert values.shape == (B, 1)
        exp_logits, exp_values = _dense_reference(cfg, ref, inputs, offsets, indices)
        np.testing.assert_allclose(logits, exp_logits, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(values, exp_values, rtol=1e-2, atol=1e-2)
    finally:
        rt.close()


def test_batch_above_largest_bucket_raises():
    cfg, _, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, compile=False)
    try:
        inputs, offsets, indices = _make_batch(300, seed=0)
        with pytest.raises(ValueError, match="bucket"):
            rt.evaluate(inputs, offsets, indices)
    finally:
        rt.close()


# --------------------------------------------------------------------------- #
#  5. non-blocking streamed path vs eager reference (covered above) + compile  #
# --------------------------------------------------------------------------- #
def test_compile_mode_matches_reference():
    cfg, ref, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, compile=True)
    try:
        inputs, offsets, indices = _make_batch(16, seed=7)
        logits, values = rt.evaluate(inputs, offsets, indices)
        assert logits.shape == (len(indices),)
        assert values.shape == (16, 1)
        exp_logits, exp_values = _dense_reference(cfg, ref, inputs, offsets, indices)
        np.testing.assert_allclose(logits, exp_logits, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(values, exp_values, rtol=2e-2, atol=2e-2)
    finally:
        rt.close()


def test_streamed_and_compiled_agree():
    cfg, _, rt_model = _make_models()
    rt_a = InferenceRuntime(cfg=cfg, model=rt_model, amp=True, compile=False)
    rt_b = InferenceRuntime(cfg=cfg, model=rt_model, amp=True, compile=True)
    try:
        inputs, offsets, indices = _make_batch(24, seed=11)
        la, va = rt_a.evaluate(inputs, offsets, indices)
        lb, vb = rt_b.evaluate(inputs, offsets, indices)
        np.testing.assert_allclose(lb, la, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(vb, va, rtol=2e-2, atol=2e-2)
    finally:
        rt_a.close()
        rt_b.close()


def test_module_level_evaluate_contract():
    cfg, _, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, compile=False)
    set_default_runtime(rt)
    try:
        inputs, offsets, indices = _make_batch(8, seed=3)
        logits, values = module_evaluate(inputs, offsets, indices)
        assert logits.shape == (len(indices),)
        assert values.shape == (8, 1)
        assert logits.dtype == np.float32 and values.dtype == np.float32
    finally:
        rt.close()


# --------------------------------------------------------------------------- #
#  7. determinism + masked/empty CSR rows                                      #
# --------------------------------------------------------------------------- #
def test_repeated_calls_are_bitwise_deterministic():
    cfg, _, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, amp=False, compile=False)
    try:
        inputs, offsets, indices = _make_batch(16, seed=5)
        l1, v1 = rt.evaluate(inputs, offsets, indices)
        l2, v2 = rt.evaluate(inputs, offsets, indices)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(v1, v2)
    finally:
        rt.close()


def test_masked_row_with_empty_csr_range_is_skipped():
    cfg, ref, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, compile=False)
    try:
        B = 3
        rng = np.random.default_rng(9)
        inputs = rng.standard_normal((B, PLANES, SIZE, SIZE)).astype(np.float32)
        # row 1 is masked (empty CSR range); rows 0 and 2 have 5 and 7 moves.
        offsets = np.array([0, 5, 5, 12], dtype=np.int32)
        idx = np.concatenate(
            [
                np.sort(rng.choice(POLICY_SIZE, 5, replace=False)),
                np.sort(rng.choice(POLICY_SIZE, 7, replace=False)),
            ]
        ).astype(np.int32)
        logits, values = rt.evaluate(inputs, offsets, idx)
        assert logits.shape == (12,)
        assert values.shape == (B, 1)
        exp_logits, exp_values = _dense_reference(cfg, ref, inputs, offsets, idx)
        np.testing.assert_allclose(logits, exp_logits, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(values, exp_values, rtol=1e-2, atol=1e-2)
    finally:
        rt.close()


def test_zero_legal_moves_batch_returns_empty_logits():
    cfg, _, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, compile=False)
    try:
        inputs = np.zeros((1, PLANES, SIZE, SIZE), dtype=np.float32)
        offsets = np.zeros(2, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        logits, values = rt.evaluate(inputs, offsets, indices)
        assert logits.shape == (0,)
        assert values.shape == (1, 1)
        assert np.isfinite(values[0, 0])
    finally:
        rt.close()


# --------------------------------------------------------------------------- #
#  validation                                                                  #
# --------------------------------------------------------------------------- #
def test_csr_validation_errors():
    cfg, _, rt_model = _make_models()
    rt = InferenceRuntime(cfg=cfg, model=rt_model, compile=False)
    try:
        inputs, offsets, indices = _make_batch(8, seed=4)

        bad_inputs = inputs.astype(np.float64)
        with pytest.raises(ValueError, match="float32"):
            rt.evaluate(bad_inputs, offsets, indices)

        bad_off_len = offsets[:-1]  # B instead of B+1
        with pytest.raises(ValueError, match="B\\+1"):
            rt.evaluate(inputs, bad_off_len, indices)

        bad_off_end = offsets.copy()
        bad_off_end[-1] -= 1  # offsets[-1] != K
        with pytest.raises(ValueError, match="CSR boundary"):
            rt.evaluate(inputs, bad_off_end, indices)

        bad_off_order = offsets.copy()
        bad_off_order[3] = bad_off_order[4] + 5  # non-monotonic
        with pytest.raises(ValueError, match="non-decreasing"):
            rt.evaluate(inputs, bad_off_order, indices)

        bad_idx = indices.copy()
        bad_idx[0] = POLICY_SIZE  # out of range
        with pytest.raises(ValueError, match="legal_indices"):
            rt.evaluate(inputs, offsets, bad_idx)
    finally:
        rt.close()
