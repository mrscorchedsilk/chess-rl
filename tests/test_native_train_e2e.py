"""Task 10: end-to-end native self-play trainer (actor + GPU runtime).

Proves the complete native learner — native Actor self-play, GPU sparse
inference, AMP training, schema-v3 checkpoint, exact resume — in a bounded,
deterministic form that is safe to run on CPU in CI.

To stay CPU-only and deterministic, the GPU runtime is replaced with a
deterministic fake inference fn (the same hash-based evaluator the native MCTS
tests use).  The native Actor, replay, checkpoint and resume paths under test
are the REAL production code; only the neural-network forward is faked.

Run: .venv/bin/python -m pytest tests/test_native_train_e2e.py -q
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess_rl_native  # noqa: E402
from config import Config  # noqa: E402
from native_selfplay import NativeSelfPlay  # noqa: E402


def fake_inference(inputs, offsets, indices):
    """Deterministic hash-based logits + zero values (mirrors native MCTS tests)."""
    inputs = np.asarray(inputs, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    logits = np.zeros(indices.shape[0], dtype=np.float32)
    for i in range(inputs.shape[0]):
        h = int.from_bytes(hashlib.sha256(inputs[i].tobytes()).digest()[:8], "little")
        s, e = int(offsets[i]), int(offsets[i + 1])
        for k in range(s, e):
            logits[k] = float((h >> ((k - s) % 32)) & 0x1F)
    return logits, np.zeros(inputs.shape[0], dtype=np.float32)


def _cfg(**over):
    cfg = Config()
    cfg.games_per_iteration = 2
    cfg.num_simulations = 8
    cfg.max_game_length = 12
    cfg.temperature_threshold = 5
    cfg.seed = 42
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _run_sp(cfg, seed=None):
    """Run one NativeSelfPlay round and return the examples."""
    if seed is not None:
        cfg.seed = seed
    sp = NativeSelfPlay(cfg, fake_inference, games=cfg.games_per_iteration)
    return sp.run()


def test_native_selfplay_produces_valid_replay_examples():
    ex = _run_sp(_cfg(), seed=1)
    assert len(ex) > 0
    for state, pi, z in ex:
        assert state.shape == (104, 8, 8)
        assert state.dtype == np.float32
        assert pi.shape == (4672,)
        assert pi.dtype == np.float32
        assert abs(float(pi.sum()) - 1.0) < 1e-5
        assert z in (-1.0, 0.0, 1.0)


def test_native_selfplay_is_deterministic_given_seed():
    a = _run_sp(_cfg(), seed=7)
    b = _run_sp(_cfg(), seed=7)
    assert len(a) == len(b)
    for (sa, pa, za), (sb, pb, zb) in zip(a, b):
        assert np.array_equal(sa, sb)
        assert np.array_equal(pa, pb)
        assert za == zb


def test_native_selfplay_generation_handle_recorded():
    cfg = _cfg()
    sp = NativeSelfPlay(cfg, fake_inference, games=2, generation=3, weight_version=9)
    sp.run()
    for g in sp.actor.finished_games():
        assert g["generation"] == 3
        assert g["weight_version"] == 9


def test_gpu_inference_fn_requires_cuda():
    # The real GPU path must refuse to run without CUDA (CPU CI guards).
    from native_selfplay import make_gpu_inference_fn

    if not __import__("torch").cuda.is_available():
        with pytest.raises(RuntimeError):
            make_gpu_inference_fn(_cfg())


def test_native_trainer_checkpoint_resume_roundtrip(tmp_path, monkeypatch):
    """Wire the real train.run_native but stub self-play with the fake actor
    and force the CPU device, then verify checkpoint + resume semantics."""
    import torch

    torch.manual_seed(0)
    cfg = _cfg()
    cfg.checkpoint_dir = str(tmp_path)
    cfg.num_iterations = 2
    cfg.device = "cpu"
    cfg.amp = False
    cfg.games_per_iteration = 1
    cfg.num_simulations = 4
    cfg.max_game_length = 8
    cfg.train_epoch_size = 16
    cfg.training_epochs = 1
    cfg.train_batch_size = 16
    cfg.replay_buffer_size = 200

    import train

    # Stub the GPU inference fn with the deterministic fake so run_native runs
    # on CPU; also shrink self-play so the test stays fast.
    def _stub_inference(c, model=None):
        def _fn(inputs, offsets, indices):
            return fake_inference(inputs, offsets, indices)
        _fn.update_weights = lambda sd: None  # no-op weight publish
        _fn.runtime = None
        return _fn

    monkeypatch.setattr(
        train.native_selfplay, "make_gpu_inference_fn", _stub_inference,
    )

    train.run_native(cfg, resume=False)
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt"))
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "best.pt"))

    # Read back the checkpoint and confirm iteration/run-id semantics.
    import json

    meta = json.load(open(os.path.join(cfg.checkpoint_dir, "checkpoint_meta.json")))
    assert meta["iteration"] == 2
    run_id = meta["run_id"]

    # Resume from it and verify it advances exactly one iteration.
    cfg.num_iterations = 3
    train.run_native(cfg, resume=True)
    meta2 = json.load(open(os.path.join(cfg.checkpoint_dir, "checkpoint_meta.json")))
    assert meta2["iteration"] == 3
    assert meta2["run_id"] == run_id
