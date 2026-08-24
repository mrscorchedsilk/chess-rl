"""Training-rate contract: `train_epoch_size` is explicit, and every iteration
reports how much learning actually happened.

Background: `_epoch_train` only ever read the epoch bound through
``getattr(cfg, "train_epoch_size", 0)``, and no ``Config`` ever defined that
attribute.  The bound therefore always fell through to
``train_batch_size * training_epochs`` == 768, i.e. 3 batches per epoch and
9 optimizer steps per iteration regardless of replay size — roughly one
gradient step per 260 freshly generated positions.  These tests pin the fix so
the collapse cannot silently return.

CPU-only; a tiny net and a synthetic replay buffer, no real training run.

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_training_rate.py -q
"""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config              # noqa: E402
from model import ChessNet             # noqa: E402
from replay import ReplayBuffer        # noqa: E402
import train                           # noqa: E402

from test_train_loop_v2 import synthetic_examples  # noqa: E402


def _tiny_net_cfg(tmp_path, **overrides):
    cfg = Config()
    cfg.num_res_blocks = 1
    cfg.num_filters = 4
    cfg.device = "cpu"
    cfg.amp = False
    cfg.checkpoint_dir = str(tmp_path / "ckpts")
    cfg.telemetry_path = str(tmp_path / "telemetry.jsonl")
    cfg.telemetry_enabled = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _filled_buffer(cfg, n, seed=0):
    buf = ReplayBuffer(cfg.replay_buffer_size, cfg.policy_size,
                       cfg.num_input_planes, cfg.board_size)
    buf.extend(synthetic_examples(n, seed=seed, policy_size=cfg.policy_size,
                                  planes=cfg.num_input_planes,
                                  board=cfg.board_size))
    return buf


# --------------------------------------------------------------------------- #
#  the defect itself                                                          #
# --------------------------------------------------------------------------- #

def test_config_defines_train_epoch_size_explicitly():
    """It must be a real attribute, not a getattr fallback."""
    assert "train_epoch_size" in vars(Config), (
        "train_epoch_size must be defined on Config, not left to getattr()"
    )
    assert Config.train_epoch_size == 8192


def test_default_config_yields_96_optimizer_steps_per_iteration():
    """8192 / 256 = 32 batches per epoch, 3 epochs -> 96 steps (was 9)."""
    cfg = Config()
    batches_per_epoch = math.ceil(cfg.train_epoch_size / cfg.train_batch_size)
    assert batches_per_epoch * cfg.training_epochs == 96


def test_epoch_bound_no_longer_collapses_to_one_batch_per_epoch():
    """The old fallback made sample_size == train_batch_size * epochs."""
    cfg = Config()
    collapsed = cfg.train_batch_size * cfg.training_epochs
    assert cfg.train_epoch_size > collapsed


# --------------------------------------------------------------------------- #
#  realised behaviour of _epoch_train                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("epoch_size,epochs,batch,expected_steps", [
    (8, 2, 4, 4),
    (64, 3, 8, 24),
    (10, 1, 4, 3),      # ceil: a ragged tail batch still steps
])
def test_optimizer_steps_follow_train_epoch_size(
    tmp_path, epoch_size, epochs, batch, expected_steps
):
    cfg = _tiny_net_cfg(tmp_path, train_epoch_size=epoch_size,
                        training_epochs=epochs, epochs_per_iteration=epochs,
                        train_batch_size=batch, replay_buffer_size=256)
    buf = _filled_buffer(cfg, 200)
    net = ChessNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=50)
    assert out["steps"] == expected_steps
    assert out["batches"] == expected_steps


def test_sample_size_is_clamped_by_replay_size(tmp_path):
    """A buffer smaller than train_epoch_size bounds the epoch, without error."""
    cfg = _tiny_net_cfg(tmp_path, train_epoch_size=8192, training_epochs=1,
                        epochs_per_iteration=1, train_batch_size=8,
                        replay_buffer_size=64)
    buf = _filled_buffer(cfg, 20)
    net = ChessNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=20)
    assert out["sample_size"] == 20
    assert out["steps"] == math.ceil(20 / 8)


def test_training_rate_accounting_is_reported(tmp_path):
    cfg = _tiny_net_cfg(tmp_path, train_epoch_size=32, training_epochs=2,
                        epochs_per_iteration=2, train_batch_size=8,
                        replay_buffer_size=256)
    buf = _filled_buffer(cfg, 200)
    net = ChessNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=16)
    assert out["sample_size"] == 32
    assert out["positions_trained"] == 64        # sample_size * epochs
    assert out["positions_generated"] == 16
    assert out["sample_reuse"] == pytest.approx(4.0)


def test_sample_reuse_is_zero_not_infinite_when_nothing_generated(tmp_path):
    """Division guard: a zero-generation iteration must not raise or emit inf."""
    cfg = _tiny_net_cfg(tmp_path, train_epoch_size=16, training_epochs=1,
                        epochs_per_iteration=1, train_batch_size=8,
                        replay_buffer_size=256)
    buf = _filled_buffer(cfg, 100)
    net = ChessNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=0)
    assert out["sample_reuse"] == 0.0
    assert math.isfinite(out["sample_reuse"])


def test_empty_buffer_returns_full_accounting_keys(tmp_path):
    """Callers read these keys unconditionally; the early return must supply them."""
    cfg = _tiny_net_cfg(tmp_path, replay_buffer_size=64)
    buf = ReplayBuffer(cfg.replay_buffer_size, cfg.policy_size,
                       cfg.num_input_planes, cfg.board_size)
    net = ChessNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=7)
    for key in ("steps", "batches", "sample_size", "positions_trained",
                "positions_generated", "sample_reuse"):
        assert key in out
    assert out["steps"] == 0
    assert out["positions_generated"] == 7


# --------------------------------------------------------------------------- #
#  telemetry surface                                                          #
# --------------------------------------------------------------------------- #

def test_training_phase_telemetry_carries_rate_fields(tmp_path):
    cfg = _tiny_net_cfg(tmp_path, train_epoch_size=32, training_epochs=2,
                        epochs_per_iteration=2, train_batch_size=8,
                        replay_buffer_size=256)
    buf = _filled_buffer(cfg, 200)
    net = ChessNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train._epoch_train(cfg, net, opt, buf, "cpu", None,
                       run_id="testrun", iteration=5, generation=1,
                       optimizer_steps_total=100, positions_generated=16)

    records = [json.loads(line)
               for line in open(cfg.telemetry_path)
               if line.strip()]
    phase = [r for r in records
             if r.get("type") == "phase" and r.get("phase") == "training"]
    assert phase, "no training phase record emitted"
    rec = phase[-1]
    assert rec["train_epoch_size"] == 32
    assert rec["sample_size"] == 32
    assert rec["epochs"] == 2
    assert rec["positions_generated"] == 16
    assert rec["positions_trained"] == 64
    assert rec["sample_reuse"] == pytest.approx(4.0)
    assert rec["steps_per_iteration"] == rec["steps"] == 8
    assert rec["replay_size"] == 200


def test_telemetry_failure_never_breaks_training(tmp_path, monkeypatch):
    """Observability is swallow-guarded; a broken emitter must not stop a step."""
    cfg = _tiny_net_cfg(tmp_path, train_epoch_size=16, training_epochs=1,
                        epochs_per_iteration=1, train_batch_size=8,
                        replay_buffer_size=256)
    buf = _filled_buffer(cfg, 100)
    net = ChessNet(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def boom(*a, **k):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(train.telemetry, "emit", boom)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=8)
    assert out["steps"] == 2
