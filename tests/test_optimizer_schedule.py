"""Optimizer choice and the learning-rate schedule.

Two gaps this closes:

  * the optimizer was ``torch.optim.Adam(weight_decay=...)`` — coupled L2
    folded into the adaptive step, not true weight decay;
  * there was NO scheduler anywhere in the tree.  The rate sat at 1e-3 for
    every step of a 2,300-iteration run, which caps final strength however
    good the rest of the pipeline gets.

The schedule is deliberately a PURE FUNCTION of (cfg, global optimizer step).
That is what makes resume exact: there is no scheduler object to serialise, so
restoring ``optimizer_steps`` restores the rate, and a resumed run cannot
drift from an uninterrupted one.

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_optimizer_schedule.py -q
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config          # noqa: E402
from model import ChessNet         # noqa: E402
from replay import ReplayBuffer    # noqa: E402
import train                       # noqa: E402

from test_train_loop_v2 import make_cfg, synthetic_examples  # noqa: E402


def _net_cfg(**over):
    cfg = Config()
    cfg.num_res_blocks, cfg.num_filters = 1, 4
    cfg.device = "cpu"
    cfg.amp = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
#  optimizer selection                                                        #
# --------------------------------------------------------------------------- #

def test_default_optimizer_is_adamw():
    assert Config.optimizer == "adamw"


@pytest.mark.parametrize("kind,cls", [
    ("adam", torch.optim.Adam),
    ("adamw", torch.optim.AdamW),
    ("sgd", torch.optim.SGD),
])
def test_optimizer_kinds_build(kind, cls):
    cfg = _net_cfg(optimizer=kind)
    opt = train._new_optimizer(cfg, ChessNet(cfg))
    assert isinstance(opt, cls)
    assert opt.param_groups[0]["weight_decay"] == cfg.weight_decay


def test_sgd_uses_momentum_and_nesterov():
    cfg = _net_cfg(optimizer="sgd", sgd_momentum=0.9)
    opt = train._new_optimizer(cfg, ChessNet(cfg))
    assert opt.param_groups[0]["momentum"] == 0.9
    assert opt.param_groups[0]["nesterov"] is True


def test_unknown_optimizer_raises_rather_than_falling_back():
    """A typo must not silently change the optimizer of a long run."""
    cfg = _net_cfg(optimizer="adamwww")
    with pytest.raises(ValueError, match="unknown cfg.optimizer"):
        train._new_optimizer(cfg, ChessNet(cfg))


# --------------------------------------------------------------------------- #
#  schedule shape                                                             #
# --------------------------------------------------------------------------- #

def test_none_schedule_is_constant():
    cfg = _net_cfg(lr_schedule="none", learning_rate=3e-4)
    assert {train._lr_at_step(cfg, s) for s in (0, 10, 10_000)} == {3e-4}


def test_warmup_rises_to_base_and_never_starts_at_zero():
    cfg = _net_cfg(lr_schedule="cosine", learning_rate=1e-3,
                   lr_warmup_steps=100, lr_total_steps=10_000, lr_min=1e-5)
    assert train._lr_at_step(cfg, 0) > 0.0
    assert train._lr_at_step(cfg, 0) < train._lr_at_step(cfg, 50)
    assert train._lr_at_step(cfg, 99) == pytest.approx(1e-3)


def test_cosine_decays_monotonically_after_warmup():
    cfg = _net_cfg(lr_schedule="cosine", learning_rate=1e-3,
                   lr_warmup_steps=100, lr_total_steps=10_000, lr_min=1e-5)
    steps = list(range(100, 10_001, 250))
    rates = [train._lr_at_step(cfg, s) for s in steps]
    assert all(b <= a + 1e-12 for a, b in zip(rates, rates[1:]))
    assert rates[0] == pytest.approx(1e-3)
    # lr_min is reached exactly AT the horizon, not at the last sampled step.
    assert train._lr_at_step(cfg, 10_000) == pytest.approx(1e-5, abs=1e-9)


def test_cosine_is_clamped_past_the_horizon():
    cfg = _net_cfg(lr_schedule="cosine", learning_rate=1e-3,
                   lr_warmup_steps=10, lr_total_steps=1_000, lr_min=1e-5)
    assert train._lr_at_step(cfg, 5_000) == pytest.approx(1e-5, abs=1e-9)


def test_step_schedule_multiplies_by_gamma():
    cfg = _net_cfg(lr_schedule="step", learning_rate=1e-3, lr_warmup_steps=0,
                   lr_step_size=1000, lr_step_gamma=0.1)
    assert train._lr_at_step(cfg, 0) == pytest.approx(1e-3)
    assert train._lr_at_step(cfg, 1000) == pytest.approx(1e-4)
    assert train._lr_at_step(cfg, 2000) == pytest.approx(1e-5)


def test_unknown_schedule_raises():
    cfg = _net_cfg(lr_schedule="triangular", lr_warmup_steps=0)
    with pytest.raises(ValueError, match="unknown cfg.lr_schedule"):
        train._lr_at_step(cfg, 5)


# --------------------------------------------------------------------------- #
#  exact resume                                                               #
# --------------------------------------------------------------------------- #

def test_schedule_depends_only_on_step_not_history():
    """Resuming at step N must give the same rate as never stopping."""
    cfg = _net_cfg(lr_schedule="cosine", lr_warmup_steps=50,
                   lr_total_steps=5_000)
    uninterrupted = [train._lr_at_step(cfg, s) for s in range(0, 2_000, 137)]
    resumed = [train._lr_at_step(_net_cfg(lr_schedule="cosine",
                                          lr_warmup_steps=50,
                                          lr_total_steps=5_000), s)
               for s in range(0, 2_000, 137)]
    assert uninterrupted == resumed


def test_apply_lr_sets_every_param_group():
    cfg = _net_cfg(lr_schedule="cosine", lr_warmup_steps=10,
                   lr_total_steps=1_000)
    net = ChessNet(cfg)
    opt = torch.optim.AdamW([
        {"params": list(net.parameters())[:1]},
        {"params": list(net.parameters())[1:]},
    ], lr=1.0)
    applied = train._apply_lr(cfg, opt, 500)
    assert all(g["lr"] == applied for g in opt.param_groups)
    assert applied == train._lr_at_step(cfg, 500)


def test_epoch_train_advances_the_rate_across_steps():
    cfg = _net_cfg(lr_schedule="cosine", learning_rate=1e-3,
                   lr_warmup_steps=1_000, lr_total_steps=100_000,
                   train_epoch_size=32, train_batch_size=8, training_epochs=1,
                   epochs_per_iteration=1, replay_buffer_size=256,
                   telemetry_enabled=False)
    buf = ReplayBuffer(cfg.replay_buffer_size, cfg.policy_size,
                       cfg.num_input_planes, cfg.board_size)
    buf.extend(synthetic_examples(100, seed=3, policy_size=cfg.policy_size,
                                  planes=cfg.num_input_planes,
                                  board=cfg.board_size))
    net = ChessNet(cfg)
    opt = train._new_optimizer(cfg, net)
    early = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                               optimizer_steps_total=0, positions_generated=32)
    late = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                              optimizer_steps_total=900, positions_generated=32)
    assert late["learning_rate"] > early["learning_rate"], "still in warmup"
    assert early["learning_rate"] < cfg.learning_rate


# --------------------------------------------------------------------------- #
#  checkpoint compatibility                                                   #
# --------------------------------------------------------------------------- #

def test_checkpoint_records_optimizer_and_schedule(tmp_path, monkeypatch):
    from test_train_loop_v2 import make_fake_selfplay, load_latest
    cfg = make_cfg(tmp_path, num_iterations=1, arena_every=10_000,
                   optimizer="adamw", lr_schedule="cosine")
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(8, seed=1)])
    train.run(cfg, resume=False)
    payload = load_latest(cfg)
    assert payload["optimizer_kind"] == "adamw"
    assert payload["lr_schedule"] == "cosine"


def test_resume_under_a_different_optimizer_is_refused(tmp_path, monkeypatch):
    """Adam and AdamW load each other's state silently; that must not happen."""
    from test_train_loop_v2 import make_fake_selfplay
    cfg = make_cfg(tmp_path, num_iterations=1, arena_every=10_000,
                   optimizer="adam")
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(8, seed=1)])
    train.run(cfg, resume=False)

    switched = make_cfg(tmp_path, num_iterations=2, arena_every=10_000,
                        optimizer="adamw")
    switched.checkpoint_dir = cfg.checkpoint_dir
    with pytest.raises(train.IncompatibleCheckpointError, match="optimizer"):
        train._load_latest_v2(switched)


def test_optimizer_change_can_be_opted_into(tmp_path, monkeypatch):
    from test_train_loop_v2 import make_fake_selfplay
    cfg = make_cfg(tmp_path, num_iterations=1, arena_every=10_000,
                   optimizer="adam")
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(8, seed=1)])
    train.run(cfg, resume=False)

    switched = make_cfg(tmp_path, num_iterations=2, arena_every=10_000,
                        optimizer="adamw", allow_optimizer_change=True)
    switched.checkpoint_dir = cfg.checkpoint_dir
    assert train._load_latest_v2(switched) is not None


def test_legacy_checkpoints_without_the_field_are_treated_as_adam(tmp_path):
    """Pre-existing v2 snapshots carry no optimizer_kind and were Adam."""
    cfg = make_cfg(tmp_path, optimizer="adam")
    payload = {"optimizer_kind": None}
    train._validate_optimizer_compat(cfg, payload, "x.pt")   # must not raise
    cfg2 = make_cfg(tmp_path, optimizer="adamw")
    with pytest.raises(train.IncompatibleCheckpointError):
        train._validate_optimizer_compat(cfg2, payload, "x.pt")
