"""Moves-left auxiliary head, end to end.

A KataGo-style auxiliary target: predict how many plies remain in the game.
It is never consulted by the search — its job is to regularise the shared
trunk, because "who is winning" and "how long until it is over" are learned
from the same features, and the extra signal speeds value convergence.

Design constraints this pins:

  * OFF by default.  Adding the head changes the state_dict, so a checkpoint
    written with it cannot load into a net without it.
  * ``forward`` returns two outputs unless the third is explicitly requested,
    so no existing call site changes.
  * The label needs no engine support: a finished game's examples are in ply
    order, so position i has ``len - 1 - i`` plies left.
  * A replay restored from a pre-auxiliary snapshot has an all-zero column and
    ``has_moves_left`` False; training must NOT fit that, or the head learns
    that every position ends immediately.

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_moves_left_head.py -q
"""
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
import native_selfplay as ns       # noqa: E402
import train                       # noqa: E402

from test_train_loop_v2 import synthetic_examples   # noqa: E402


def _cfg(**over):
    cfg = Config()
    cfg.num_res_blocks, cfg.num_filters = 1, 8
    cfg.device = "cpu"
    cfg.amp = False
    cfg.telemetry_enabled = False
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
#  the head                                                                   #
# --------------------------------------------------------------------------- #

def test_head_is_off_by_default():
    assert Config.moves_left_head is False
    net = ChessNet(_cfg())
    assert net.has_moves_left is False
    assert net.heads_id == "pv"


def test_head_adds_parameters_and_changes_the_head_id():
    plain = ChessNet(_cfg())
    with_head = ChessNet(_cfg(moves_left_head=True))
    assert with_head.parameter_count() > plain.parameter_count()
    assert with_head.heads_id == "pv+ml"


def test_forward_returns_two_outputs_unless_asked():
    net = ChessNet(_cfg(moves_left_head=True))
    out = net(torch.randn(3, 104, 8, 8))
    assert len(out) == 2
    out3 = net(torch.randn(3, 104, 8, 8), with_moves_left=True)
    assert len(out3) == 3


def test_predictions_are_non_negative():
    """Plies remaining cannot be negative; softplus enforces it."""
    net = ChessNet(_cfg(moves_left_head=True))
    _, _, ml = net(torch.randn(16, 104, 8, 8), with_moves_left=True)
    assert ml.shape == (16, 1)
    assert bool((ml >= 0).all())


def test_requesting_the_head_without_building_it_is_a_clear_error():
    net = ChessNet(_cfg())
    with pytest.raises(RuntimeError, match="no moves-left head"):
        net(torch.randn(2, 104, 8, 8), with_moves_left=True)


def test_body_identity_is_unchanged_by_the_head():
    """architecture_id names the BODY; the head set is heads_id."""
    plain = ChessNet(_cfg())
    with_head = ChessNet(_cfg(moves_left_head=True))
    assert plain.architecture_id == with_head.architecture_id


def test_state_dicts_are_incompatible_across_head_sets():
    """The reason heads_id exists: this must fail loudly, not silently."""
    plain = ChessNet(_cfg())
    with_head = ChessNet(_cfg(moves_left_head=True))
    with pytest.raises(RuntimeError):
        plain.load_state_dict(with_head.state_dict())


# --------------------------------------------------------------------------- #
#  the label                                                                  #
# --------------------------------------------------------------------------- #

def test_label_counts_down_to_zero_within_each_game():
    game = [(None, None, 0.0)] * 7
    got = [ml for _, _, _, ml in ns._with_moves_left(game, True)]
    assert got == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]


def test_single_position_game_is_labelled_zero():
    got = list(ns._with_moves_left([(None, None, 1.0)], True))
    assert [ml for _, _, _, ml in got] == [0.0]


def test_labels_are_not_emitted_when_the_head_is_off():
    """The (state, pi, z) example contract must not change for everyone else."""
    game = [(None, None, 0.0)] * 4
    got = list(ns._with_moves_left(game, False))
    assert got == game
    assert all(len(e) == 3 for e in got)


def _fake_eval(inputs, offsets, indices):
    return (np.zeros(len(indices), dtype=np.float32),
            np.zeros((len(offsets) - 1, 1), dtype=np.float32))


def test_selfplay_emits_three_tuples_by_default():
    cfg = _cfg(num_simulations=6, max_game_length=8)
    examples = ns.NativeSelfPlay(cfg, _fake_eval, games=2, seed=5).run()
    assert all(len(e) == 3 for e in examples)


def test_selfplay_emits_four_tuples_when_the_head_is_enabled():
    cfg = _cfg(num_simulations=6, max_game_length=8, moves_left_head=True)
    examples = ns.NativeSelfPlay(cfg, _fake_eval, games=2, seed=5).run()
    assert all(len(e) == 4 for e in examples)
    labels = [e[3] for e in examples]
    assert min(labels) == 0.0
    assert all(m >= 0 for m in labels)


def test_sharded_selfplay_honours_the_same_gate():
    cfg = _cfg(num_simulations=6, max_game_length=8, moves_left_head=True)
    examples = ns.ShardedSelfPlay(cfg, _fake_eval, games=4, shards=2,
                                  seed=5).run()
    assert all(len(e) == 4 for e in examples)
    off = _cfg(num_simulations=6, max_game_length=8)
    plain = ns.ShardedSelfPlay(off, _fake_eval, games=4, shards=2, seed=5).run()
    assert all(len(e) == 3 for e in plain)


# --------------------------------------------------------------------------- #
#  storage                                                                    #
# --------------------------------------------------------------------------- #

def _buf(cfg, n=64, with_labels=True, seed=0):
    buf = ReplayBuffer(256, cfg.policy_size, cfg.num_input_planes,
                       cfg.board_size)
    rows = synthetic_examples(n, seed=seed, policy_size=cfg.policy_size,
                              planes=cfg.num_input_planes,
                              board=cfg.board_size)
    if with_labels:
        rows = [(s, p, z, float(n - i - 1)) for i, (s, p, z) in enumerate(rows)]
    buf.extend(rows)
    return buf


def test_replay_accepts_both_arities():
    cfg = _cfg()
    with_labels = _buf(cfg, 8, with_labels=True)
    without = _buf(cfg, 8, with_labels=False)
    assert with_labels.has_moves_left is True
    assert without.has_moves_left is False
    assert len(with_labels) == len(without) == 8


def test_labels_survive_a_state_dict_round_trip():
    cfg = _cfg()
    buf = _buf(cfg, 16)
    restored = ReplayBuffer(256, cfg.policy_size, cfg.num_input_planes,
                            cfg.board_size)
    restored.load_state_dict(buf.state_dict())
    assert restored.has_moves_left is True
    assert torch.equal(restored.moves_left_tensor(range(16)),
                       buf.moves_left_tensor(range(16)))


def test_legacy_snapshot_without_the_key_still_loads():
    """Resume compatibility: pre-auxiliary checkpoints have no moves_left."""
    cfg = _cfg()
    sd = _buf(cfg, 12).state_dict()
    legacy = {k: v for k, v in sd.items()
              if k not in ("moves_left", "has_moves_left")}
    restored = ReplayBuffer(256, cfg.policy_size, cfg.num_input_planes,
                            cfg.board_size)
    restored.load_state_dict(legacy)
    assert len(restored) == 12
    assert restored.has_moves_left is False
    assert float(restored.moves_left_tensor([0]).item()) == 0.0


def test_mismatched_label_length_is_rejected():
    cfg = _cfg()
    sd = _buf(cfg, 12).state_dict()
    sd["moves_left"] = np.zeros(5, dtype=np.float32)
    restored = ReplayBuffer(256, cfg.policy_size, cfg.num_input_planes,
                            cfg.board_size)
    with pytest.raises(ValueError, match="moves_left"):
        restored.load_state_dict(sd)


# --------------------------------------------------------------------------- #
#  the loss                                                                   #
# --------------------------------------------------------------------------- #

def test_loss_is_reported_when_head_and_labels_are_both_present():
    cfg = _cfg(moves_left_head=True, train_epoch_size=16, train_batch_size=8,
               training_epochs=1, epochs_per_iteration=1)
    buf = _buf(cfg, 64)
    net = ChessNet(cfg)
    opt = train._new_optimizer(cfg, net)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=16)
    assert out["moves_left_head"] is True
    assert out["moves_left_loss"] is not None and out["moves_left_loss"] >= 0.0


def test_zero_label_column_is_not_trained_on():
    """A replay restored from a pre-auxiliary snapshot must be skipped."""
    cfg = _cfg(moves_left_head=True, train_epoch_size=16, train_batch_size=8,
               training_epochs=1, epochs_per_iteration=1)
    buf = _buf(cfg, 64, with_labels=False)
    net = ChessNet(cfg)
    opt = train._new_optimizer(cfg, net)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=16)
    assert out["moves_left_head"] is False
    assert out["moves_left_loss"] is None


def test_labels_without_a_head_are_simply_ignored():
    cfg = _cfg(train_epoch_size=16, train_batch_size=8, training_epochs=1,
               epochs_per_iteration=1)
    buf = _buf(cfg, 64)
    net = ChessNet(cfg)
    opt = train._new_optimizer(cfg, net)
    out = train._epoch_train(cfg, net, opt, buf, "cpu", None,
                             positions_generated=16)
    assert out["moves_left_head"] is False
    assert out["steps"] > 0


def test_auxiliary_loss_actually_reaches_the_head_parameters():
    cfg = _cfg(moves_left_head=True, train_epoch_size=16, train_batch_size=8,
               training_epochs=1, epochs_per_iteration=1,
               moves_left_loss_weight=1.0)
    buf = _buf(cfg, 64)
    net = ChessNet(cfg)
    before = net.moves_left_fc2.weight.detach().clone()
    opt = train._new_optimizer(cfg, net)
    train._epoch_train(cfg, net, opt, buf, "cpu", None, positions_generated=16)
    assert not torch.equal(before, net.moves_left_fc2.weight.detach())
