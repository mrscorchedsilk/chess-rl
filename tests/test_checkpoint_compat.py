"""Checkpoint/resume compatibility across this branch's changes.

This branch adds config fields (optimizer, lr_schedule, replay preflight,
augmentation, shards, moves-left head) and two new checkpoint keys.  None of
that may invalidate a snapshot written before it — the live v2 lineage is
2,300 iterations deep and its `latest.pt` has to keep resuming.

The specific hazards:

  * CRITICAL_CONFIG_KEYS feeds a fingerprint that a resume validates against.
    Adding new fields to that tuple would change every existing fingerprint
    and reject every existing checkpoint.
  * Adam and AdamW share a state_dict layout, so the optimizer guard must
    treat a checkpoint with NO optimizer_kind as "adam" (which it was) rather
    than as "whatever the config now says".
  * ReplayBuffer.load_state_dict must tolerate a snapshot with no moves_left
    column.

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_checkpoint_compat.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config   # noqa: E402
import train                # noqa: E402

from test_train_loop_v2 import (   # noqa: E402
    load_latest, make_cfg, make_fake_arena, make_fake_selfplay,
    state_dicts_equal, synthetic_examples,
)


NEW_CONFIG_FIELDS = (
    "optimizer", "lr_schedule", "lr_warmup_steps", "lr_min", "lr_total_steps",
    "train_epoch_size", "train_channels_last", "train_prefetch",
    "augment_colour_flip", "selfplay_shards", "selfplay_games_in_flight",
    "selfplay_leaves_per_game", "selfplay_max_batch", "moves_left_head",
    "resign_enabled", "draw_adjudication_enabled", "replay_preflight",
    "arena_confidence", "arena_require_lower_bound",
)


# --------------------------------------------------------------------------- #
#  the fingerprint                                                            #
# --------------------------------------------------------------------------- #

def test_new_config_fields_are_not_in_the_critical_set():
    """Adding any of these to CRITICAL_CONFIG_KEYS invalidates every
    existing checkpoint, because the fingerprint is validated on resume."""
    for field in NEW_CONFIG_FIELDS:
        assert field not in train.CRITICAL_CONFIG_KEYS, (
            f"{field} is in CRITICAL_CONFIG_KEYS; that changes the "
            "fingerprint of every checkpoint written before it existed"
        )


def test_fingerprint_is_unchanged_by_the_new_fields():
    base = Config()
    changed = Config()
    changed.optimizer = "sgd"
    changed.lr_schedule = "step"
    changed.train_epoch_size = 99_999
    changed.selfplay_shards = 7
    changed.moves_left_head = True
    changed.augment_colour_flip = 0.0
    assert train._config_fingerprint(base) == train._config_fingerprint(changed)


def test_fingerprint_still_reacts_to_genuinely_critical_changes():
    """The guard must not have been weakened into uselessness."""
    base = Config()
    for field, value in (("num_filters", 999), ("train_batch_size", 7),
                         ("policy_size", 123), ("replay_buffer_size", 11)):
        other = Config()
        setattr(other, field, value)
        assert train._config_fingerprint(base) != train._config_fingerprint(other), field


# --------------------------------------------------------------------------- #
#  resuming a snapshot written before these features existed                  #
# --------------------------------------------------------------------------- #

def _write_run(tmp_path, **cfg_over):
    cfg_over.setdefault("num_iterations", 2)
    cfg_over.setdefault("arena_every", 10_000)
    return make_cfg(tmp_path, **cfg_over)


def test_legacy_snapshot_without_new_keys_resumes(tmp_path, monkeypatch):
    cfg = _write_run(tmp_path, optimizer="adam")
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=3)])
    train.run(cfg, resume=False)

    path = train._checkpoint_paths(cfg)["latest"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # simulate a snapshot from before this branch
    for key in ("optimizer_kind", "lr_schedule"):
        payload.pop(key, None)
    payload["replay"].pop("moves_left", None)
    payload["replay"].pop("has_moves_left", None)
    torch.save(payload, path)

    resumed = train._load_latest_v2(cfg)
    assert resumed is not None
    assert resumed["iteration"] == payload["iteration"]
    assert resumed["run_id"] == payload["run_id"]


def test_legacy_snapshot_resumes_into_a_continuing_run(tmp_path, monkeypatch):
    cfg = _write_run(tmp_path, optimizer="adam", num_iterations=2)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=4)])
    train.run(cfg, resume=False)
    first = load_latest(cfg)

    path = train._checkpoint_paths(cfg)["latest"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("optimizer_kind", None)
    payload["replay"].pop("moves_left", None)
    payload["replay"].pop("has_moves_left", None)
    torch.save(payload, path)

    cfg2 = _write_run(tmp_path, optimizer="adam", num_iterations=4)
    cfg2.checkpoint_dir = cfg.checkpoint_dir
    cfg2.metrics_path = cfg.metrics_path
    train.run(cfg2, resume=True)
    second = load_latest(cfg2)

    assert second["run_id"] == first["run_id"], "resume must keep the lineage"
    assert second["iteration"] > first["iteration"]
    assert second["generation"] >= first["generation"]


def test_replay_survives_the_legacy_round_trip(tmp_path, monkeypatch):
    cfg = _write_run(tmp_path, optimizer="adam")
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(12, seed=5)])
    train.run(cfg, resume=False)
    path = train._checkpoint_paths(cfg)["latest"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n = len(payload["replay"]["z"])
    payload["replay"].pop("moves_left", None)
    payload["replay"].pop("has_moves_left", None)
    torch.save(payload, path)

    from replay import ReplayBuffer
    buf = ReplayBuffer(cfg.replay_buffer_size, cfg.policy_size,
                       cfg.num_input_planes, cfg.board_size)
    buf.load_state_dict(train._load_latest_v2(cfg)["replay"])
    assert len(buf) == n
    assert buf.has_moves_left is False


# --------------------------------------------------------------------------- #
#  the new keys round-trip                                                    #
# --------------------------------------------------------------------------- #

def test_new_keys_are_written_and_read_back(tmp_path, monkeypatch):
    cfg = _write_run(tmp_path, optimizer="adamw", lr_schedule="cosine")
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=6)])
    train.run(cfg, resume=False)
    payload = load_latest(cfg)
    assert payload["optimizer_kind"] == "adamw"
    assert payload["lr_schedule"] == "cosine"
    resumed = train._load_latest_v2(cfg)
    assert resumed["optimizer_kind"] == "adamw"


def test_weights_round_trip_exactly(tmp_path, monkeypatch):
    cfg = _write_run(tmp_path, optimizer="adamw")
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=7)])
    train.run(cfg, resume=False)
    payload = load_latest(cfg)
    from model import ChessNet
    net = ChessNet(cfg)
    net.load_state_dict(payload["candidate"])
    assert state_dicts_equal(net.state_dict(), payload["candidate"])


def test_channels_last_training_does_not_change_the_state_dict_keys(tmp_path):
    """channels_last changes strides, not the checkpoint contract."""
    from model import ChessNet
    cfg = Config()
    cfg.num_res_blocks, cfg.num_filters = 1, 8
    plain = ChessNet(cfg)
    nhwc = ChessNet(cfg).to(memory_format=torch.channels_last)
    assert set(plain.state_dict()) == set(nhwc.state_dict())
    plain.load_state_dict(nhwc.state_dict())
