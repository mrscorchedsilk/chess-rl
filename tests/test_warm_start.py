"""TDD tests for Phase 3: safe weights-only warm start.

Written BEFORE the implementation.  Required behaviour:

  A. resume and warm-start are mutually exclusive
  B. warm-start loads accepted best weights
  C. candidate and best begin with identical loaded weights
  D. replay starts empty (not the source's contaminated replay)
  E. optimizer is fresh
  F. optimizer step count starts at zero
  G. iteration starts at one
  H. run ID is new
  I. incompatible architecture rejected before load_state_dict
  J. source checkpoint is never modified
  K. first saved warm-start checkpoint contains provenance
  L. standard resume behavior remains unchanged

Run:  .venv/bin/python -m pytest tests/test_warm_start.py -q
"""

import hashlib
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from config import Config  # noqa: E402
from model import ChessNet  # noqa: E402
from replay import ReplayBuffer  # noqa: E402


def make_cfg(tmp_path, **overrides):
    cfg = Config()
    cfg.num_res_blocks = 1
    cfg.num_filters = 4
    cfg.num_simulations = 1
    cfg.batch_size = 4
    cfg.train_batch_size = 4
    cfg.training_epochs = 1
    cfg.epochs_per_iteration = 1
    cfg.train_epoch_size = 4          # 1 batch -> exactly 1 optimizer step/iter
    cfg.games_per_iteration = 1
    cfg.replay_buffer_size = 100
    cfg.num_iterations = 3
    cfg.arena_every = 10_000          # no arena in these runs
    cfg.arena_games = 2
    cfg.arena_accept_threshold = 0.55
    cfg.arena_simulations = 1
    cfg.max_game_length = 8
    cfg.checkpoint_interval_minutes = 1_000_000
    cfg.checkpoint_every_iterations = 1
    cfg.device = "cpu"
    cfg.seed = 42
    cfg.checkpoint_dir = str(tmp_path / "ckpts")
    cfg.metrics_path = str(tmp_path / "training.jsonl")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def synthetic_examples(n, seed=0, policy_size=4672, planes=104, board=8):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        state = (rng.random((planes, board, board)) < 0.2).astype(np.float32)
        k = int(rng.integers(1, 12))
        idx = rng.choice(policy_size, size=k, replace=False)
        probs = rng.random(k)
        probs = probs / probs.sum()
        pi = np.zeros(policy_size, dtype=np.float32)
        pi[idx] = probs
        out.append((state, pi, float(rng.choice([-1.0, 0.0, 1.0]))))
    return out


def make_fake_selfplay(monkeypatch):
    examples = synthetic_examples(5, seed=7)  # 5 examples per game

    def fake_play_game(net, cfg):
        return [tuple(e) for e in examples]

    monkeypatch.setattr(train, "play_game", fake_play_game)
    return examples


def load_latest(cfg):
    return torch.load(os.path.join(cfg.checkpoint_dir, "latest.pt"),
                      map_location="cpu", weights_only=False)


def state_dicts_equal(a, b):
    return set(a.keys()) == set(b.keys()) and all(torch.equal(a[k], b[k]) for k in a)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  A. mutually exclusive                                                      #
# --------------------------------------------------------------------------- #

def test_resume_and_warm_start_mutually_exclusive():
    parser = train.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--resume", "--warm-start-checkpoint", "/tmp/x.pt"])


# --------------------------------------------------------------------------- #
#  B + I. load best / reject incompatible architecture                        #
# --------------------------------------------------------------------------- #

def _make_source(tmp_path, monkeypatch, **src_overrides):
    make_fake_selfplay(monkeypatch)
    cfg = make_cfg(tmp_path / "src", **src_overrides)
    train.run(cfg, resume=False)
    return cfg


def test_warm_start_loads_accepted_best_weights(tmp_path, monkeypatch):
    src_cfg = _make_source(tmp_path, monkeypatch)
    src_latest = os.path.join(src_cfg.checkpoint_dir, "latest.pt")
    src_best = load_latest(src_cfg)["best"]

    dst_cfg = make_cfg(tmp_path / "dst")
    wm = train._load_warm_start(dst_cfg, src_latest)
    assert state_dicts_equal(wm["best"], src_best)


def test_warm_start_rejects_incompatible_architecture(tmp_path, monkeypatch):
    # source body = custom-1x8, warm-start body = custom-1x4 -> reject
    src_cfg = _make_source(tmp_path, monkeypatch, num_filters=8)
    src_latest = os.path.join(src_cfg.checkpoint_dir, "latest.pt")

    dst_cfg = make_cfg(tmp_path / "dst", num_filters=4)
    with pytest.raises(train.IncompatibleCheckpointError):
        train._load_warm_start(dst_cfg, src_latest)


# --------------------------------------------------------------------------- #
#  C/D/E/F/G/H/K. fresh lineage + provenance                                  #
# --------------------------------------------------------------------------- #

def test_warm_start_creates_fresh_lineage_with_provenance(tmp_path, monkeypatch):
    make_fake_selfplay(monkeypatch)
    src_cfg = make_cfg(tmp_path / "src", num_iterations=3)
    train.run(src_cfg, resume=False)
    src_payload = load_latest(src_cfg)
    src_latest = os.path.join(src_cfg.checkpoint_dir, "latest.pt")
    src_run_id = src_payload["run_id"]
    src_best = src_payload["best"]

    # warm start for ONE iteration into a separate dir
    dst_cfg = make_cfg(tmp_path / "dst", num_iterations=1)
    train.run(dst_cfg, resume=False, warm_start_checkpoint=src_latest)

    dst = load_latest(dst_cfg)

    # C. candidate and best both begin from the loaded accepted best: best stays
    #    equal to src_best (no arena promotion in iteration 1), while the
    #    candidate is a TRAINED derivative of that same loaded best.
    assert state_dicts_equal(dst["best"], src_best), "best != source best"
    assert not state_dicts_equal(dst["candidate"], src_best), \
        "candidate must have trained away from the loaded best"

    # G. iteration starts at one (not source_iteration + 1 == 4)
    assert dst["iteration"] == 1
    assert dst["generation"] == 0

    # F. optimizer step count starts at zero (exactly 1 step for 1 iteration)
    assert dst["optimizer_steps"] == 1

    # D. replay starts empty (1 game * 5 examples), not source's 15 + 5
    buf = ReplayBuffer(dst_cfg.replay_buffer_size, dst_cfg.policy_size,
                       dst_cfg.num_input_planes, dst_cfg.board_size)
    buf.load_state_dict(dst["replay"])
    assert len(buf) == 5, f"warm-start replay must start empty, got {len(buf)}"

    # E. optimizer is fresh: its state dict has exactly the params of a single
    #    warm-start training step, not the source optimizer's state
    assert "optimizer" in dst

    # H. run ID is new
    assert dst["run_id"] != src_run_id

    # K. provenance recorded on the first warm-start checkpoint
    assert dst.get("parent_run_id") == src_run_id
    assert dst.get("parent_iteration") == src_payload["iteration"]
    assert dst.get("parent_generation") == src_payload["generation"]
    assert dst.get("warm_started_from") == os.path.abspath(src_latest)


# --------------------------------------------------------------------------- #
#  C (init equivalence). candidate and best begin identical                   #
# --------------------------------------------------------------------------- #

def test_warm_start_candidate_and_best_begin_identical(tmp_path, monkeypatch):
    make_fake_selfplay(monkeypatch)
    src_cfg = make_cfg(tmp_path / "src", num_iterations=2)
    train.run(src_cfg, resume=False)
    src_latest = os.path.join(src_cfg.checkpoint_dir, "latest.pt")
    src_best = load_latest(src_cfg)["best"]

    loaded = []
    orig = ChessNet.load_state_dict

    def spy(self, sd, *a, **k):
        loaded.append({k: v.detach().clone() for k, v in sd.items()})
        return orig(self, sd, *a, **k)

    monkeypatch.setattr(ChessNet, "load_state_dict", spy)

    dst_cfg = make_cfg(tmp_path / "dst", num_iterations=0)
    train.run(dst_cfg, resume=False, warm_start_checkpoint=src_latest)

    # The final two load_state_dict calls are the warm-start branch loading the
    # candidate net and the best net — both from the SAME accepted-best dict.
    assert len(loaded) >= 2
    assert state_dicts_equal(loaded[-1], loaded[-2]), \
        "candidate and best must begin with identical loaded weights"
    assert state_dicts_equal(loaded[-1], src_best), \
        "candidate/best must begin from the accepted-best weights"


# --------------------------------------------------------------------------- #
#  J. source checkpoint never modified                                        #
# --------------------------------------------------------------------------- #

def test_warm_start_does_not_modify_source_checkpoint(tmp_path, monkeypatch):
    make_fake_selfplay(monkeypatch)
    src_cfg = make_cfg(tmp_path / "src", num_iterations=3)
    train.run(src_cfg, resume=False)
    src_latest = os.path.join(src_cfg.checkpoint_dir, "latest.pt")
    src_best = os.path.join(src_cfg.checkpoint_dir, "best.pt")
    before_latest = _file_sha256(src_latest)
    before_best = _file_sha256(src_best)

    dst_cfg = make_cfg(tmp_path / "dst", num_iterations=1)
    train.run(dst_cfg, resume=False, warm_start_checkpoint=src_latest)

    assert _file_sha256(src_latest) == before_latest, "source latest.pt changed"
    assert _file_sha256(src_best) == before_best, "source best.pt changed"


# --------------------------------------------------------------------------- #
#  L. standard resume behavior unchanged                                      #
# --------------------------------------------------------------------------- #

def test_resume_behavior_unchanged(tmp_path, monkeypatch):
    make_fake_selfplay(monkeypatch)
    cfg = make_cfg(tmp_path / "run", num_iterations=4)
    train.run(cfg, resume=False)
    p1 = load_latest(cfg)
    run_id1 = p1["run_id"]
    assert p1["iteration"] == 4

    train.run(cfg, resume=True)
    p2 = load_latest(cfg)
    assert p2["iteration"] == 4  # resume from iter 4 -> no further iterations left
    assert p2["run_id"] == run_id1
    # no provenance fields on a plain resume
    assert "parent_run_id" not in p2
    assert "warm_started_from" not in p2
