"""Milestone capture: every arena-accepted champion is persisted immutably.

Verifies the checkpoint-retention change: on arena acceptance the trainer now
writes a weights-only milestone (``milestones/best-genNNNN-iterNNNN-<run_id>.pt``)
plus a JSON sidecar recording arena provenance and architecture identity, so
historical champions survive snapshot pruning.  CPU-only; self-play and arena
are fakes; the network is tiny.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from config import Config  # noqa: E402
from model import ChessNet  # noqa: E402


def make_cfg(tmp_path, **overrides):
    cfg = Config()
    cfg.num_res_blocks = 1
    cfg.num_filters = 4
    cfg.num_simulations = 1
    cfg.batch_size = 4
    cfg.train_batch_size = 4
    cfg.training_epochs = 2
    cfg.epochs_per_iteration = 2
    cfg.train_epoch_size = 8
    cfg.games_per_iteration = 1
    cfg.replay_buffer_size = 100
    cfg.num_iterations = 1
    cfg.arena_every = 1
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


def milestone_files(cfg):
    d = os.path.join(cfg.checkpoint_dir, "milestones")
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".pt"))


def test_acceptance_writes_immutable_milestone(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    examples = synthetic_examples(10, seed=9)

    monkeypatch.setattr(train, "play_game",
                        lambda net, cfg: [tuple(e) for e in examples])
    # score 1.0 (2 wins, 0 losses, 0 draws) -> accepted
    monkeypatch.setattr(train, "play_match",
                        lambda a, b, cfg, num_games, openings=None:
                        {"a": 2, "b": 0, "draws": 0})

    train.run(cfg, resume=False)

    pts = milestone_files(cfg)
    assert len(pts) == 1, f"expected exactly one milestone, got {pts}"
    pt = pts[0]
    assert pt.startswith("best-gen0001-iter0001-"), pt

    # sidecar JSON records arena provenance + architecture identity
    sidecar = os.path.join(cfg.checkpoint_dir, "milestones", pt[:-3] + ".json")
    assert os.path.exists(sidecar), "milestone sidecar missing"
    with open(sidecar) as f:
        meta = json.load(f)
    assert meta["generation"] == 1
    assert meta["iteration"] == 1
    assert meta["run_id"], "milestone must carry run_id"
    assert meta["architecture_id"], "milestone must record architecture identity"
    assert meta["arena"]["wins"] == 2
    assert meta["arena"]["draws"] == 0
    assert meta["arena"]["losses"] == 0
    assert meta["arena"]["score"] == 1.0

    # the milestone weights are exactly the accepted best's state dict
    milestone = torch.load(os.path.join(cfg.checkpoint_dir, "milestones", pt),
                           map_location="cpu", weights_only=False)
    best = torch.load(os.path.join(cfg.checkpoint_dir, "best.pt"),
                      map_location="cpu", weights_only=False)
    assert set(milestone.keys()) == set(best.keys())
    for k in best:
        assert torch.equal(milestone[k], best[k]), k


def test_rejection_writes_no_milestone(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=2)
    examples = synthetic_examples(10, seed=7)

    monkeypatch.setattr(train, "play_game",
                        lambda net, cfg: [tuple(e) for e in examples])
    # score 0.0 (2 losses) -> rejected at every arena gate
    monkeypatch.setattr(train, "play_match",
                        lambda a, b, cfg, num_games, openings=None:
                        {"a": 0, "b": 2, "draws": 0})

    train.run(cfg, resume=False)

    assert milestone_files(cfg) == [], \
        "rejected candidates must never produce milestones"


def test_each_promotion_is_a_distinct_milestone(tmp_path, monkeypatch):
    """Two accepted gates -> two immutable milestones (never overwritten)."""
    cfg = make_cfg(tmp_path, num_iterations=4, arena_every=2)
    examples = synthetic_examples(10, seed=11)

    monkeypatch.setattr(train, "play_game",
                        lambda net, cfg: [tuple(e) for e in examples])
    monkeypatch.setattr(train, "play_match",
                        lambda a, b, cfg, num_games, openings=None:
                        {"a": 2, "b": 0, "draws": 0})  # always accepted

    train.run(cfg, resume=False)

    pts = milestone_files(cfg)
    assert len(pts) == 2, f"expected 2 milestones (arenas at iters 2 and 4), got {pts}"
    gens = sorted(int(p.split("-gen")[1].split("-")[0]) for p in pts)
    assert gens == [1, 2], f"generations must increment 1,2; got {gens}"
