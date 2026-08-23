"""TDD tests for Phase 5: minimal observability (round seed + arena suite hash).

Written BEFORE the implementation.  Requirements:

  * per-iteration records carry the derived self-play round seed (native path)
  * per-arena-event records carry arena opening seed / pair count / suite hash
  * OLD metrics readers (the dashboard) tolerate the extra JSON fields
  * round seed and arena suite hash are stable / deterministic

Run:  .venv/bin/python -m pytest tests/test_observability.py -q
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from config import Config  # noqa: E402
import arena  # noqa: E402


def _cfg(tmp_path):
    cfg = Config()
    cfg.metrics_path = str(tmp_path / "training.jsonl")
    return cfg


def _read(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def test_round_seed_logged_and_stable(tmp_path):
    cfg = _cfg(tmp_path)
    train._log_metrics(cfg, run_id="r", iteration=10, generation=0,
                       policy_loss=1.0, value_loss=0.5, entropy=0.1,
                       optimizer_steps=7, replay_size=100, games=120,
                       round_seed=123456789)
    rec = _read(cfg.metrics_path)[0]
    assert rec["round_seed"] == 123456789
    assert isinstance(rec["round_seed"], int)


def test_arena_suite_fields_logged(tmp_path):
    cfg = _cfg(tmp_path)
    train._log_arena_event(cfg, run_id="r", iteration=20, generation=1,
                           wins=10, draws=10, losses=0, score=0.75,
                           accepted=True, opening_seed=424242,
                           opening_pairs=10, opening_suite_hash="abc123")
    rec = _read(cfg.metrics_path)[0]
    assert rec["opening_seed"] == 424242
    assert rec["opening_pairs"] == 10
    assert rec["opening_suite_hash"] == "abc123"


def test_old_reader_tolerates_extra_fields(tmp_path):
    cfg = _cfg(tmp_path)
    train._log_metrics(cfg, run_id="r", iteration=1, generation=0,
                       policy_loss=1.0, value_loss=0.5, entropy=0.1,
                       optimizer_steps=1, replay_size=10, games=12,
                       round_seed=999)
    train._log_arena_event(cfg, run_id="r", iteration=1, generation=0,
                           wins=1, draws=1, losses=0, score=0.5, accepted=False,
                           opening_seed=424242, opening_pairs=1,
                           opening_suite_hash="hash")
    recs = _read(cfg.metrics_path)

    # An "old" reader extracts only the legacy fields; the new fields are extra
    # keys it ignores.  Nothing may raise, and the legacy fields must persist.
    old_fields = {"run_id", "iteration", "generation", "policy_loss",
                  "value_loss", "entropy", "optimizer_steps", "replay_size",
                  "games"}
    arena_fields = {"event", "wins", "draws", "losses", "score", "accepted"}
    for r in recs:
        if r.get("event") == "arena":
            assert arena_fields <= set(r)
        else:
            assert old_fields <= set(r)


def test_arena_suite_hash_stable_and_seed_sensitive():
    a = arena.generate_arena_openings(10, 8, 424242)
    assert arena.arena_suite_hash(a) == arena.arena_suite_hash(a)
    b = arena.generate_arena_openings(10, 8, 424243)
    assert arena.arena_suite_hash(a) != arena.arena_suite_hash(b)
