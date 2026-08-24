"""Ticket A: permanent phase telemetry — swallow-guard, schema conformance,
determinism, diversity parity and resource grace.

Per docs/telemetry-design.md §6:
  1. swallow-guard: a forced `telemetry.emit` failure must NOT abort
     `train.run_native` (the loop completes, checkpoints land).
  2. schema conformance: every emitted line parses as JSON, carries
     `schema == "telemetry/v1"` and a valid `type`; required fields per §2.
  3. semantic neutrality: replay examples / checkpoints are byte-identical
     with telemetry on vs off.
  4. diversity parity: `telemetry.replay_diversity` matches
     `scripts/audit_replay.audit_replay` on a synthetic buffer.
  5. resource grace: with CUDA unavailable, `sample_resources()` returns GPU
     fields as None without raising; psutil fields stay populated.

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_telemetry.py -q
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import time

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import telemetry  # noqa: E402
from config import Config  # noqa: E402
from replay import ReplayBuffer  # noqa: E402
from native_selfplay import NativeSelfPlay  # noqa: E402
import audit_replay  # noqa: E402

PLANES, BOARD, POLICY_SIZE = 104, 8, 4672


# --------------------------------------------------------------------------- #
#  shared fixtures                                                             #
# --------------------------------------------------------------------------- #

def fake_inference(inputs, offsets, indices):
    """Deterministic hash-based logits + zero values (mirrors the native MCTS
    tests and tests/test_native_train_e2e.py)."""
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


def _cfg(tmp_path, **over):
    cfg = Config()
    cfg.checkpoint_dir = str(tmp_path)
    cfg.device = "cpu"
    cfg.amp = False
    cfg.seed = 42
    cfg.games_per_iteration = 1
    cfg.num_simulations = 4
    cfg.max_game_length = 8
    cfg.temperature_threshold = 5
    cfg.num_iterations = 2
    cfg.train_epoch_size = 16
    cfg.training_epochs = 1
    cfg.train_batch_size = 16
    cfg.replay_buffer_size = 200
    cfg.arena_every = 1
    cfg.arena_games = 2
    cfg.arena_accept_threshold = 0.55
    cfg.checkpoint_every_iterations = 1
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _stub_inference(c, model=None):
    """CPU stand-in for `native_selfplay.make_gpu_inference_fn`: the real
    native actor runs against the deterministic fake evaluator."""
    def _fn(inputs, offsets, indices):
        return fake_inference(inputs, offsets, indices)

    _fn.update_weights = lambda sd: None  # no-op weight publish
    _fn.runtime = None
    return _fn


def _run_sp(cfg, seed):
    sp = NativeSelfPlay(cfg, fake_inference, games=cfg.games_per_iteration,
                        seed=seed)
    return sp.run()


def _read_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# --------------------------------------------------------------------------- #
#  PhaseTimer / emit primitives                                                #
# --------------------------------------------------------------------------- #

def test_phase_timer_measures_duration():
    with telemetry.PhaseTimer("x") as t:
        time.sleep(0.01)
    assert t.name == "x"
    assert t.duration_s >= 0.01


def test_phase_timer_does_not_swallow_body_exceptions():
    with pytest.raises(ValueError):
        with telemetry.PhaseTimer("x"):
            raise ValueError("body")


def test_emit_writes_json_lines(tmp_path):
    cfg = Config()
    cfg.telemetry_path = str(tmp_path / "telemetry.jsonl")
    telemetry.emit(cfg, {"type": "phase", "phase": "smoke", "duration_s": 0.5})
    recs = _read_records(cfg.telemetry_path)
    assert len(recs) == 1
    assert recs[0]["schema"] == "telemetry/v1"
    assert recs[0]["type"] == "phase"
    assert recs[0]["phase"] == "smoke"
    assert isinstance(recs[0]["t"], float)


def test_emit_disabled_writes_nothing(tmp_path):
    cfg = Config()
    cfg.telemetry_enabled = False
    cfg.telemetry_path = str(tmp_path / "telemetry.jsonl")
    telemetry.emit(cfg, {"type": "phase"})
    assert not os.path.exists(cfg.telemetry_path)


def test_emit_never_raises_on_io_failure():
    cfg = Config()
    cfg.telemetry_path = "/nonexistent-dir-xyz/telemetry.jsonl"
    telemetry.emit(cfg, {"type": "phase"})  # must not raise


def test_emit_without_checkpoint_dir_writes_nothing(tmp_path, monkeypatch):
    # A degenerate cfg (no checkpoint_dir, no telemetry_path) must not write
    # into the process CWD.
    class BareCfg:
        pass
    cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    telemetry.emit(BareCfg(), {"type": "phase"})
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------------- #
#  1. swallow-guard: a forced emit failure must never abort run_native         #
# --------------------------------------------------------------------------- #

def test_emit_failure_does_not_abort_run_native(tmp_path, monkeypatch):
    import train

    def boom(cfg, record):
        raise RuntimeError("telemetry emit failed")

    monkeypatch.setattr(telemetry, "emit", boom)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(train.native_selfplay, "make_gpu_inference_fn",
                        _stub_inference)
    # Force the arena gate + checkpoint snapshot so every emit path is hit.
    monkeypatch.setattr(train, "play_match",
                        lambda a, b, cfg_, num_games, openings=None: {"a": 1, "b": 1, "draws": 0})

    train.run_native(cfg, resume=False)  # must complete without raising

    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt"))
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "best.pt"))
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "checkpoint_meta.json"))


def test_arena_emit_failure_preserves_return_contract(tmp_path, monkeypatch):
    import arena

    class _FakeNet:
        def eval(self):
            return self

    monkeypatch.setattr(telemetry, "emit",
                        lambda cfg_, rec: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(arena, "_play_arena_game", lambda *a, **k: 0.0)
    cfg = Config()
    cfg.arena_games = 2
    cfg.max_game_length = 20
    result = arena.play_match(_FakeNet(), _FakeNet(), cfg, num_games=2)
    assert set(result) == {"a", "b", "draws"}
    assert result == {"a": 0, "b": 0, "draws": 2}


# --------------------------------------------------------------------------- #
#  2. schema conformance: every emitted line is a valid telemetry record       #
# --------------------------------------------------------------------------- #

_PHASE_FIELDS = {
    "selfplay": {"games", "examples", "inference_calls", "batch_min",
                 "batch_mean", "batch_p50", "batch_p90", "batch_max",
                 "simulations", "sims_per_s", "games_per_hour", "round_seed"},
    "gather_apply_advance": {"gather_calls", "apply_calls", "advance_calls",
                             "gather_s", "apply_s", "advance_s",
                             "inference_calls", "simulations",
                             "batch_min", "batch_mean", "batch_p50",
                             "batch_p90", "batch_max"},
    "training": {"steps", "batches", "train_batch_size", "policy_loss",
                 "value_loss", "entropy", "optimizer_steps"},
    "arena": {"arena_games", "arena_sims", "wins", "draws", "losses",
              "score", "accepted", "opening_seed", "opening_pairs",
              "opening_suite_hash"},
    # ShardedSelfPlay emits its own round record in addition to the trainer's
    # `selfplay` one: it is the driver that knows the shard split and the GPU
    # occupancy, and it is also used standalone by the benchmarks.
    "selfplay_sharded": {"shards", "shard_games", "games", "examples",
                         "gather_s", "apply_s", "advance_s", "infer_s",
                         "gpu_busy_s", "gpu_busy_fraction", "inference_calls",
                         "gather_calls", "simulations", "leaves_per_game",
                         "max_batch"},
    "checkpoint": {"snapshot", "bytes", "reason"},
}

_RESOURCE_FIELDS = {"cpu_percent", "cpu_count", "ram_used_mb", "ram_total_mb",
                    "ram_percent", "swap_used_mb", "swap_total_mb",
                    "gpu_util_percent", "vram_used_mb", "vram_total_mb",
                    "torch_alloc_mb", "torch_reserved_mb"}


def test_run_native_emits_schema_conformant_records(tmp_path, monkeypatch):
    import train

    cfg = _cfg(tmp_path)
    monkeypatch.setattr(train.native_selfplay, "make_gpu_inference_fn",
                        _stub_inference)
    monkeypatch.setattr(train, "play_match",
                        lambda a, b, cfg_, num_games, openings=None: {"a": 0, "b": 2, "draws": 0})

    train.run_native(cfg, resume=False)

    path = os.path.join(cfg.checkpoint_dir, "telemetry.jsonl")
    assert os.path.exists(path), "telemetry.jsonl must be written"
    recs = _read_records(path)
    assert recs, "no telemetry records emitted"

    types = set()
    phases = set()
    run_ids = set()
    for r in recs:
        assert r["schema"] == "telemetry/v1"
        assert r["type"] in {"phase", "resource", "diversity"}
        assert isinstance(r["t"], float)
        types.add(r["type"])
        if r["type"] == "phase":
            assert r["phase"] in _PHASE_FIELDS, r
            phases.add(r["phase"])
            assert "duration_s" in r and isinstance(r["duration_s"], float)
            for k in _PHASE_FIELDS[r["phase"]]:
                assert k in r, f"phase={r['phase']} missing {k}"
            if r["run_id"] is not None:
                run_ids.add(r["run_id"])
        elif r["type"] == "resource":
            for k in _RESOURCE_FIELDS:
                assert k in r, f"resource missing {k}"
        elif r["type"] == "diversity":
            assert r["source"] in {"replay_buffer", "selfplay_round"}
            assert "replay_size" in r
            assert "unique_trajectory_hashes" in r
            assert "most_repeated_trajectory_count" in r
            assert "trajectory_hashes" in r
            assert len(r["trajectory_hashes"]) <= 32
            if r["source"] == "replay_buffer":
                for k in ("unique_packed_states", "unique_exact_examples",
                          "unique_state_fraction", "unique_example_fraction"):
                    assert k in r

    # every phase type fires across the run (selfplay/gather/training/arena/
    # checkpoint with arena_every=1 and checkpoint_every_iterations=1)
    assert {"selfplay", "gather_apply_advance", "training", "arena",
            "checkpoint"} <= phases, phases
    assert types == {"phase", "resource", "diversity"}, types

    # records carry the run's actual run_id where they know it
    meta = json.load(open(os.path.join(cfg.checkpoint_dir,
                                       "checkpoint_meta.json")))
    assert meta["run_id"] in run_ids, run_ids


# --------------------------------------------------------------------------- #
#  3. semantic neutrality: byte-identical output with telemetry on vs off      #
# --------------------------------------------------------------------------- #

def test_selfplay_deterministic_telemetry_on_vs_off(tmp_path):
    import tempfile

    on_dir = tempfile.mkdtemp(dir=str(tmp_path))
    off_dir = tempfile.mkdtemp(dir=str(tmp_path))

    cfg_on = _cfg(on_dir)
    cfg_on.telemetry_path = os.path.join(on_dir, "telemetry.jsonl")
    ex_on = _run_sp(cfg_on, seed=7)

    cfg_off = _cfg(off_dir)
    cfg_off.telemetry_enabled = False
    cfg_off.telemetry_path = os.path.join(off_dir, "telemetry.jsonl")
    ex_off = _run_sp(cfg_off, seed=7)

    assert len(ex_on) == len(ex_off)
    for (sa, pa, za), (sb, pb, zb) in zip(ex_on, ex_off):
        assert np.array_equal(sa, sb), "state differs with telemetry on"
        assert np.array_equal(pa, pb), "policy differs with telemetry on"
        assert za == zb, "z differs with telemetry on"
    assert os.path.exists(cfg_on.telemetry_path)
    assert not os.path.exists(cfg_off.telemetry_path)


def test_run_native_checkpoints_identical_telemetry_on_vs_off(tmp_path, monkeypatch):
    import torch
    import train

    def run_once(subdir, telemetry_enabled):
        cfg = _cfg(tmp_path / subdir)
        cfg.num_iterations = 2
        cfg.telemetry_enabled = telemetry_enabled
        monkeypatch.setattr(train.native_selfplay, "make_gpu_inference_fn",
                            _stub_inference)
        train.run_native(cfg, resume=False)
        return torch.load(os.path.join(cfg.checkpoint_dir, "latest.pt"),
                          map_location="cpu", weights_only=False)

    p_on = run_once("on", True)
    p_off = run_once("off", False)

    assert p_on["iteration"] == p_off["iteration"] == 2
    assert p_on["generation"] == p_off["generation"]
    assert p_on["optimizer_steps"] == p_off["optimizer_steps"]
    for key in ("candidate", "best"):
        for wk in p_on[key]:
            assert torch.equal(p_on[key][wk], p_off[key][wk]), \
                f"{key}.{wk} differs with telemetry on"
    for arr in ("positions", "legal_idx", "probs", "offsets", "z",
                "state_extra_idx", "state_extra_values", "state_extra_offsets"):
        assert np.array_equal(p_on["replay"][arr], p_off["replay"][arr]), arr


# --------------------------------------------------------------------------- #
#  4. diversity parity: replay_diversity == audit_replay on a synthetic buffer #
# --------------------------------------------------------------------------- #

def _state(game_start, tag=0):
    state = np.zeros((PLANES, BOARD, BOARD), dtype=np.float32)
    r, c = (tag // BOARD) % BOARD, tag % BOARD
    state[0, r, c] = 1.0
    if not game_start:
        state[12, 0, 0] = 1.0       # history step 1 non-zero -> not a start
    state[96].fill(1.0)             # side-to-move plane
    return state


def _example(game_start, tag):
    pi = np.zeros(POLICY_SIZE, dtype=np.float32)
    pi[int(tag) % POLICY_SIZE] = 1.0
    z = float((tag % 3) - 1)
    return (_state(game_start, tag), pi, z)


def _make_game(tag, plies=3):
    return [_example(True, tag)] + \
           [_example(False, tag + j + 1) for j in range(plies - 1)]


def test_replay_diversity_matches_audit_replay(tmp_path):
    # one 2-game block repeated 3x + one unique game -> 7 games, 3 trajectories
    block = _make_game(100) + _make_game(200)
    examples = list(block) * 3 + _make_game(300)

    buf = ReplayBuffer(capacity=len(examples) * 4, policy_size=POLICY_SIZE,
                       num_input_planes=PLANES, board_size=BOARD)
    for e in examples:
        buf.add(*e)

    d = telemetry.replay_diversity(buf)

    # checkpoint round-trip for the reference audit (same state_dict bytes)
    payload = {
        "schema_version": 2,
        "checkpoint_format": "schema-v3",
        "architecture_id": "v2-6x128",
        "run_id": "run-test",
        "iteration": 1,
        "generation": 0,
        "policy_size": POLICY_SIZE,
        "num_input_planes": PLANES,
        "board_size": BOARD,
        "config": {"games_per_iteration": 2},
        "replay": buf.state_dict(),
        "best": {},
    }
    path = str(tmp_path / "latest.pt")
    torch_save = __import__("torch").save
    torch_save(payload, path)
    r = audit_replay.audit_replay(path)

    assert d["replay_size"] == r["replay_example_count"] == len(examples)
    assert d["unique_packed_states"] == r["unique_packed_states"]
    assert d["unique_exact_examples"] == r["unique_exact_examples"]
    assert d["unique_state_fraction"] == pytest.approx(r["unique_state_fraction"])
    assert d["unique_example_fraction"] == pytest.approx(r["unique_example_fraction"])
    assert d["unique_trajectory_hashes"] == r["unique_full_game_trajectory_hashes"]
    assert d["most_repeated_trajectory_count"] == r["most_repeated_trajectory_count"]
    # 3 distinct trajectories, game A repeated 3x
    assert d["unique_trajectory_hashes"] == 3
    assert d["most_repeated_trajectory_count"] == 3
    assert len(d["trajectory_hashes"]) == 3
    assert all(len(h) == 32 for h in d["trajectory_hashes"])


def test_replay_diversity_empty_buffer():
    buf = ReplayBuffer(10, POLICY_SIZE, PLANES, BOARD)
    d = telemetry.replay_diversity(buf)
    assert d["replay_size"] == 0
    assert d["unique_packed_states"] == 0
    assert d["unique_exact_examples"] == 0
    assert d["unique_trajectory_hashes"] == 0
    assert d["most_repeated_trajectory_count"] == 0
    assert d["trajectory_hashes"] == []


class _BrokenBuffer:
    def state_dict(self):
        raise RuntimeError("boom")


def test_replay_diversity_never_raises():
    d = telemetry.replay_diversity(_BrokenBuffer())
    assert d["replay_size"] == 0


# --------------------------------------------------------------------------- #
#  5. resource grace                                                           #
# --------------------------------------------------------------------------- #

def test_sample_resources_grace_without_cuda():
    # test command runs with CUDA_VISIBLE_DEVICES="" -> GPU fields must be None
    r = telemetry.sample_resources()
    assert r["gpu_util_percent"] is None
    assert r["vram_used_mb"] is None
    assert r["vram_total_mb"] is None
    assert r["torch_alloc_mb"] is None
    assert r["torch_reserved_mb"] is None
    # psutil fields are populated
    assert isinstance(r["cpu_percent"], float)
    assert isinstance(r["cpu_count"], int) and r["cpu_count"] > 0
    assert isinstance(r["ram_used_mb"], float) and r["ram_used_mb"] > 0
    assert isinstance(r["ram_total_mb"], float) and r["ram_total_mb"] > 0
    assert isinstance(r["ram_percent"], float)


def test_sample_resources_never_raises_on_psutil_failure(monkeypatch):
    import psutil

    def boom(*a, **k):
        raise RuntimeError("psutil broken")

    monkeypatch.setattr(psutil, "cpu_percent", boom)
    monkeypatch.setattr(psutil, "virtual_memory", boom)
    monkeypatch.setattr(psutil, "swap_memory", boom)
    r = telemetry.sample_resources()  # must not raise
    assert r["cpu_percent"] is None
    assert r["ram_used_mb"] is None
    assert r["ram_total_mb"] is None
    assert r["swap_used_mb"] is None


# --------------------------------------------------------------------------- #
#  supporting contracts                                                        #
# --------------------------------------------------------------------------- #

def test_game_trajectory_hash_stable_and_content_sensitive():
    zero = np.zeros((PLANES, BOARD, BOARD), dtype=np.float32)
    ex = [(zero, np.zeros(POLICY_SIZE, dtype=np.float32), 1.0),
          (zero, np.zeros(POLICY_SIZE, dtype=np.float32), -1.0)]
    assert telemetry.game_trajectory_hash(ex) == telemetry.game_trajectory_hash(ex)
    assert len(telemetry.game_trajectory_hash(ex)) == 32  # 16-byte BLAKE2 hex
    ex2 = [(np.ones((PLANES, BOARD, BOARD), dtype=np.float32),
            np.zeros(POLICY_SIZE, dtype=np.float32), 1.0),
           (zero, np.zeros(POLICY_SIZE, dtype=np.float32), -1.0)]
    assert telemetry.game_trajectory_hash(ex2) != telemetry.game_trajectory_hash(ex)
    # ordered (state, pi, z): reordering the examples changes the digest
    ex3 = list(reversed(ex))
    assert telemetry.game_trajectory_hash(ex3) != telemetry.game_trajectory_hash(ex)


def test_gpu_runtime_stats_contract():
    from gpu_runtime import InferenceRuntime

    rt = InferenceRuntime.__new__(InferenceRuntime)
    rt.call_count = 4
    rt.batch_b = collections.deque([8, 8, 16, 32], maxlen=10_000)
    rt.total_forward_s = 2.5
    s = rt.stats()
    assert s["calls"] == 4
    assert s["batch_min"] == 8
    assert s["batch_mean"] == pytest.approx(16.0)
    assert s["batch_p50"] == pytest.approx(12.0)
    assert s["batch_p90"] == pytest.approx(27.2)
    assert s["batch_max"] == 32
    assert s["total_forward_s"] == 2.5

    rt2 = InferenceRuntime.__new__(InferenceRuntime)
    rt2.call_count = 0
    rt2.batch_b = collections.deque(maxlen=10_000)
    rt2.total_forward_s = 0.0
    s2 = rt2.stats()
    assert s2["calls"] == 0
    assert s2["batch_min"] is None
    assert s2["batch_max"] is None


class _FakeRuntime:
    """Mimics InferenceRuntime.stats() for the merge test (CPU-safe)."""

    def __init__(self):
        self.call_count = 5
        self.batch_b = collections.deque([4, 8, 12], maxlen=10_000)
        self.total_forward_s = 0.25

    def stats(self):
        b = list(self.batch_b)
        arr = np.asarray(b, dtype=np.float64)
        return {"calls": self.call_count,
                "batch_min": float(arr.min()),
                "batch_mean": float(arr.mean()),
                "batch_p50": float(np.percentile(arr, 50)),
                "batch_p90": float(np.percentile(arr, 90)),
                "batch_max": float(arr.max()),
                "total_forward_s": self.total_forward_s}


def test_run_native_merges_gpu_runtime_stats(tmp_path, monkeypatch):
    import train

    def _stub_with_runtime(c, model=None):
        def _fn(inputs, offsets, indices):
            return fake_inference(inputs, offsets, indices)

        _fn.update_weights = lambda sd: None
        _fn.runtime = _FakeRuntime()
        return _fn

    cfg = _cfg(tmp_path)
    monkeypatch.setattr(train.native_selfplay, "make_gpu_inference_fn",
                        _stub_with_runtime)
    monkeypatch.setattr(train, "play_match",
                        lambda a, b, cfg_, num_games, openings=None: {"a": 0, "b": 2, "draws": 0})

    train.run_native(cfg, resume=False)

    recs = _read_records(os.path.join(cfg.checkpoint_dir, "telemetry.jsonl"))
    resources = [r for r in recs if r["type"] == "resource"]
    assert resources, "no resource records"
    # the canonical GPU-side counters are merged into the resource record
    assert resources[0]["gpu_calls"] == 5
    assert resources[0]["gpu_batch_max"] == 12
    assert resources[0]["gpu_batch_mean"] == pytest.approx(8.0)
    assert resources[0]["gpu_total_forward_s"] == 0.25
    # and into the selfplay record
    selfplays = [r for r in recs
                 if r["type"] == "phase" and r["phase"] == "selfplay"]
    assert selfplays and selfplays[0]["gpu_calls"] == 5


def test_native_selfplay_exposes_stats_and_emits_records(tmp_path):
    cfg = _cfg(tmp_path)
    sp = NativeSelfPlay(cfg, fake_inference, games=2,
                        weight_version=0, generation=3, seed=123)
    ex = sp.run()
    assert len(ex) > 0
    assert sp.gather_calls >= 1
    assert sp.apply_calls >= 1
    assert sp.advance_calls >= sp.apply_calls
    assert sp.inference_calls == sp.apply_calls
    assert sp.simulations > 0
    assert sp.round_duration_s >= 0.0
    assert len(sp.trajectory_hashes) == 2
    assert sp.batch_stats["batch_min"] is not None

    recs = _read_records(os.path.join(cfg.checkpoint_dir, "telemetry.jsonl"))
    assert recs
    ga = [r for r in recs if r.get("phase") == "gather_apply_advance"]
    assert ga, "gather_apply_advance record missing"
    assert ga[0]["gather_calls"] == sp.gather_calls
    assert ga[0]["apply_calls"] == sp.apply_calls
    assert ga[0]["advance_calls"] == sp.advance_calls
    div = [r for r in recs if r.get("source") == "selfplay_round"]
    assert div, "selfplay_round diversity record missing"
    assert len(div[0]["trajectory_hashes"]) == 2
    assert div[0]["generation"] == 3


def test_arena_play_match_emits_arena_phase(tmp_path, monkeypatch):
    import arena

    class _FakeNet:
        def eval(self):
            return self

    cfg = Config()
    cfg.telemetry_path = str(tmp_path / "telemetry.jsonl")
    cfg.arena_games = 2
    cfg.arena_simulations = 2
    cfg.max_game_length = 20
    cfg.arena_seed = 424242
    cfg.arena_opening_plies = 4

    calls = []

    def fake_game(mcts_white, mcts_black, cfg_, num_sims, opening_moves):
        calls.append(tuple(opening_moves))
        return 0.0

    monkeypatch.setattr(arena, "_play_arena_game", fake_game)
    result = arena.play_match(_FakeNet(), _FakeNet(), cfg, num_games=2)
    assert result == {"a": 0, "b": 0, "draws": 2}
    assert len(calls) == 2

    recs = _read_records(cfg.telemetry_path)
    arena_recs = [r for r in recs if r.get("phase") == "arena"]
    assert arena_recs, "arena phase record missing"
    assert arena_recs[0]["arena_games"] == 2
    assert arena_recs[0]["arena_sims"] == 2
    assert "duration_s" in arena_recs[0]
