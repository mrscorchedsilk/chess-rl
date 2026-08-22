"""Sprint A/C vertical slice: train.py v2 training-loop reliability and learning semantics.

Strict TDD (RED on the v1 train.py, GREEN after the v2 rewrite).  All tests are
CPU-only: self-play and the arena are replaced with deterministic fakes, the
network is tiny, and no real training run is ever executed.

Run:  .venv/bin/python -m pytest tests/test_train_loop_v2.py -q

Slices proven here:
  * exact checkpoint round-trip (schema v2 payload, candidate/best/optimizer/
    iteration/RNGs/compressed replay/config snapshot/run_id/generation/
    optimizer_steps)
  * checkpoint_meta.json consistency with latest.pt
  * resume continues at iteration+1 and preserves run_id / generation /
    optimizer_steps / replay
  * replay restoration after resume
  * rejection reverts candidate to best and resets the optimizer
  * acceptance installs candidate as best and increments generation
  * optimizer_steps equals the ACTUAL number of optimizer.step() calls
  * arena metrics are event-only; iteration records carry run_id/generation/
    policy/value/entropy/optimizer_steps/replay_size separately
  * standard arena score (wins + 0.5*draws)/games, draws score half
  * finally-save on injected interruption (reason=interrupt) and normal
    completion (reason=final)
  * legacy v1 checkpoints raise a clear incompatibility error and are never
    auto-loaded
  * self-play uses the accepted best as teacher, never the candidate
  * parallel result envelopes (kind=game/error) are unpacked with
    timeout + liveness checks (no unbounded Queue.get hang)
"""
import json
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


class _Interrupt(Exception):
    """Simulated graceful interruption injected between iterations."""


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #

def make_cfg(tmp_path, **overrides):
    """Tiny CPU-only config; every run lands in tmp_path."""
    cfg = Config()
    cfg.num_res_blocks = 1
    cfg.num_filters = 4
    cfg.num_simulations = 1
    cfg.batch_size = 4
    cfg.train_batch_size = 4
    cfg.training_epochs = 2
    cfg.epochs_per_iteration = 2
    cfg.train_epoch_size = 8          # bounded shuffled-epoch sample
    cfg.games_per_iteration = 1
    cfg.replay_buffer_size = 100
    cfg.num_iterations = 10
    cfg.arena_every = 2
    cfg.arena_games = 2
    cfg.arena_accept_threshold = 0.55
    cfg.arena_simulations = 1
    cfg.max_game_length = 8
    cfg.checkpoint_interval_minutes = 1_000_000   # never trigger time-based path
    cfg.checkpoint_every_iterations = 1           # versioned snapshot each iteration
    cfg.device = "cpu"
    cfg.seed = 42
    cfg.checkpoint_dir = str(tmp_path / "ckpts")
    cfg.metrics_path = str(tmp_path / "training.jsonl")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def synthetic_examples(n, seed=0, policy_size=4672, planes=104, board=8):
    """(state, pi, z) triples with realistic v2 shapes; pi is sparse."""
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


def make_fake_selfplay(monkeypatch, examples_factory):
    """Replace train.play_game; records (clones of) the net used as teacher."""
    seen = {"teachers": []}

    def fake_play_game(net, cfg):
        seen["teachers"].append(
            {k: v.detach().clone() for k, v in net.state_dict().items()}
        )
        return examples_factory()

    monkeypatch.setattr(train, "play_game", fake_play_game)
    return seen


def make_fake_arena(monkeypatch, result):
    monkeypatch.setattr(train, "play_match",
                        lambda net_a, net_b, cfg, num_games: dict(result))


def state_dicts_equal(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def load_latest(cfg):
    return torch.load(os.path.join(cfg.checkpoint_dir, "latest.pt"),
                      map_location="cpu", weights_only=False)


def load_meta(cfg):
    with open(os.path.join(cfg.checkpoint_dir, "checkpoint_meta.json")) as f:
        return json.load(f)


def read_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def snapshots(cfg):
    return sorted(f for f in os.listdir(cfg.checkpoint_dir)
                  if f.startswith("ckpt-iter"))


def make_replay(cfg):
    return ReplayBuffer(cfg.replay_buffer_size, cfg.policy_size,
                        cfg.num_input_planes, cfg.board_size)


# --------------------------------------------------------------------------- #
#  1. exact checkpoint round-trip + metadata consistency                       #
# --------------------------------------------------------------------------- #

def test_checkpoint_roundtrip_and_meta_consistency(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=2, arena_every=10_000)
    examples = synthetic_examples(10, seed=3)
    make_fake_selfplay(monkeypatch, lambda: [tuple(e) for e in examples])

    def interrupt(it):
        if it >= 2:
            raise _Interrupt()

    with pytest.raises(_Interrupt):
        train.run(cfg, resume=False, on_iteration=interrupt)

    latest = os.path.join(cfg.checkpoint_dir, "latest.pt")
    meta_path = os.path.join(cfg.checkpoint_dir, "checkpoint_meta.json")
    assert os.path.exists(latest), "latest.pt missing after interrupted run"
    assert os.path.exists(meta_path), "checkpoint_meta.json missing"

    payload = load_latest(cfg)
    # -- required schema keys --
    assert payload["schema_version"] == train.CHECKPOINT_SCHEMA_VERSION == 2
    for key in ("candidate", "best", "optimizer", "iteration", "torch_rng",
                "cuda_rng", "random_rng", "numpy_rng", "replay", "config",
                "run_id", "generation", "optimizer_steps"):
        assert key in payload, f"missing checkpoint key: {key}"
    assert payload["iteration"] == 2
    assert payload["generation"] == 0

    # -- candidate/best/optimizer load cleanly (exact round-trip) --
    net = ChessNet(cfg).to("cpu")
    net.load_state_dict(payload["candidate"])
    best = ChessNet(cfg).to("cpu")
    best.load_state_dict(payload["best"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    opt.load_state_dict(payload["optimizer"])
    assert len(opt.state) > 0, "optimizer state must round-trip (steps were taken)"

    # -- RNGs round-trip --
    assert isinstance(payload["torch_rng"], torch.Tensor)
    torch.set_rng_state(payload["torch_rng"])
    assert torch.equal(torch.get_rng_state(), payload["torch_rng"])
    assert isinstance(payload["random_rng"], tuple)
    assert isinstance(payload["numpy_rng"], tuple)

    # -- compressed replay restores exactly, in insertion order --
    buf = make_replay(cfg)
    buf.load_state_dict(payload["replay"])
    assert len(buf) == 20, f"2 iterations x 10 examples, got {len(buf)}"
    s0, p0, z0 = buf.sample_indices(np.array([0]))
    e0 = examples[0]
    assert np.array_equal(s0[0].numpy(), e0[0]), "state round-trip lost bits"
    assert np.array_equal(p0[0].numpy(), e0[1]), "policy round-trip lost mass"
    assert float(z0[0, 0]) == e0[2]
    s10, _, _ = buf.sample_indices(np.array([10]))
    assert np.array_equal(s10[0].numpy(), e0[0]), "second game's first example"

    # -- config snapshot --
    for k in ("policy_size", "num_input_planes", "board_size", "checkpoint_dir",
              "train_batch_size", "training_epochs", "num_iterations"):
        assert payload["config"].get(k) == getattr(cfg, k), k

    # -- metadata consistency with the payload --
    meta = load_meta(cfg)
    assert meta["schema_version"] == 2
    assert meta["run_id"] == payload["run_id"]
    assert meta["iteration"] == payload["iteration"] == 2
    assert meta["generation"] == payload["generation"]
    assert meta["optimizer_steps"] == payload["optimizer_steps"]
    assert meta["replay_size"] == len(buf) == 20
    assert meta["reason"] == "interrupt", "interrupted run must say so in meta"
    assert meta["config"] == payload["config"]

    # -- per-iteration versioned snapshots exist --
    assert len(snapshots(cfg)) == 2, "a versioned snapshot per completed iteration"


# --------------------------------------------------------------------------- #
#  2. resume continues at next iteration, preserving run_id/generation/steps   #
# --------------------------------------------------------------------------- #

def test_resume_continues_next_iteration_and_keeps_state(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=4, arena_every=10_000)
    examples = synthetic_examples(10, seed=5)
    make_fake_selfplay(monkeypatch, lambda: [tuple(e) for e in examples])

    def interrupt1(it):
        if it >= 2:
            raise _Interrupt()

    with pytest.raises(_Interrupt):
        train.run(cfg, resume=False, on_iteration=interrupt1)

    p1 = load_latest(cfg)
    run_id1 = p1["run_id"]
    assert p1["iteration"] == 2
    assert p1["optimizer_steps"] == 8  # 2 iters x (2 epochs x 2 batches)

    seen2 = []

    def interrupt2(it):
        seen2.append(it)
        if it >= 3:
            raise _Interrupt()

    with pytest.raises(_Interrupt):
        train.run(cfg, resume=True, on_iteration=interrupt2)

    # -- resumes at iteration+1, not restarting at 1 --
    assert seen2[0] == 3, f"resume must continue at next iteration, saw {seen2}"
    p2 = load_latest(cfg)
    assert p2["iteration"] == 3
    assert p2["run_id"] == run_id1, "run_id must survive resume"
    assert p2["generation"] == p1["generation"] == 0

    # -- optimizer stepped on top of the restored optimizer, not a fresh one --
    assert p2["optimizer_steps"] == 12, \
        "resumed optimizer must continue counting (8 restored + 4 new)"

    # -- best survived byte-for-byte; candidate evolved --
    assert state_dicts_equal(p2["best"], p1["best"]), \
        "accepted best must round-trip exactly across resume"
    assert not state_dicts_equal(p2["candidate"], p1["candidate"]), \
        "resumed candidate must keep training (weights must change)"

    # -- replay restored then extended --
    buf = make_replay(cfg)
    buf.load_state_dict(p2["replay"])
    assert len(buf) == 30, f"3 iterations x 10 examples, got {len(buf)}"
    s20, _, _ = buf.sample_indices(np.array([20]))
    assert np.array_equal(s20[0].numpy(), examples[0][0]), \
        "resumed buffer must contain the pre-resume examples first"

    # -- metrics share the run_id across both halves of the run --
    recs = read_jsonl(cfg.metrics_path)
    assert len(recs) == 3, f"iterations 1..3, got {len(recs)}"
    assert all(r["run_id"] == run_id1 for r in recs), \
        "all metric records must carry the same run_id"


# --------------------------------------------------------------------------- #
#  3. rejection reverts candidate + resets optimizer                            #
# --------------------------------------------------------------------------- #

def test_rejected_candidate_reverts_to_best_and_resets_optimizer(
        tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=2, arena_every=1)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=7)])
    make_fake_arena(monkeypatch, {"a": 0, "b": 2, "draws": 0})  # score 0.0 -> reject

    train.run(cfg, resume=False)

    payload = load_latest(cfg)
    assert payload["iteration"] == 2
    assert payload["generation"] == 0, "rejection must not increment generation"

    net = ChessNet(cfg).to("cpu")
    net.load_state_dict(payload["candidate"])
    best = ChessNet(cfg).to("cpu")
    best.load_state_dict(payload["best"])
    assert state_dicts_equal(net.state_dict(), best.state_dict()), \
        "rejected candidate must revert to the accepted best"

    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    opt.load_state_dict(payload["optimizer"])
    assert len(opt.state) == 0, "optimizer must be freshly reset after rejection"
    assert payload["optimizer_steps"] == 0, \
        "fresh optimizer has taken zero steps"

    events = [r for r in read_jsonl(cfg.metrics_path) if r.get("event") == "arena"]
    assert len(events) == 2
    assert all(e["accepted"] is False for e in events)
    assert all(e["score"] == pytest.approx(0.0) for e in events)


# --------------------------------------------------------------------------- #
#  4. acceptance installs candidate as best + generation increments            #
# --------------------------------------------------------------------------- #

def test_accepted_candidate_becomes_new_best_and_generation_increments(
        tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=1, arena_every=1)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=9)])
    make_fake_arena(monkeypatch, {"a": 2, "b": 0, "draws": 0})  # score 1.0 -> accept

    train.run(cfg, resume=False)

    payload = load_latest(cfg)
    assert payload["generation"] == 1, "acceptance must increment generation"

    net = ChessNet(cfg).to("cpu")
    net.load_state_dict(payload["candidate"])
    best = ChessNet(cfg).to("cpu")
    best.load_state_dict(payload["best"])
    assert state_dicts_equal(net.state_dict(), best.state_dict()), \
        "accepted candidate must be installed as the new best"

    on_disk = torch.load(os.path.join(cfg.checkpoint_dir, "best.pt"),
                         map_location="cpu", weights_only=False)
    assert state_dicts_equal(on_disk, net.state_dict()), \
        "best.pt must hold the accepted candidate weights"

    events = [r for r in read_jsonl(cfg.metrics_path) if r.get("event") == "arena"]
    assert len(events) == 1
    assert events[0]["accepted"] is True
    assert events[0]["score"] == pytest.approx(1.0)
    assert events[0]["wins"] == 2 and events[0]["losses"] == 0
    assert events[0]["draws"] == 0


# --------------------------------------------------------------------------- #
#  5. optimizer_steps counts ACTUAL gradient steps                             #
# --------------------------------------------------------------------------- #

def test_optimizer_steps_counts_real_gradient_steps(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=2, arena_every=10_000,
                   train_epoch_size=8, train_batch_size=4, training_epochs=2)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=11)])

    real_adam = torch.optim.Adam
    calls = {"n": 0}

    class CountingAdam(real_adam):
        def step(self, *a, **k):
            calls["n"] += 1
            return super().step(*a, **k)

    monkeypatch.setattr(train, "Adam", CountingAdam)

    train.run(cfg, resume=False)

    # bounded sample = min(10, 8) = 8 -> 2 batches x 2 epochs = 4 steps/iteration
    expected = 2 * (8 // 4) * 2  # 2 iterations x 2 batches x 2 epochs = 8
    assert calls["n"] == expected, \
        f"recorded {expected} steps must equal real step() calls: {calls['n']}"
    payload = load_latest(cfg)
    assert payload["optimizer_steps"] == expected
    assert load_meta(cfg)["optimizer_steps"] == expected

    recs = read_jsonl(cfg.metrics_path)
    assert [r["optimizer_steps"] for r in recs] == [4, 8], \
        "per-iteration metric must reflect the cumulative step count"


# --------------------------------------------------------------------------- #
#  6. real bounded shuffled epochs: separate losses + entropy, event-only arena#
# --------------------------------------------------------------------------- #

def test_iteration_metrics_event_only_arena_and_separate_losses(
        tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=3, arena_every=2)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=13)])
    make_fake_arena(monkeypatch, {"a": 1, "b": 1, "draws": 0})  # score 0.5 -> reject

    train.run(cfg, resume=False)

    recs = read_jsonl(cfg.metrics_path)
    iters = [r for r in recs if "event" not in r]
    events = [r for r in recs if r.get("event") == "arena"]

    assert len(iters) == 3, "one iteration record per iteration"
    assert len(events) == 1, "arena metrics are event-only: 1 gate -> 1 event"

    for r in iters:
        assert r["run_id"]
        for key in ("policy_loss", "value_loss", "entropy", "optimizer_steps",
                    "replay_size", "generation", "iteration"):
            assert key in r, f"iteration record missing {key}"
        for banned in ("wins", "draws", "score", "accepted"):
            assert banned not in r, \
                f"iteration record must not carry arena fields: {banned}"
        assert np.isfinite(r["policy_loss"]) and np.isfinite(r["value_loss"])
        assert np.isfinite(r["entropy"]) and r["entropy"] > 0.0

    ev = events[0]
    assert ev["iteration"] == 2
    assert ev["wins"] == 1 and ev["losses"] == 1 and ev["draws"] == 0
    assert ev["score"] == pytest.approx(0.5)
    assert ev["accepted"] is False


# --------------------------------------------------------------------------- #
#  7. standard arena score: draws count as half                               #
# --------------------------------------------------------------------------- #

def test_arena_standard_score_counts_draws_as_half(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=1, arena_every=1, arena_games=4)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=27)])

    # 1 win + 1 draw + 2 losses -> score 0.375 < 0.55 -> reject
    make_fake_arena(monkeypatch, {"a": 1, "b": 2, "draws": 1})
    train.run(cfg, resume=False)
    ev = [r for r in read_jsonl(cfg.metrics_path) if r.get("event") == "arena"][0]
    assert ev["score"] == pytest.approx(0.375)
    assert ev["accepted"] is False

    # 2 wins + 1 draw + 1 loss -> score 0.625 >= 0.55 -> accept (v1 wins/games
    # would have said 0.5 and rejected — this proves the standard score)
    cfg2 = make_cfg(tmp_path / "run2", num_iterations=1, arena_every=1,
                    arena_games=4)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=29)])
    make_fake_arena(monkeypatch, {"a": 2, "b": 1, "draws": 1})
    train.run(cfg2, resume=False)
    ev2 = [r for r in read_jsonl(cfg2.metrics_path) if r.get("event") == "arena"][0]
    assert ev2["score"] == pytest.approx(0.625)
    assert ev2["accepted"] is True
    assert load_latest(cfg2)["generation"] == 1


# --------------------------------------------------------------------------- #
#  8. finally-save: injected interruption and normal completion                #
# --------------------------------------------------------------------------- #

def test_finally_save_on_injected_interruption(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=5, arena_every=10_000)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=17)])

    def interrupt(it):
        if it == 3:
            raise _Interrupt()

    with pytest.raises(_Interrupt):
        train.run(cfg, resume=False, on_iteration=interrupt)

    latest = os.path.join(cfg.checkpoint_dir, "latest.pt")
    assert os.path.exists(latest), "finally must save latest.pt on interruption"
    payload = load_latest(cfg)
    assert payload["iteration"] == 3
    meta = load_meta(cfg)
    assert meta["iteration"] == 3
    assert meta["reason"] == "interrupt"
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "best.pt"))
    assert len(snapshots(cfg)) == 3, "snapshots for all completed iterations"


def test_normal_completion_saves_final_checkpoint(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=3, arena_every=10_000)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=19)])

    train.run(cfg, resume=False)

    meta = load_meta(cfg)
    assert meta["reason"] == "final"
    assert meta["iteration"] == 3
    assert load_latest(cfg)["iteration"] == 3
    assert len(snapshots(cfg)) == 3, "one versioned snapshot per iteration"


# --------------------------------------------------------------------------- #
#  9. legacy v1 checkpoints: clear error on load, never auto-loaded            #
# --------------------------------------------------------------------------- #

def _write_v1_checkpoint(cfg, schema_version=None):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    net = ChessNet(cfg).to("cpu")
    payload = {
        "iteration": 7,
        "net": net.state_dict(),
        "best_net": net.state_dict(),
        "optimizer": torch.optim.Adam(net.parameters()).state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
        "random_rng": __import__("random").getstate(),
        "numpy_rng": np.random.get_state(),
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    torch.save(payload, os.path.join(cfg.checkpoint_dir, "latest.pt"))


def test_legacy_v1_checkpoint_raises_incompatibility(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=1)
    _write_v1_checkpoint(cfg)  # no schema_version -> v1

    with pytest.raises(train.IncompatibleCheckpointError) as ei:
        train._load_latest_v2(cfg)
    assert "schema_version" in str(ei.value), "error must name schema_version"

    with pytest.raises(train.IncompatibleCheckpointError):
        train.run(cfg, resume=True)  # resume refuses v1, never auto-loads

    # an explicit schema_version=1 is equally refused
    cfg2 = make_cfg(tmp_path / "explicit", num_iterations=1)
    _write_v1_checkpoint(cfg2, schema_version=1)
    with pytest.raises(train.IncompatibleCheckpointError):
        train._load_latest_v2(cfg2)


def test_fresh_start_never_auto_loads_legacy_checkpoint(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=1)
    _write_v1_checkpoint(cfg)
    make_fake_selfplay(monkeypatch,
                       lambda: [tuple(e) for e in synthetic_examples(10, seed=21)])

    train.run(cfg, resume=False)  # must NOT raise and must NOT load v1

    payload = load_latest(cfg)
    assert payload["schema_version"] == 2 and payload["iteration"] == 1
    archived = [f for f in os.listdir(cfg.checkpoint_dir)
                if f.startswith("latest-") and f.endswith(".pt")]
    assert len(archived) == 1, "legacy latest.pt must be archived, not loaded"
    legacy = torch.load(os.path.join(cfg.checkpoint_dir, archived[0]),
                        map_location="cpu", weights_only=False)
    assert legacy["iteration"] == 7, "archived legacy payload preserved"


# --------------------------------------------------------------------------- #
#  10. self-play teacher is the accepted best, never the candidate             #
# --------------------------------------------------------------------------- #

def test_selfplay_teacher_is_accepted_best_not_candidate(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, num_iterations=3, arena_every=10_000)  # no arena
    seen = make_fake_selfplay(
        monkeypatch, lambda: [tuple(e) for e in synthetic_examples(10, seed=23)])

    def interrupt(it):
        if it >= 1:
            raise _Interrupt()

    with pytest.raises(_Interrupt):
        train.run(cfg, resume=False, on_iteration=interrupt)

    payload = load_latest(cfg)
    assert len(seen["teachers"]) == 1
    teacher = seen["teachers"][0]
    assert state_dicts_equal(teacher, payload["best"]), \
        "self-play must use the accepted best as teacher"
    assert not state_dicts_equal(teacher, payload["candidate"]), \
        "the trained candidate must never be the self-play teacher"


# --------------------------------------------------------------------------- #
#  11. parallel result envelopes: unpack + timeout + liveness                  #
# --------------------------------------------------------------------------- #

def test_unpack_worker_result_envelopes():
    ex = [("s", "p", 0.0)]
    kind, payload = train._unpack_worker_result({"kind": "game", "examples": ex})
    assert kind == "game" and payload == ex
    kind, payload = train._unpack_worker_result({"kind": "game", "payload": ex})
    assert kind == "game" and payload == ex
    kind, payload = train._unpack_worker_result({"kind": "error",
                                                 "traceback": "boom in worker"})
    assert kind == "error" and "boom" in payload
    kind, payload = train._unpack_worker_result(ex)   # legacy raw list
    assert kind == "game" and payload == ex
    kind, payload = train._unpack_worker_result(("s2", "p2", 0.5))
    assert kind == "game"


class FakeProc:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


def test_collect_games_error_dead_timeout_and_happy_path(tmp_path, monkeypatch):
    import queue as _queue
    cfg = make_cfg(tmp_path, result_timeout_seconds=0.05, games_per_iteration=1)

    # error envelope -> bounded RuntimeError (no hang)
    q = _queue.Queue()
    q.put({"kind": "error", "traceback": "worker crashed"})
    with pytest.raises(RuntimeError, match="worker"):
        train._collect_games(cfg, q, [FakeProc(alive=True)], games_needed=1)

    # dead worker -> bounded RuntimeError
    q = _queue.Queue()
    with pytest.raises(RuntimeError, match="died"):
        train._collect_games(cfg, q, [FakeProc(alive=False)], games_needed=1)

    # live but silent workers -> bounded timeout, not an unbounded Queue.get
    q = _queue.Queue()
    with pytest.raises(RuntimeError, match="timed out"):
        train._collect_games(cfg, q, [FakeProc(alive=True)], games_needed=1)

    # happy path: envelopes and raw lists both deliver games
    q = _queue.Queue()
    q.put({"kind": "game", "examples": [("s1", "p1", 0.0)]})
    q.put([("s2", "p2", 0.5)])
    out = train._collect_games(cfg, q, [FakeProc(alive=True)], games_needed=2)
    assert out == [("s1", "p1", 0.0), ("s2", "p2", 0.5)]
