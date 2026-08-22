"""Tests for opt-in, low-overhead parallel-pipeline diagnostics."""

import json
import os
import queue
import threading
from types import SimpleNamespace

import numpy as np
import torch

import parallel
from benchmarks.parallel_pipeline import (
    _seed_benchmark,
    host_fingerprint,
    parse_args,
    summarize_profile,
)


def test_profile_event_is_noop_when_disabled(tmp_path, monkeypatch):
    path = tmp_path / "profile.jsonl"
    monkeypatch.delenv("CHESS_PROFILE_JSONL", raising=False)

    parallel._profile_event("worker_started", worker=3)

    assert not path.exists()


def test_profile_event_appends_machine_readable_process_attributed_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "profile.jsonl"
    monkeypatch.setenv("CHESS_PROFILE_JSONL", str(path))

    parallel._profile_event("worker_started", worker=3)
    parallel._profile_event("batch_formed", jobs=4, positions=128)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["worker_started", "batch_formed"]
    assert rows[0]["worker"] == 3
    assert rows[1]["jobs"] == 4
    assert rows[1]["positions"] == 128
    for row in rows:
        assert row["pid"] == os.getpid()
        assert isinstance(row["wall_time"], float)
        assert isinstance(row["monotonic_ns"], int)


def test_inference_client_profiles_request_and_reply_latency(tmp_path, monkeypatch):
    path = tmp_path / "profile.jsonl"
    monkeypatch.setenv("CHESS_PROFILE_JSONL", str(path))
    requests = queue.Queue()
    replies = queue.Queue()
    replies.put((np.zeros((1, 4672), dtype=np.float32),
                 np.zeros((1, 1), dtype=np.float32), "dense"))
    client = parallel.InferenceClient(requests, replies, policy_size=4672, timeout=1)

    client(torch.zeros((1, 104, 8, 8), dtype=torch.float32))

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "inference_request_sent", "inference_reply_received"
    ]
    assert rows[0]["positions"] == 1
    assert rows[1]["positions"] == 1
    assert rows[1]["kind"] == "dense"
    assert rows[1]["wait_ms"] >= 0.0


def test_inference_server_profiles_batch_transfer_forward_and_reply(tmp_path, monkeypatch):
    path = tmp_path / "profile.jsonl"
    monkeypatch.setenv("CHESS_PROFILE_JSONL", str(path))

    class TinyNet(torch.nn.Module):
        def forward(self, x):
            return (torch.zeros((len(x), 4672), dtype=torch.float32),
                    torch.zeros((len(x), 1), dtype=torch.float32))

    response = queue.Queue()
    server = parallel.InferenceServer(
        TinyNet(), "cpu", [], sparse_response=False,
    )
    server._process([(response, np.zeros((2, 104, 8, 8), dtype=np.float32))])

    payload, values, kind = response.get_nowait()
    assert payload.shape == (2, 4672)
    assert values.shape == (2, 1)
    assert kind == "dense"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "batch_formed", "inference_batch_completed", "inference_reply_sent"
    ]
    metrics = rows[1]
    assert metrics["positions"] == 2
    assert metrics["jobs"] == 1
    for key in ("h2d_ms", "forward_ms", "d2h_ms", "legal_ms", "total_ms"):
        assert metrics[key] >= 0.0


def test_worker_profiles_lifecycle_ply_and_completed_game(tmp_path, monkeypatch):
    path = tmp_path / "profile.jsonl"
    monkeypatch.setenv("CHESS_PROFILE_JSONL", str(path))
    stop = threading.Event()

    def fake_play_game(_client, _cfg, on_ply=None):
        assert on_ply is not None
        on_ply(1)
        stop.set()
        return [("state", "policy", 0.0)]

    monkeypatch.setattr(parallel, "play_game", fake_play_game)
    results = queue.Queue()
    cfg = SimpleNamespace(
        device="cpu", policy_size=4672, result_timeout_seconds=1,
    )

    parallel.worker_loop(
        cfg, queue.Queue(), queue.Queue(), results, seed=7,
        stop_event=stop, worker_id=2,
    )

    envelope = results.get_nowait()
    assert envelope["kind"] == "game"
    assert envelope["worker"] == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "worker_started", "game_started", "ply_completed", "game_completed"
    ]
    assert rows[2]["ply"] == 1
    assert rows[3]["plies"] == 1
    assert rows[3]["examples"] == 1


def test_summarize_profile_reports_batch_and_transfer_breakdown():
    rows = [
        {"event": "batch_formed", "positions": 64, "jobs": 4},
        {"event": "batch_formed", "positions": 32, "jobs": 2},
        {"event": "inference_batch_completed", "positions": 64, "jobs": 4,
         "h2d_ms": 1.0, "forward_ms": 5.0, "d2h_ms": 2.0,
         "legal_ms": 4.0, "total_ms": 12.0},
        {"event": "inference_batch_completed", "positions": 32, "jobs": 2,
         "h2d_ms": 0.5, "forward_ms": 3.0, "d2h_ms": 1.0,
         "legal_ms": 2.0, "total_ms": 6.5},
        {"event": "game_completed", "plies": 80, "examples": 80,
         "game_ms": 1000.0},
    ]

    summary = summarize_profile(rows)

    assert summary["games_completed"] == 1
    assert summary["plies_completed"] == 80
    assert summary["batch_positions"]["min"] == 32
    assert summary["batch_positions"]["median"] == 48.0
    assert summary["batch_positions"]["max"] == 64
    assert summary["timing_ms"]["h2d"] == 1.5
    assert summary["timing_ms"]["forward"] == 8.0
    assert summary["timing_ms"]["d2h"] == 3.0
    assert summary["timing_ms"]["legal"] == 6.0
    assert summary["transfer_fraction_of_measured_batch"] == 4.5 / 18.5


def test_host_fingerprint_records_reproducibility_context():
    fingerprint = host_fingerprint()

    assert fingerprint["python"]
    assert fingerprint["torch"]
    assert len(fingerprint["git_commit"]) == 40
    assert fingerprint["git_dirty"] in (True, False)
    assert "cuda_runtime" in fingerprint
    assert "gpu" in fingerprint


def test_benchmark_seed_reproduces_parent_numpy_and_torch_rng():
    _seed_benchmark(42)
    first_np = np.random.random(4)
    first_torch = torch.rand(4)

    _seed_benchmark(42)
    second_np = np.random.random(4)
    second_torch = torch.rand(4)

    assert np.array_equal(first_np, second_np)
    assert torch.equal(first_torch, second_torch)


def test_benchmark_profiling_is_opt_in():
    args = parse_args([])
    assert args.profile_jsonl is None
