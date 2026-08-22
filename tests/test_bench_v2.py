"""Regression tests for benchmark tools on CPU-only and CUDA hosts."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_sync_device_is_safe_on_cpu(monkeypatch):
    import bench_mcts

    called = []
    monkeypatch.setattr(bench_mcts.torch.cuda, "synchronize", lambda: called.append(True))
    bench_mcts.sync_device("cpu")
    assert called == []


def test_sync_device_uses_cuda_for_cuda_device(monkeypatch):
    import bench_mcts

    called = []
    monkeypatch.setattr(bench_mcts.torch.cuda, "synchronize", lambda: called.append(True))
    bench_mcts.sync_device("cuda")
    assert called == [True]
