"""Sprint A — controller reliability/security + dashboard truthfulness (strict TDD).

Every test runs against train_server.py with TEMP fake commands (never real
training — train.py is never invoked) or monkeypatched controller state.

RED set (fails on the baseline train_server.py / training.html, passes after
the Sprint A implementation).
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import train_server as ts  # noqa: E402

DASHBOARD = os.path.join(
    os.path.dirname(REPO), "chess-training-dashboard", "training.html"
)


# --------------------------------------------------------------------------- #
#  helpers / fixtures                                                         #
# --------------------------------------------------------------------------- #

def write_fake_train(tmp_path, body):
    """Write an executable fake train command (temp controller command)."""
    p = tmp_path / "fake_train.py"
    p.write_text("#!/usr/bin/env python3\n" + body)
    os.chmod(p, 0o755)
    return str(p)


def write_jsonl(path, records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def wait_until(fn, timeout=15.0, interval=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(interval)
    return False


class FakeProc:
    """Minimal stand-in for a Popen: poll() None == running."""

    def __init__(self, pid=4242, rc=None):
        self.pid = pid
        self._rc = rc

    def poll(self):
        return self._rc


@pytest.fixture()
def ctrl(tmp_path, monkeypatch):
    """Fresh Controller + temp paths + fake train command.

    PYTHON is pointed at the fake script itself so even the BASELINE server
    (which hardcodes ['<PYTHON>', 'train.py', ...]) can never run real training.
    """
    c = ts.Controller()
    monkeypatch.setattr(ts, "CTRL", c)
    monkeypatch.setattr(ts, "PYTHON", str(tmp_path / "fake_train.py"))
    monkeypatch.setattr(ts, "TRAIN_LOG", str(tmp_path / "train_server.log"))
    monkeypatch.setattr(ts, "TRAINING_JSONL", str(tmp_path / "training.jsonl"))
    monkeypatch.setattr(ts, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    (tmp_path / "checkpoints").mkdir()
    return c


# --------------------------------------------------------------------------- #
#  security: bind host                                                        #
# --------------------------------------------------------------------------- #

def test_default_bind_host_is_loopback(monkeypatch):
    monkeypatch.delenv("CHESS_TRAIN_HOST", raising=False)
    monkeypatch.delenv("CHESS_TRAIN_PORT", raising=False)
    host, port = ts._bind_config()
    assert host == "127.0.0.1", "default bind must be loopback, never 0.0.0.0"
    assert not host.startswith("0.0")
    assert port == 8792


def test_bind_host_and_port_env_override(monkeypatch):
    monkeypatch.setenv("CHESS_TRAIN_HOST", "10.9.8.7")
    monkeypatch.setenv("CHESS_TRAIN_PORT", "9999")
    host, port = ts._bind_config()
    assert host == "10.9.8.7"
    assert port == 9999


# --------------------------------------------------------------------------- #
#  controller: thread lock around start/stop                                  #
# --------------------------------------------------------------------------- #

def test_concurrent_start_exactly_one_wins(ctrl, tmp_path, monkeypatch):
    write_fake_train(tmp_path, "import time\nwhile True: time.sleep(30)\n")
    real_popen = subprocess.Popen
    spawned = []
    recorded_cmds = []

    def delayed_popen(cmd, **kw):
        time.sleep(0.25)  # widen the race window: baseline double-starts here
        p = real_popen(cmd, **kw)
        spawned.append(p)
        recorded_cmds.append(cmd)
        return p

    monkeypatch.setattr(ts.subprocess, "Popen", delayed_popen)
    try:
        results = [None] * 8

        def worker(i):
            results[i] = ctrl.start(workers=8, resume=True)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        oks = [r for r in results if r and r.get("ok")]
        assert len(oks) == 1, "exactly one concurrent start must win: %r" % results
        assert ctrl.is_running()
        # never real training: the executable must always be the fake script
        assert all(
            os.path.basename(c[0]) == "fake_train.py" for c in recorded_cmds
        )
    finally:
        for p in spawned:
            try:
                p.kill()
            except Exception:
                pass


def test_concurrent_stop_single_winner(ctrl, tmp_path, monkeypatch):
    write_fake_train(tmp_path, "import time\nwhile True: time.sleep(30)\n")
    assert ctrl.start(workers=8)["ok"]
    real_kill = os.kill

    def delayed_kill(pid, sig):
        time.sleep(0.25)  # widen the race: baseline reports two successful stops
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", delayed_kill)
    results = [None] * 4

    def worker(i):
        results[i] = ctrl.stop()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    oks = [r for r in results if r and r.get("ok")]
    assert len(oks) == 1, "exactly one concurrent stop must win: %r" % results
    assert not ctrl.is_running()


def test_start_after_exit_restarts(ctrl, tmp_path):
    write_fake_train(tmp_path, "import sys\nsys.exit(0)\n")
    assert ctrl.start()["ok"]
    assert wait_until(lambda: not ctrl.is_running(), timeout=15)
    assert ctrl.start()["ok"], "must be able to restart after a clean exit"
    ctrl.stop()


def test_start_records_error_when_spawn_fails(ctrl, monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("no such interpreter")

    monkeypatch.setattr(ts.subprocess, "Popen", boom)
    r = ctrl.start()
    assert not r["ok"]
    assert ctrl.error is not None
    assert "no such interpreter" in ctrl.error


# --------------------------------------------------------------------------- #
#  controller: return code / error tracking                                   #
# --------------------------------------------------------------------------- #

def test_return_code_and_error_tracked_on_unexpected_exit(ctrl, tmp_path):
    write_fake_train(tmp_path, "import sys\nsys.exit(5)\n")
    assert ctrl.start()["ok"]
    assert wait_until(lambda: not ctrl.is_running(), timeout=15)
    assert ctrl.returncode == 5
    assert ctrl.error and "5" in ctrl.error
    st = ts._status()
    assert st["running"] is False
    assert st["exit_code"] == 5
    assert st["error"] == ctrl.error
    assert st["stopped_at"] is not None


def test_clean_exit_has_no_error(ctrl, tmp_path):
    write_fake_train(tmp_path, "import sys\nsys.exit(0)\n")
    assert ctrl.start()["ok"]
    assert wait_until(lambda: not ctrl.is_running(), timeout=15)
    assert ctrl.returncode == 0
    assert ctrl.error is None
    assert ts._status()["error"] is None


# --------------------------------------------------------------------------- #
#  status: metrics associated with the ACTIVE run (run_id / pid / start time)  #
# --------------------------------------------------------------------------- #

def test_status_associates_metrics_with_active_run(ctrl, tmp_path):
    write_jsonl(ts.TRAINING_JSONL, [
        {"t": 1000.0, "run_id": "run-A", "iteration": 1, "loss": 9.0, "games": 20},
        {"t": 1010.0, "run_id": "run-A", "iteration": 2, "loss": 8.5, "games": 40},
        {"t": 2000.0, "run_id": "run-B", "generation": 2, "iteration": 1,
         "total_loss": 7.0, "policy_loss": 4.0, "value_loss": 3.0,
         "entropy": 1.1, "optimizer_steps": 3, "replay_size": 150,
         "arena_score": 0.55, "arena_w": 11, "arena_l": 8, "arena_d": 1,
         "accepted": True, "saved_iteration": 1},
        {"t": 2005.0, "run_id": "run-B", "generation": 2, "iteration": 2,
         "total_loss": 6.0, "policy_loss": 3.0, "value_loss": 3.0,
         "entropy": 1.0, "optimizer_steps": 6, "replay_size": 300,
         "accepted": None, "saved_iteration": 1},
        {"t": 2010.0, "run_id": "run-B", "generation": 2, "iteration": 3,
         "total_loss": 5.0, "policy_loss": 2.5, "value_loss": 2.5,
         "entropy": 0.9, "optimizer_steps": 9, "replay_size": 450,
         "accepted": None, "saved_iteration": 2},
    ])
    ctrl.proc = FakeProc(pid=777)
    ctrl.started_at = 1999.0
    ctrl.workers = 8
    ctrl.resume = True
    st = ts._status()
    assert st["running"] is True
    assert st["pid"] == 777
    assert st["started_at"] == 1999.0
    assert st["run_id"] == "run-B", "metrics must be scoped to the ACTIVE run"
    assert st["generation"] == 2
    assert st["live_iteration"] == 3
    assert st["iteration"] == 3  # back-compat alias
    assert st["saved_iteration"] == 2
    assert st["saved_iteration"] != st["live_iteration"]
    assert st["total_loss"] == 5.0
    assert st["policy_loss"] == 2.5
    assert st["value_loss"] == 2.5
    assert st["loss"] == 5.0  # total alias
    assert st["entropy"] == 0.9
    assert st["optimizer_steps"] == 9
    assert st["replay_size"] == 450
    assert st["arena"]["score"] == 0.55
    assert st["arena"]["win_rate"] == 0.55
    assert st["arena"]["w"] == 11 and st["arena"]["l"] == 8 and st["arena"]["d"] == 1


def test_status_stale_when_running_without_fresh_metrics(ctrl, tmp_path, monkeypatch):
    ctrl.proc = FakeProc(pid=1)
    ctrl.started_at = time.time() - 600
    assert ts._status()["stale"] is True

    write_jsonl(ts.TRAINING_JSONL, [
        {"t": time.time() - 10, "run_id": "run-X", "iteration": 4, "loss": 1.0}
    ])
    assert ts._status()["stale"] is False

    c2 = ts.Controller()
    c2.proc = FakeProc(pid=2)
    c2.started_at = time.time()  # just started: not stale yet, even with no records
    monkeypatch.setattr(ts, "CTRL", c2)
    assert ts._status()["stale"] is False


# --------------------------------------------------------------------------- #
#  status: saved_iteration via lightweight checkpoint_meta.json (NO torch)     #
# --------------------------------------------------------------------------- #

def test_checkpoint_meta_read_lightweight_without_torch(ctrl, tmp_path):
    """saved_iteration/run identity come from checkpoint_meta.json — status
    polling must never torch.load latest.pt (which may embed a huge replay
    buffer)."""
    write_jsonl(ts.TRAINING_JSONL, [
        {"t": 100.0, "run_id": "run-M", "iteration": 9, "loss": 1.0}
    ])
    meta = {
        "schema_version": 1,
        "saved_iteration": 12,
        "run_id": "run-M",
        "generation": 3,
        "replay_size": 12345,
        "saved_at": 1234.5,
    }
    with open(os.path.join(ts.CHECKPOINT_DIR, "checkpoint_meta.json"), "w") as f:
        json.dump(meta, f)

    ctrl.proc = FakeProc(pid=9)
    ctrl.started_at = 50.0
    ctrl.run_id = "run-9-50"
    st = ts._status()

    # the server module must never import torch at all -> no torch.load possible
    assert not hasattr(ts, "torch"), "train_server must not import torch"
    assert st["saved_iteration"] == 12
    assert st["live_iteration"] == 9
    assert st["checkpoint_meta"]["saved_iteration"] == 12
    assert st["checkpoint_meta"]["run_id"] == "run-M"


def test_checkpoint_meta_missing_falls_back_gracefully(ctrl, tmp_path):
    write_jsonl(ts.TRAINING_JSONL, [
        {"t": 100.0, "run_id": "run-F", "iteration": 9, "loss": 1.0}
    ])
    (tmp_path / "checkpoints" / "ckpt-iter0007-20260101-000000.pt").touch()
    ctrl.proc = FakeProc(pid=3)
    ctrl.started_at = 50.0
    st = ts._status()
    assert st["checkpoint_meta"] == {}
    assert st["saved_iteration"] == 7  # filename fallback, still no torch
    assert st["live_iteration"] == 9
    assert st["run_id"] == "run-F"


# --------------------------------------------------------------------------- #
#  checkpoints: honest counts, no misleading '0 checkpoints'                   #
# --------------------------------------------------------------------------- #

def test_checkpoint_info_not_misleading(ctrl, tmp_path):
    ck = ts._checkpoint_info()
    assert ck["any"] is False and ck["best"] is False and ck["latest"] is False
    assert ck["versioned_snapshots"] == 0 and ck["archived_best"] == 0
    assert ts._status()["checkpoint_count"] == 0

    d = tmp_path / "checkpoints"
    (d / "best.pt").touch()
    (d / "latest.pt").touch()
    (d / "ckpt-iter0005-20260101-000000.pt").touch()
    (d / "best-20260101-000000.pt").touch()
    ck = ts._checkpoint_info()
    assert ck["best"] is True and ck["latest"] is True
    assert ck["versioned_snapshots"] == 1
    assert ck["archived_best"] == 1
    assert ck["any"] is True
    assert ts._status()["checkpoint_count"] == 4  # best + latest + 1 snap + 1 arch


# --------------------------------------------------------------------------- #
#  headroom: max(CPU, GPU, RAM), not only CPU/GPU                              #
# --------------------------------------------------------------------------- #

def test_headroom_uses_max_of_cpu_gpu_ram():
    assert ts._headroom(10.0, 90.0, 20.0) == pytest.approx(0.10)
    # old behavior (CPU/GPU only) would report 0.80 — the bug being fixed
    assert ts._headroom(10.0, 90.0, 20.0) != pytest.approx(0.80)
    assert ts._headroom(None, 90.0, None) == pytest.approx(0.10)
    assert ts._headroom(95.0, 10.0, None) == pytest.approx(0.05)
    assert ts._headroom(None, None, None) is None


def test_resources_include_ram_aware_headroom(monkeypatch):
    monkeypatch.setattr(ts, "_RES_CACHE", None)
    res = ts._read_resources()
    assert "headroom" in res
    vals = [v for v in (res["cpu"], res["ram"], res["gpu"]) if v is not None]
    if vals:
        assert res["headroom"] == pytest.approx(max(0.0, 1.0 - max(vals) / 100.0),
                                                abs=0.01)
    else:
        assert res["headroom"] is None


# --------------------------------------------------------------------------- #
#  history: robust to multiple runs + arena points event-only                  #
# --------------------------------------------------------------------------- #

def test_history_segments_multiple_runs(ctrl, tmp_path):
    write_jsonl(ts.TRAINING_JSONL, [
        {"t": 1000.0, "run_id": "run-A", "iteration": 1, "loss": 9.0,
         "arena_win_rate": 0.5},
        {"t": 1010.0, "run_id": "run-A", "iteration": 2, "loss": 8.0,
         "arena_win_rate": 0.5},  # carried forward -> not an arena event
        {"t": 2000.0, "run_id": "run-B", "iteration": 1, "loss": 7.0},
        {"t": 2010.0, "run_id": "run-B", "iteration": 2, "loss": 6.0,
         "arena_win_rate": 0.4},
    ])
    h = ts._history()
    assert h["total"] == 4
    assert [r["run_id"] for r in h["runs"]] == ["run-A", "run-B"]
    run_b = h["runs"][1]
    assert [r["iteration"] for r in run_b["records"]] == [1, 2]
    assert run_b["run_id"] == "run-B"
    assert h["runs"][0]["arena_points"] == [h["runs"][0]["records"][0]]
    assert h["runs"][1]["arena_points"] == [h["runs"][1]["records"][1]]


def test_legacy_records_without_run_id_segmented(ctrl, tmp_path):
    write_jsonl(ts.TRAINING_JSONL, [
        {"t": 1000.0, "iteration": 1, "loss": 9.0, "games": 20},
        {"t": 1010.0, "iteration": 2, "loss": 8.0, "games": 40},
        {"t": 5000.0, "iteration": 1, "loss": 7.0, "games": 4},  # new run
    ])
    h = ts._history()
    assert h["total"] == 3
    assert len(h["runs"]) == 2
    assert h["runs"][0]["run_id"] != h["runs"][1]["run_id"]
    assert h["runs"][1]["records"][0]["run_id"] == h["runs"][1]["run_id"]


def test_arena_points_event_only():
    recs = [
        {"t": 1, "iteration": 1, "arena_win_rate": 0.5},
        {"t": 2, "iteration": 2, "arena_win_rate": 0.5},          # carried
        {"t": 3, "iteration": 3, "arena_win_rate": 0.6},          # new result
        {"t": 4, "iteration": 4, "arena_win_rate": 0.6, "arena": True},  # explicit
        {"t": 5, "iteration": 5, "arena_score": 0.3},             # new-format
        {"t": 6, "iteration": 6, "arena_score": 0.3},             # carried
    ]
    flags = [ts._is_arena_event(recs[i], recs[i - 1] if i else None)
             for i in range(len(recs))]
    assert flags == [True, False, True, True, True, False]


# --------------------------------------------------------------------------- #
#  dashboard: static contract (run id, gen, live/saved iter, split loss,       #
#  replay/opt steps, checkpoint labels, RAM headroom, workers=8, same-origin,  #
#  stale/offline/error states)                                                 #
# --------------------------------------------------------------------------- #

def test_dashboard_static_contract():
    html = open(DASHBOARD, encoding="utf-8").read()
    for ident in [
        "t-run", "t-gen", "t-iter-live", "t-iter-saved",
        "t-loss-total", "t-loss-policy", "t-loss-value",
        "t-replay", "t-opt-steps", "res-head", "offline", "errbanner",
    ]:
        assert ('id="%s"' % ident) in html, "dashboard missing #%s" % ident
    # worker recommendation 8
    assert 'id="inp-workers"' in html
    assert 'value="8"' in html
    assert "sweet spot" in html
    # same-origin: no hardcoded LAN/Tailscale IP; apiBase defaults to ''
    assert "100.84" not in html
    assert "return q || ''" in html
    # RAM-aware headroom (client-side fallback must consider RAM)
    assert "r.ram" in html
    # checkpoints: never the misleading '0 checkpoints'; has a none-yet label
    assert "0 checkpoints" not in html
    assert "none yet" in html
    # clear stale/offline/error states
    assert "stale" in html.lower()
    assert "offline" in html.lower()
    assert "errbanner" in html


def test_dashboard_javascript_test_runs_clean():
    """The optional node static test must exist and pass."""
    js = os.path.join(os.path.dirname(DASHBOARD), "test_dashboard.js")
    assert os.path.exists(js), "missing chess-training-dashboard/test_dashboard.js"
    r = subprocess.run(["node", js], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, "node test_dashboard.js failed:\n%s" % r.stdout + r.stderr
    assert "PASS" in r.stdout
