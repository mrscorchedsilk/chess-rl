#!/usr/bin/env python3
"""Real training controller + telemetry server for the chess-rl AlphaZero trainer.

Manages the `train.py` subprocess and exposes REAL (never simulated) telemetry
over HTTP so the dashboard can start/stop training and chart actual loss,
iteration, arena win-rate, checkpoints and resource usage.

Endpoints:
  GET  /              -> the training dashboard (training.html)
  GET  /api/status    -> live training state + resources + log tail
  GET  /api/history   -> full training.jsonl (loss/iteration/arena per iteration)
  POST /api/control   -> {"action":"start"|"stop", "workers":int, "resume":bool}

Run:  .venv/bin/python train_server.py   (serves on 0.0.0.0:8792)
"""

import json
import os
import signal
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8792

DASHBOARD_DIR = os.path.join(os.path.dirname(HERE), "chess-training-dashboard")
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, "training.html")

PYTHON = os.path.join(HERE, ".venv", "bin", "python")
TRAIN_LOG = os.path.join(HERE, "train_server.log")       # live stdout of train.py
TRAINING_JSONL = os.path.join(HERE, "training.jsonl")
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")


# --------------------------------------------------------------------------- #
#  Train subprocess controller                                                #
# --------------------------------------------------------------------------- #

class Controller:
    """Owns the train.py subprocess lifecycle."""

    def __init__(self):
        self.proc = None
        self.started_at = None
        self.workers = 0
        self.resume = True

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, workers=8, resume=True):
        if self.is_running():
            return {"ok": False, "error": "training already running (pid %s)" % self.proc.pid}
        workers = max(1, int(workers))
        cmd = [PYTHON, "train.py", "--workers", str(workers)]
        if resume:
            cmd.append("--resume")

        try:
            with open(TRAIN_LOG, "a") as f:
                f.write("\n=== start %s  workers=%d  resume=%s ===\n"
                        % (time.ctime(), workers, resume))
        except Exception:
            pass
        logf = open(TRAIN_LOG, "a")
        # Own session/process-group so stop() can kill train.py AND all its
        # parallel self-play worker processes in one signal.
        self.proc = subprocess.Popen(
            cmd, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started_at = time.time()
        self.workers = workers
        self.resume = resume
        return {"ok": True, "pid": self.proc.pid, "workers": workers, "resume": resume}

    def stop(self):
        if not self.is_running():
            return {"ok": False, "error": "not running"}
        pid = self.proc.pid
        pgid = None
        try:
            pgid = os.getpgid(pid)
        except Exception:
            pgid = None

        # 1) Graceful: SIGINT to the parent -> Python raises KeyboardInterrupt,
        #    which unwinds run_parallel's finally block (stop_event -> workers
        #    finish and exit, inference server stops). Clean teardown.
        try:
            os.kill(pid, signal.SIGINT)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # 2) Escalate: SIGTERM the whole process group.
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                # 3) Hard kill the group.
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        pass
                try:
                    self.proc.kill()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=3)
                except Exception:
                    pass
        self.proc = None
        return {"ok": True, "stopped_pid": pid}


CTRL = Controller()


# --------------------------------------------------------------------------- #
#  Real telemetry readers                                                     #
# --------------------------------------------------------------------------- #

def _read_history():
    out = []
    try:
        if os.path.exists(TRAINING_JSONL):
            with open(TRAINING_JSONL) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return out


_RES_CACHE = None


def _read_resources():
    """Live CPU %, RAM %, GPU % + temp (cached ~2s). Real, from /proc + nvidia-smi."""
    global _RES_CACHE
    now = time.time()
    if _RES_CACHE and now - _RES_CACHE[0] < 2.0:
        return _RES_CACHE[1]
    out = {"cpu": None, "ram": None, "gpu": None, "temp": None}
    try:
        n = os.cpu_count() or 1
        out["cpu"] = round(min(100.0, os.getloadavg()[0] / n * 100.0), 1)
    except Exception:
        pass
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                mem[k.strip()] = int(v.split()[0])
        total = mem.get("MemTotal", 1)
        avail = mem.get("MemAvailable", 0)
        out["ram"] = round(max(0.0, (total - avail) / total * 100.0), 1)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            util, _used, _total, temp = r.stdout.strip().split(",")
            out["gpu"] = float(util)
            out["temp"] = float(temp)
    except Exception:
        pass
    _RES_CACHE = (now, out)
    return out


def _checkpoint_count():
    if not os.path.isdir(CHECKPOINT_DIR):
        return 0, {}
    files = os.listdir(CHECKPOINT_DIR)
    snapshots = [f for f in files if f.startswith("ckpt-") and f.endswith(".pt")]
    archived = [f for f in files if f.startswith("best-") and f.endswith(".pt")]
    info = {
        "versioned_snapshots": len(snapshots),
        "archived_best": len(archived),
        "best": "best.pt" in files,
        "latest": "latest.pt" in files,
    }
    return len(snapshots) + len(archived), info


def _log_tail(n=40):
    try:
        with open(TRAIN_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return []
            f.seek(max(0, size - 8192))
            lines = f.read().decode("utf-8", "replace").splitlines()
            return lines[-n:]
    except Exception:
        return []


def _status():
    history = _read_history()
    last = history[-1] if history else None
    running = CTRL.is_running()
    n, ck = _checkpoint_count()
    elapsed = int(time.time() - CTRL.started_at) if CTRL.started_at else None
    return {
        "running": running,
        "pid": CTRL.proc.pid if running else None,
        "started_at": CTRL.started_at,
        "elapsed_s": elapsed,
        "workers": CTRL.workers,
        "resume": CTRL.resume,
        "iteration": last.get("iteration") if last else None,
        "loss": last.get("loss") if last else None,
        "games": last.get("games") if last else None,
        "arena_win_rate": last.get("arena_win_rate") if last else None,
        "accepted": last.get("accepted") if last else None,
        "last_t": last.get("t") if last else None,
        "history_len": len(history),
        "checkpoint_count": n,
        "checkpoints": ck,
        "resources": _read_resources(),
        "log_tail": _log_tail(),
        "server_time": time.time(),
    }


# --------------------------------------------------------------------------- #
#  HTTP server                                                                #
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        try:
            with open(DASHBOARD_FILE, "rb") as f:
                body = f.read()
        except Exception:
            body = b"dashboard not found: training.html"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            try:
                self.rfile.read(length)
            except Exception:
                pass
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html", "/training.html"):
            return self._html()
        if self.path == "/api/status":
            return self._json(_status())
        if self.path == "/api/history":
            return self._json(_read_history())
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/control":
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            data = {}
        action = data.get("action")
        if action == "start":
            result = CTRL.start(
                workers=data.get("workers", 8),
                resume=bool(data.get("resume", True)),
            )
        elif action == "stop":
            result = CTRL.stop()
        else:
            result = {"ok": False, "error": "unknown action %r" % action}
        return self._json(result)

    def log_message(self, *a):
        pass  # keep the console clean


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[train_server] real training controller on http://0.0.0.0:{PORT}/", flush=True)
    print(f"[train_server] dashboard:  http://100.84.103.120:{PORT}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
