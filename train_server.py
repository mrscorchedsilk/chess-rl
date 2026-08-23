#!/usr/bin/env python3
"""Real training controller + telemetry server for the chess-rl AlphaZero trainer.

Manages the `train.py` subprocess and exposes REAL (never simulated) telemetry
over HTTP so the dashboard can start/stop training and chart actual loss,
iteration, arena win-rate, checkpoints and resource usage.

Endpoints:
  GET  /              -> the training dashboard (training.html)
  GET  /api/status    -> live training state + resources + log tail
  GET  /api/history   -> training.jsonl segmented into runs (arena points
                         event-only)
  POST /api/control   -> {"action":"start"|"stop", "workers":int, "resume":bool}

Sprint A (reliability + security):
  - Controller start/stop are serialized by a thread lock; concurrent control
    requests can never double-start or corrupt state.
  - Default bind host is 127.0.0.1 (loopback); override with CHESS_TRAIN_HOST
    (port with CHESS_TRAIN_PORT). Unauthenticated control is NEVER exposed on
    0.0.0.0 by default, and responses carry no wildcard CORS.
  - Process return codes and errors are tracked and surfaced in /api/status.
  - Status metrics are associated with the ACTIVE run (run_id / pid / start
    time); live_iteration and saved_iteration are separate fields.
  - saved_iteration is read from checkpoint_meta.json (lightweight JSON beside
    latest.pt) or from metrics records / versioned snapshot filenames — status
    polling NEVER torch.loads latest.pt (it may embed a huge replay buffer).
  - Checkpoint info reports best/latest existence + versioned snapshot counts
    (no misleading bare '0 checkpoints').
  - Headroom = 1 - max(CPU, GPU, RAM) / 100 (RAM-aware).

Run:  .venv/bin/python train_server.py   (serves on http://127.0.0.1:8792/)
"""

import json
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8792

DASHBOARD_DIR = os.path.join(os.path.dirname(HERE), "chess-training-dashboard")
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, "training.html")

PYTHON = os.path.join(HERE, ".venv", "bin", "python")
TRAIN_SCRIPT = "train.py"                       # tests point this at temp fakes
TRAIN_LOG = os.path.join(HERE, "train_server.log")       # live stdout of train.py
TRAINING_JSONL = os.path.join(HERE, "training.jsonl")
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints", "v2")

STALE_S = 300        # running without fresh metrics this long -> STALE
LEGACY_GAP_S = 900   # t-gap that splits legacy (no run_id) records into new runs


def _bind_config():
    """(host, port) for the HTTP server.

    Default host is loopback 127.0.0.1 — unauthenticated control endpoints must
    not be reachable from the network by default. CHESS_TRAIN_HOST /
    CHESS_TRAIN_PORT opt in to exposing it.
    """
    host = os.environ.get("CHESS_TRAIN_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("CHESS_TRAIN_PORT", PORT))
    except ValueError:
        port = PORT
    return host, port


# --------------------------------------------------------------------------- #
#  Train subprocess controller                                                #
# --------------------------------------------------------------------------- #

class Controller:
    """Owns the train.py subprocess lifecycle.

    All lifecycle transitions (start / stop / is_running / reap) are
    serialized by `lock` (an RLock), so concurrent control requests can't
    double-start training or tear state apart.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.started_at = None
        self.stopped_at = None
        self.workers = 0
        self.resume = True
        self.run_id = None
        self.returncode = None
        self.error = None

    # -- internals ---------------------------------------------------------- #

    def _reap(self):
        """Finalize a dead subprocess: capture return code + error."""
        if self.proc is None:
            return
        rc = self.proc.poll()
        if rc is None:
            return
        self.returncode = rc
        self.stopped_at = time.time()
        if rc != 0:
            self.error = "training exited unexpectedly (exit code %s)" % rc
        self.proc = None

    def is_running(self):
        with self.lock:
            self._reap()
            return self.proc is not None and self.proc.poll() is None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, workers=8, resume=True, backend="python",
              num_simulations=None, games_per_iteration=None,
              num_iterations=None, arena_every=None,
              warm_start_checkpoint=None, checkpoint_dir=None):
        with self.lock:
            self._reap()
            if self.proc is not None:
                return {"ok": False,
                        "error": "training already running (pid %s)" % self.proc.pid}
            workers = max(1, int(workers))
            if backend == "native":
                cmd = [PYTHON, TRAIN_SCRIPT, "--selfplay-backend", "native"]
            else:
                cmd = [PYTHON, TRAIN_SCRIPT, "--workers", str(workers)]
            if warm_start_checkpoint is not None:
                # Weights-only warm start (new lineage): mutually exclusive with
                # --resume, so it suppresses the resume flag below.
                cmd += ["--warm-start-checkpoint", str(warm_start_checkpoint)]
            elif resume:
                cmd.append("--resume")
            if checkpoint_dir is not None:
                cmd += ["--checkpoint-dir", str(checkpoint_dir)]
            if num_simulations is not None:
                cmd += ["--num-simulations", str(int(num_simulations))]
            if games_per_iteration is not None:
                cmd += ["--games-per-iteration", str(int(games_per_iteration))]
            if num_iterations is not None:
                cmd += ["--num-iterations", str(int(num_iterations))]
            if arena_every is not None:
                cmd += ["--arena-every", str(int(arena_every))]

            try:
                with open(TRAIN_LOG, "a") as f:
                    f.write("\n=== start %s  workers=%d  resume=%s ===\n"
                            % (time.ctime(), workers, resume))
            except Exception:
                pass
            logf = open(TRAIN_LOG, "a")
            try:
                # Own session/process-group so stop() can kill train.py AND all
                # its parallel self-play worker processes in one signal.
                self.proc = subprocess.Popen(
                    cmd, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as e:
                self.proc = None
                self.error = "failed to start training: %s" % e
                return {"ok": False, "error": self.error}

            self.started_at = time.time()
            self.stopped_at = None
            self.returncode = None
            self.error = None
            self.workers = workers
            self.resume = resume
            self.backend = backend
            self.run_id = "run-%d-%d" % (self.proc.pid, int(self.started_at))
            return {"ok": True, "pid": self.proc.pid, "workers": workers,
                    "resume": resume, "backend": backend,
                    "run_id": self.run_id}

    def stop(self):
        with self.lock:
            self._reap()
            if self.proc is None:
                return {"ok": False, "error": "not running",
                        "exit_code": self.returncode, "error_detail": self.error}
            pid = self.proc.pid
            pgid = None
            try:
                pgid = os.getpgid(pid)
            except Exception:
                pgid = None

            # 1) Graceful: SIGINT -> Python raises KeyboardInterrupt, which
            #    unwinds run_parallel's finally block cleanly.
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
            self.returncode = self.proc.returncode
            self.stopped_at = time.time()
            # An intentional stop is not an error even if the SIGINT-induced
            # exit code is non-zero; unexpected exits are flagged by _reap().
            self.proc = None
            return {"ok": True, "stopped_pid": pid, "exit_code": self.returncode}


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


def _is_arena_event(rec, prev):
    """True when `rec` is a NEW arena result, not a carried-forward value.

    Rules:
      - explicit ``arena: true`` / ``arena_event: true`` -> event
      - new-format ``arena_score`` present and changed vs prev -> event
      - new-format arena W/L/D present and any changed vs prev -> event
      - legacy ``arena_win_rate`` present and changed vs prev -> event
    """
    if rec.get("arena") is True or rec.get("arena_event") is True:
        return True
    score = rec.get("arena_score")
    if score is not None:
        return prev is None or score != prev.get("arena_score")
    if any(k in rec for k in ("arena_w", "arena_l", "arena_d")):
        cur = [rec.get(k) for k in ("arena_w", "arena_l", "arena_d")]
        if any(v is not None for v in cur):
            prv = [prev.get(k) if prev else None
                   for k in ("arena_w", "arena_l", "arena_d")]
            return cur != prv
    awr = rec.get("arena_win_rate")
    if awr is not None:
        return prev is None or prev.get("arena_win_rate") != awr
    return False


def _history():
    """Full JSONL history, segmented into runs and enriched with run_id +
    arena_event flags (arena points are event-only)."""
    raw = _read_history()
    runs = []
    current = None
    records = []
    for rec in raw:
        t = rec.get("t")
        rid = rec.get("run_id")
        prev_rec = current["records"][-1] if current and current["records"] else None
        prev_t = prev_rec.get("t") if prev_rec else None

        if rid is None and current is not None:
            same = (t is not None and prev_t is not None
                    and 0 <= t - prev_t <= LEGACY_GAP_S)
            games = rec.get("games")
            prev_games = prev_rec.get("games") if prev_rec else None
            if same and not (isinstance(games, (int, float))
                             and isinstance(prev_games, (int, float))
                             and games < prev_games):
                rid = current["run_id"]
        if rid is None:
            rid = "legacy-%d" % int(t or time.time())

        if current is None or current["run_id"] != rid:
            if current is not None:
                current["stopped_at"] = prev_t
            current = {"run_id": rid, "started_at": t, "stopped_at": None,
                       "records": [], "arena_points": []}
            runs.append(current)

        prev_rec = current["records"][-1] if current["records"] else None
        e = dict(rec)
        e["run_id"] = rid
        e["arena_event"] = _is_arena_event(rec, prev_rec)
        current["records"].append(e)
        if e["arena_event"]:
            current["arena_points"].append(e)
        records.append(e)
    if current is not None:
        current["stopped_at"] = current["records"][-1].get("t")
    for i, run in enumerate(runs):
        run["index"] = i
    return {"runs": runs, "records": records, "total": len(records)}


_RES_CACHE = None


def _headroom(cpu, ram, gpu):
    """1 - max(CPU, GPU, RAM)/100, over whichever metrics are available.

    RAM is included: a RAM-starved box must not look 'healthy' just because
    CPU/GPU are idle.
    """
    vals = [v for v in (cpu, ram, gpu) if isinstance(v, (int, float))]
    if not vals:
        return None
    used = max(min(100.0, float(v)) for v in vals)
    return max(0.0, 1.0 - used / 100.0)


def _read_resources():
    """Live CPU %, RAM %, GPU % + temp (cached ~2s). Real, from /proc + nvidia-smi."""
    global _RES_CACHE
    now = time.time()
    if _RES_CACHE and now - _RES_CACHE[0] < 2.0:
        return _RES_CACHE[1]
    out = {"cpu": None, "ram": None, "gpu": None, "temp": None, "headroom": None}
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
    out["headroom"] = _headroom(out["cpu"], out["ram"], out["gpu"])
    _RES_CACHE = (now, out)
    return out


def _checkpoint_info():
    """Honest checkpoint inventory: best/latest existence + versioned counts."""
    info = {"any": False, "best": False, "latest": False,
            "versioned_snapshots": 0, "archived_best": 0}
    if not os.path.isdir(CHECKPOINT_DIR):
        return info
    try:
        files = os.listdir(CHECKPOINT_DIR)
    except Exception:
        return info
    snapshots = [f for f in files if re.match(r"^ckpt-iter\d+-.*\.pt$", f)]
    archived = [f for f in files if f.startswith("best-") and f.endswith(".pt")]
    info["versioned_snapshots"] = len(snapshots)
    info["archived_best"] = len(archived)
    info["best"] = "best.pt" in files
    info["latest"] = "latest.pt" in files
    info["any"] = bool(info["best"] or info["latest"] or snapshots or archived)
    return info


def _checkpoint_count():
    ck = _checkpoint_info()
    return (ck["versioned_snapshots"] + ck["archived_best"]
            + int(ck["best"]) + int(ck["latest"]))


def _checkpoint_meta():
    """Lightweight metadata written beside latest.pt by train.py.

    schema_version, iteration, run_id, generation, replay_size, saved_at.
    Read as plain JSON — NEVER torch.load: latest.pt may embed a huge replay
    buffer and must not be touched on the status polling path.
    """
    path = os.path.join(CHECKPOINT_DIR, "checkpoint_meta.json")
    try:
        with open(path) as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _saved_iteration(recs, meta):
    """saved_iteration: freshest record value -> checkpoint_meta.json ->
    versioned snapshot filenames (all lightweight)."""
    for r in reversed(recs):
        si = r.get("saved_iteration")
        if isinstance(si, (int, float)) and si > 0:
            return int(si)
    si = meta.get("saved_iteration", meta.get("iteration"))
    if isinstance(si, (int, float)) and si > 0:
        return int(si)
    best = -1
    try:
        for f in os.listdir(CHECKPOINT_DIR):
            m = re.match(r"^ckpt-iter(\d+)-", f)
            if m:
                best = max(best, int(m.group(1)))
    except Exception:
        pass
    return best if best >= 0 else None


def _run_records(history, started_at, stopped_at):
    """Records belonging to the controller's active (or last) run window."""
    if started_at is None:
        return []
    out = []
    for r in history:
        t = r.get("t")
        if t is None:
            continue
        if t >= started_at - 1 and (stopped_at is None or t <= stopped_at + 1):
            out.append(r)
    return out


def _status():
    history = _read_history()
    meta = _checkpoint_meta()
    with CTRL.lock:
        running = CTRL.is_running()
        started_at = CTRL.started_at
        stopped_at = CTRL.stopped_at

        recs = _run_records(history, started_at, stopped_at)
        last = recs[-1] if recs else None
        if last is None and history:
            last = history[-1]  # server restarted mid-run: fall back to latest
        run_id = ((last or {}).get("run_id")
                  or meta.get("run_id") or CTRL.run_id)

        stale = False
        if running:
            last_t = last.get("t") if last else None
            if last_t is None:
                stale = (time.time() - (started_at or time.time())) > STALE_S
            else:
                stale = (time.time() - last_t) > STALE_S

        elapsed = int(time.time() - started_at) if started_at else None

        arena = None
        for r in reversed(recs):
            if _is_arena_event(r, None):
                arena = {
                    "score": r.get("arena_score"),
                    "win_rate": r.get("arena_win_rate", r.get("arena_score")),
                    "w": r.get("arena_w"),
                    "l": r.get("arena_l"),
                    "d": r.get("arena_d"),
                    "accepted": r.get("accepted"),
                }
                break

        def pick(*keys):
            for k in keys:
                v = last.get(k) if last else None
                if v is not None:
                    return v
            return None

        return {
            "running": running,
            "pid": CTRL.proc.pid if running else None,
            "started_at": started_at,
            "stopped_at": stopped_at,
            "elapsed_s": elapsed,
            "workers": CTRL.workers,
            "resume": CTRL.resume,
            "backend": getattr(CTRL, "backend", "python"),
            "run_id": run_id,
            "exit_code": CTRL.returncode,
            "error": CTRL.error,
            "stale": stale,
            "generation": pick("generation"),
            "iteration": pick("iteration", "live_iteration"),
            "live_iteration": pick("iteration", "live_iteration"),
            "saved_iteration": _saved_iteration(recs, meta),
            "loss": pick("loss", "total_loss"),
            "total_loss": pick("total_loss", "loss"),
            "policy_loss": pick("policy_loss"),
            "value_loss": pick("value_loss"),
            "entropy": pick("entropy"),
            "optimizer_steps": pick("optimizer_steps"),
            "replay_size": pick("replay_size"),
            "games": pick("games", "replay_size"),
            "arena_score": pick("arena_score"),
            "arena_win_rate": pick("arena_win_rate"),
            "arena": arena,
            "accepted": pick("accepted"),
            "last_t": last.get("t") if last else None,
            "history_len": len(history),
            "runs_count": len(_history()["runs"]),
            "checkpoint_count": _checkpoint_count(),
            "checkpoints": _checkpoint_info(),
            "checkpoint_meta": meta,
            "resources": _read_resources(),
            "log_tail": _log_tail(),
            "server_time": time.time(),
        }


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


# --------------------------------------------------------------------------- #
#  HTTP server (same-origin; no wildcard CORS)                                #
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html", "/training.html"):
            return self._html()
        if self.path == "/api/status":
            return self._json(_status())
        if self.path == "/api/history":
            return self._json(_history())
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/control":
            return self._json({"error": "not found"}, 404)
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return self._json({"error": "Content-Type must be application/json"}, 415)
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return self._json({"error": "invalid Content-Length"}, 400)
        if length < 0 or length > 65536:
            return self._json({"error": "request body too large"}, 413)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._json({"error": "invalid JSON body"}, 400)
        if not isinstance(data, dict):
            return self._json({"error": "JSON body must be an object"}, 400)
        action = data.get("action")
        if action == "start":
            backend = data.get("backend", "python")
            if backend not in ("python", "native"):
                return self._json({"error": "backend must be 'python' or 'native'"}, 400)
            try:
                workers = int(data.get("workers", 8))
            except (TypeError, ValueError):
                return self._json({"error": "workers must be an integer"}, 400)
            if not 1 <= workers <= 32:
                return self._json({"error": "workers must be in [1, 32]"}, 400)
            resume = data.get("resume", True)
            if not isinstance(resume, bool):
                return self._json({"error": "resume must be a boolean"}, 400)

            def _opt_int(key):
                v = data.get(key)
                if v is None:
                    return None
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None

            warm_start_checkpoint = data.get("warm_start_checkpoint")
            checkpoint_dir = data.get("checkpoint_dir")
            if warm_start_checkpoint is not None and not isinstance(warm_start_checkpoint, str):
                return self._json({"error": "warm_start_checkpoint must be a string"}, 400)
            if checkpoint_dir is not None and not isinstance(checkpoint_dir, str):
                return self._json({"error": "checkpoint_dir must be a string"}, 400)

            result = CTRL.start(
                workers=workers, resume=resume, backend=backend,
                num_simulations=_opt_int("num_simulations"),
                games_per_iteration=_opt_int("games_per_iteration"),
                num_iterations=_opt_int("num_iterations"),
                arena_every=_opt_int("arena_every"),
                warm_start_checkpoint=warm_start_checkpoint,
                checkpoint_dir=checkpoint_dir,
            )
        elif action == "stop":
            result = CTRL.stop()
        else:
            result = {"ok": False, "error": "unknown action %r" % action}
        return self._json(result)

    def log_message(self, *a):
        pass  # keep the console clean


def main():
    host, port = _bind_config()
    srv = ThreadingHTTPServer((host, port), Handler)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    print(f"[train_server] real training controller on http://{shown}:{port}/", flush=True)
    if host in ("0.0.0.0", "::", ""):
        print("[train_server] WARNING: unauthenticated control API exposed on ALL "
              "interfaces (CHESS_TRAIN_HOST). Prefer 127.0.0.1.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
