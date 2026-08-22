#!/usr/bin/env python3
"""Visual self-play server + integration-ready local model API.

Both sides of the self-play game are played by the same neural network
(ResNet policy/value head) guided by Monte-Carlo Tree Search — the AlphaZero
self-play loop, made visible.  The self-play viewer keeps working unchanged.

API (for Light Chess and other clients):

    POST /move   {"fen": "<FEN>", "sims": 100}
        ->  {"move": "e2e4", "value": 0.23, "top_moves": [...],
             "sims": 100, "time_ms": 42,
             "model": {"source": "best.pt", "generation": 56}}
        400 {"error": "..."} on invalid FEN / game-over position / bad sims.

    POST /control  {"action": "reload"}  — hot-reload best.pt right now.
    POST /control  {"action": "new" | "pause" | "resume", "sims": N}

best.pt is also re-checked automatically at every self-play game boundary and
swapped in if it changed (see reload_if_newer).  The new net only ever takes
over between games or on explicit reload, never mid-search.

Run:  .venv/bin/python serve.py
      (binds 127.0.0.1:8790 by default; CHESS_SERVE_HOST / CHESS_SERVE_PORT
      override host and port)
"""

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import numpy as np
import torch

import evaluate  # noqa: F401  (seeded_context + encoding.encode_batch bridge)
from config import Config
from model import ChessNet
from mcts import MCTS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PORT = 8790
DEFAULT_HOST = "127.0.0.1"
VIEWER = os.path.join(HERE, "viewer.html")


# --------------------------------------------------------------------------- #
#  Model loading                                                              #
# --------------------------------------------------------------------------- #

def bind_host_port():
    """Resolve (host, port) for the HTTP server.

    Defaults to localhost-only; override via the CHESS_SERVE_HOST and
    CHESS_SERVE_PORT environment variables.
    """
    host = os.environ.get("CHESS_SERVE_HOST", DEFAULT_HOST)
    port = DEFAULT_PORT
    raw = os.environ.get("CHESS_SERVE_PORT")
    if raw:
        try:
            port = int(raw)
        except ValueError:
            pass
    return host, port


def infer_generation(cfg, source):
    """Best-effort training-generation number for a loaded checkpoint.

    best.pt is a raw state dict with no iteration metadata, so we infer the
    generation from the newest ckpt-iterNNNN-*.pt snapshot file inside
    *cfg.checkpoint_dir* (strictly cfg-local: the repo's training.jsonl is
    deliberately NOT consulted, so tests and multi-config setups stay
    isolated).  latest.pt carries an explicit 'iteration' key.  Returns None
    when nothing is known.
    """
    try:
        if source and source.endswith("latest.pt"):
            meta_path = os.path.join(cfg.checkpoint_dir, "checkpoint_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                value = meta.get("generation", meta.get("iteration"))
                return None if value is None else int(value)
        if source and source.endswith("best.pt"):
            best_iter = None
            if os.path.isdir(cfg.checkpoint_dir):
                for fn in os.listdir(cfg.checkpoint_dir):
                    if fn.startswith("ckpt-iter") and fn.endswith(".pt"):
                        try:
                            n = int(fn[len("ckpt-iter"): fn.index("-", len("ckpt-iter"))])
                            best_iter = n if best_iter is None else max(best_iter, n)
                        except ValueError:
                            continue
            if best_iter is not None:
                return best_iter
    except Exception:  # noqa: BLE001  (generation info is best-effort)
        pass
    return None


def load_net(cfg):
    """Load the best trained weights if present, else a fresh random net.

    Falls back gracefully when a checkpoint's shapes no longer match the
    current architecture (e.g. mid-rewrite of the encoder/policy head), so the
    server keeps serving with whatever is loadable.
    """
    net = ChessNet(cfg).to(cfg.device)
    net.eval()
    # Serve only the arena-accepted weights-only artifact. The full resumable
    # latest.pt contains Python/NumPy RNG and replay objects and is deliberately
    # never unpickled by the HTTP service.
    candidates = [("best.pt", os.path.join(cfg.checkpoint_dir, "best.pt"))]
    for source, path in candidates:
        if not os.path.exists(path):
            continue
        try:
            state = torch.load(path, map_location=cfg.device, weights_only=True)
            net.load_state_dict(state)
            return net, source
        except Exception as e:  # noqa: BLE001
            print(f"[serve] {source} unloadable ({e}); trying next", flush=True)
    return net, "untrained (random init)"


# --------------------------------------------------------------------------- #
#  Move API + hot reload                                                      #
# --------------------------------------------------------------------------- #

_MAX_SIMS = 2000


def compute_move(net, cfg, fen, sims=None, seed=None, model_info=None):
    """One-shot MCTS move for the integration API (POST /move).

    Args:
        net: ChessNet (eval mode).
        cfg: Config.
        fen: FEN of the position; the side to move must have legal moves.
        sims: MCTS simulations (default cfg.num_simulations, clamped to
              [1, _MAX_SIMS]).
        seed: optional; wraps the search in a seeded context so the same
              (fen, sims, seed) reproduces the same move.
        model_info: dict merged into the "model" response field.

    Returns a dict {move, value, top_moves, sims, time_ms, model}.
    Raises ValueError for unparsable FENs, game-over positions or bad sims.
    """
    if sims is None:
        sims = cfg.num_simulations
    net = net.to(cfg.device)  # MCTS moves inputs to cfg.device; net must follow
    if not isinstance(sims, int) or isinstance(sims, bool) or not (1 <= sims <= _MAX_SIMS):
        raise ValueError(f"sims must be an int in [1, {_MAX_SIMS}], got {sims!r}")
    # Strict 6-field FEN contract (the Light Chess client always sends six):
    # python-chess would silently fill defaults for 4- and 5-field strings,
    # which hides malformed input from API callers.
    fields = fen.split() if isinstance(fen, str) else None
    if fields is None or len(fields) != 6:
        raise ValueError(
            "invalid FEN: expected exactly 6 space-separated fields "
            f"(board turn castling ep halfmove fullmove), got {len(fields) if fields is not None else 'non-string'}"
        )
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"invalid FEN: {e}") from e
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed must be an integer or null")
    if board.is_game_over(claim_draw=True):
        raise ValueError("game is over or draw is claimable: no move should be played")

    mcts = MCTS(net, cfg)
    t0 = time.time()
    with evaluate.seeded_context(seed):
        pi = mcts.search(board, temperature=0.0, num_sims=sims)
    t_ms = int((time.time() - t0) * 1000)
    if not pi:
        raise ValueError("search produced no move")
    move = max(pi, key=pi.get)

    root = mcts.root
    value = 0.0
    if root is not None and root.N > 0:
        value = float(root.W / root.N)  # side-to-move perspective

    ranked = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
    top_moves = [
        {
            "uci": m.uci(),
            "san": board.san(m),
            "visits": int(n.N),
            "prior": round(float(n.P), 4),
        }
        for m, n in ranked[:5]
    ]

    model = {
        "source": (model_info or {}).get("source", "untrained (random init)"),
        "generation": (model_info or {}).get("generation"),
        "device": str(cfg.device),
    }
    return {
        "move": move.uci(),
        "value": round(value, 4),
        "top_moves": top_moves,
        "sims": sims,
        "time_ms": t_ms,
        "model": model,
    }


def reload_if_newer(game, force=False):
    """Hot-reload best.pt into `game` if the file changed since last load.

    Safe to call only at a game boundary (self-play loop) or on explicit user
    request: the swap replaces game.net and game.mcts under the game lock and
    bumps the epoch so any in-flight search for a stale board is discarded.

    Returns True when a new net was loaded.
    """
    best = os.path.join(game.cfg.checkpoint_dir, "best.pt")
    if not os.path.exists(best):
        return False
    try:
        sig = (os.path.getmtime(best), os.path.getsize(best))
    except OSError:
        return False
    with game.lock:
        if not force and sig == getattr(game, "_ckpt_signature", None):
            return False
        try:
            state = torch.load(best, map_location=game.cfg.device, weights_only=True)
            net = ChessNet(game.cfg).to(game.cfg.device)
            net.load_state_dict(state)
            net.eval()
        except Exception as e:  # noqa: BLE001  (shape mismatch, corrupt file)
            print(f"[serve] reload skipped: best.pt unloadable ({e})", flush=True)
            return False
        game.net = net
        game.mcts = MCTS(net, game.cfg)
        game.trained_from = "best.pt"
        game.generation = infer_generation(game.cfg, "best.pt")
        game._ckpt_signature = sig
        game.epoch += 1  # invalidate in-flight searches for the old board
        print(f"[serve] hot-reloaded best.pt (gen {game.generation})", flush=True)
        return True


# --------------------------------------------------------------------------- #
#  Aggregate self-play stats (persisted across restarts)                       #
# --------------------------------------------------------------------------- #

_RES_CACHE = None


def _read_resources():
    """Live CPU %, RAM %, GPU % + temp from /proc and nvidia-smi (cached ~2s)."""
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


def _read_training():
    """Latest train.py metrics + checkpoint status (for the dashboard)."""
    path = os.path.join(HERE, "training.jsonl")
    last = None
    mtime = 0
    try:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
    except Exception:
        pass
    ckpt_dir = os.path.join(HERE, "checkpoints")
    best_pt = os.path.exists(os.path.join(ckpt_dir, "best.pt"))
    latest_pt = os.path.exists(os.path.join(ckpt_dir, "latest.pt"))
    running = False
    try:
        running = subprocess.run(["pgrep", "-f", "train.py"],
                                 capture_output=True, timeout=3).returncode == 0
    except Exception:
        pass
    return {"last": last, "best_pt": best_pt, "latest_pt": latest_pt, "active": running}


class Stats:
    """Counts completed self-play games and their results; persists to stats.json."""

    def __init__(self, path):
        self.path = os.path.join(path, "stats.json")
        self.data = {
            "started_at": time.time(),
            "games_completed": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "total_plies": 0,
        }
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path) as f:
                    d = json.load(f)
                for k in ("games_completed", "white_wins", "black_wins",
                          "draws", "total_plies"):
                    if k in d:
                        self.data[k] = d[k]
        except Exception:
            pass

    def record(self, result, plies):
        self.data["games_completed"] += 1
        self.data["total_plies"] += plies
        if result == "white":
            self.data["white_wins"] += 1
        elif result == "black":
            self.data["black_wins"] += 1
        else:
            self.data["draws"] += 1
        self._save()

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f)
        except Exception:
            pass

    def snapshot(self):
        d = dict(self.data)
        elapsed = max(1.0, time.time() - d["started_at"])
        d["uptime_seconds"] = int(elapsed)
        d["games_per_hour"] = round(d["games_completed"] / elapsed * 3600.0, 2)
        d["avg_plies"] = (
            round(d["total_plies"] / d["games_completed"], 1)
            if d["games_completed"] else 0
        )
        return d


# --------------------------------------------------------------------------- #
#  Shared game state                                                          #
# --------------------------------------------------------------------------- #

class Game:
    def __init__(self, cfg, net, trained_from):
        self.cfg = cfg
        self.net = net
        self.mcts = MCTS(net, cfg)
        self.trained_from = trained_from
        self.generation = infer_generation(cfg, trained_from)
        # Record the (mtime, size) of the best.pt we started from so the FIRST
        # reload_if_newer() call returns False while the file is unchanged —
        # otherwise every server start would hot-reload the same file once.
        self._ckpt_signature = None   # (mtime, size) of the loaded best.pt
        if trained_from == "best.pt":
            best = os.path.join(cfg.checkpoint_dir, "best.pt")
            try:
                self._ckpt_signature = (os.path.getmtime(best), os.path.getsize(best))
            except OSError:
                self._ckpt_signature = None
        self.lock = threading.Lock()
        self.stats = Stats(HERE)
        self.sims = cfg.num_simulations  # user-adjustable via /control; persists across games
        self.epoch = 0                   # bumped on reset/reload; lets the loop discard a stale in-flight search
        self.reset()

    def reset(self):
        self.board = chess.Board()
        self.moves = []          # list of dicts (see _record)
        self.status = "playing"
        self.result_text = None
        self.thinking = False
        self.last_move = None
        self.eval_white = 0.0
        self.top_moves = []
        self.clock_ms = 0
        # NOTE: self.sims intentionally NOT reset here. It is the user's live
        # search-strength knob and should survive game restarts. Set once in
        # __init__ above.
        self.epoch += 1          # invalidate any MCTS search still in flight
        self.paused = False

    # -- helpers ------------------------------------------------------------

    def _net_value_white(self):
        """Net value estimate, white perspective, in [-1, 1]."""
        x = torch.from_numpy(
            __import__("encoding").encode_board(self.board)
        ).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            _, value = self.net(x)
        v = float(value[0, 0])  # side-to-move perspective
        return v if self.board.turn == chess.WHITE else -v

    def _terminal(self):
        b = self.board
        if b.is_checkmate():
            return "checkmate", "Black wins" if b.turn == chess.WHITE else "White wins"
        if b.is_stalemate():
            return "stalemate", "Draw — stalemate"
        if b.is_insufficient_material():
            return "material", "Draw — insufficient material"
        if b.can_claim_threefold_repetition():
            return "repetition", "Draw — threefold repetition claim"
        if b.can_claim_fifty_moves():
            return "fifty", "Draw — fifty-move claim"
        if len(b.move_stack) >= self.cfg.max_game_length:
            return "length", "Draw — move limit"
        return None, None

    def _record(self, move, san, t_ms, visits_top):
        self.moves.append({
            "uci": move.uci(),
            "san": san,
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "time_ms": t_ms,
            "top": visits_top,
        })
        self.last_move = {"from": chess.square_name(move.from_square),
                          "to": chess.square_name(move.to_square), "san": san}

    # -- the self-play loop (runs in its own thread) ------------------------

    def loop(self):
        import encoding  # noqa: F401  (local import keeps encoding import lazy)
        while True:
            with self.lock:
                if self.paused:
                    time.sleep(0.1)
                    continue

                reason, text = self._terminal()
                if reason:
                    # finalise
                    if reason == "checkmate":
                        self.status = "white_won" if text == "White wins" else "black_won"
                    else:
                        self.status = "draw"
                    self.result_text = text
                    self.thinking = False
                else:
                    # -- compute one move with MCTS --
                    self.thinking = True
                    ply = len(self.board.move_stack)
                    temp = self.cfg.temperature if ply < self.cfg.temperature_threshold else 0.0
                    # snapshot board for the search so we don't mutate mid-search
                    board = self.board.copy()
                    epoch = self.epoch

            # (search happens OUTSIDE the lock so /state stays responsive)
            if not (reason or self.paused):
                t0 = time.time()
                pi = self.mcts.search(board, temperature=temp, num_sims=self.sims)
                t_ms = int((time.time() - t0) * 1000)
                if not pi:
                    # Search can return no moves for a claimable draw. Keep the
                    # viewer thread alive even if terminal detection changes.
                    with self.lock:
                        self.thinking = False
                        self.status = "draw"
                        self.result_text = "Draw — claimable terminal position"
                    continue
                move = max(pi, key=pi.get)

                # top moves by visit count for the insight panel
                children = self.mcts.root.children if self.mcts.root else {}
                ranked = sorted(children.items(), key=lambda kv: kv[1].N, reverse=True)
                visits_top = [
                    {"san": board.san(m), "visits": int(n.N), "prior": round(float(n.P), 3)}
                    for m, n in ranked[:3]
                ]

                san = board.san(move)

                with self.lock:
                    if epoch != self.epoch:
                        # board was reset mid-search (New game / game finished):
                        # the move is for a stale position — drop it, don't crash.
                        self.thinking = False
                        continue
                    self._record(move, san, t_ms, visits_top)
                    self.board.push(move)
                    self.eval_white = self._net_value_white()
                    self.top_moves = visits_top
                    self.clock_ms = t_ms
                    self.thinking = False

            # after a finished game: hot-reload best.pt (if changed) at this
            # safe game boundary, then pause a beat and auto-restart
            with self.lock:
                finished = self.status != "playing"
            if finished:
                reload_if_newer(self)   # safe: no search in flight
                time.sleep(4.0)
                with self.lock:
                    plies = len(self.board.move_stack)
                    result = ("white" if self.status == "white_won"
                              else "black" if self.status == "black_won" else "draw")
                    self.stats.record(result, plies)
                    self.reset()
                continue

            time.sleep(0.35)  # give the viewer time to animate each move


# --------------------------------------------------------------------------- #
#  HTTP server                                                                #
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    game = None  # set by main()

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
        with open(VIEWER, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Answer CORS preflight. The dashboard is served from :8791 but talks
        to this server on :8790, so the browser preflights cross-origin POSTs
        (pause/resume/new/sims). Without this, those POSTs are silently
        blocked, which is exactly why the Pause button and the sims field
        appeared broken. Read the body to keep the connection clean."""
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
        if self.path in ("/", "/index.html"):
            return self._html()
        if self.path == "/state":
            with self.game.lock:
                check_sq = None
                if self.game.board.is_check():
                    k = self.game.board.king(self.game.board.turn)
                    if k is not None:
                        check_sq = chess.square_name(k)
                state = {
                    "fen": self.game.board.fen(),
                    "check_square": check_sq,
                    "turn": "w" if self.game.board.turn == chess.WHITE else "b",
                    "status": self.game.status,
                    "result_text": self.game.result_text,
                    "move_number": self.game.board.fullmove_number,
                    "ply": len(self.game.board.move_stack),
                    "thinking": self.game.thinking,
                    "sims": self.game.sims,
                    "trained_from": self.game.trained_from,
                    "generation": self.game.generation,
                    "eval_white": round(self.game.eval_white, 4),
                    "top_moves": self.game.top_moves,
                    "last_move": self.game.last_move,
                    "clock_ms": self.game.clock_ms,
                    "moves": [m["san"] for m in self.game.moves],
                }
            return self._json(state)
        if self.path == "/stats":
            with self.game.lock:
                s = self.game.stats.snapshot()
                s["sims"] = self.game.sims
                s["paused"] = self.game.paused
                s["trained_from"] = self.game.trained_from
                s["device"] = self.game.cfg.device
                s["resources"] = _read_resources()
                s["training"] = _read_training()
                s["current"] = {
                    "fen": self.game.board.fen(),
                    "status": self.game.status,
                    "result_text": self.game.result_text,
                    "move_number": self.game.board.fullmove_number,
                    "ply": len(self.game.board.move_stack),
                    "eval_white": round(self.game.eval_white, 4),
                    "thinking": self.game.thinking,
                    "last_move": self.game.last_move,
                    "moves": [m["san"] for m in self.game.moves],
                }
            return self._json(s)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path not in ("/control", "/move"):
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

        if self.path == "/move":
            try:
                with self.game.lock:
                    net = self.game.net
                    cfg = self.game.cfg
                    model_info = {
                        "source": self.game.trained_from,
                        "generation": self.game.generation,
                    }
                result = compute_move(
                    net,
                    cfg,
                    data.get("fen"),
                    sims=data.get("sims"),
                    seed=data.get("seed"),
                    model_info=model_info,
                )
                return self._json(result)
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            except Exception as exc:
                return self._json({"error": f"move computation failed: {exc}"}, 500)

        action = data.get("action")
        if action == "reload":
            # reload_if_newer takes the game lock itself; must NOT be called
            # while we already hold it (threading.Lock is not reentrant).
            reloaded = reload_if_newer(self.game, force=True)
            with self.game.lock:
                return self._json({"ok": True, "reloaded": reloaded,
                                   "trained_from": self.game.trained_from,
                                   "generation": self.game.generation,
                                   "paused": self.game.paused,
                                   "sims": self.game.sims})
        with self.game.lock:
            if action == "new":
                self.game.reset()
            elif action == "pause":
                self.game.paused = True
            elif action == "resume":
                self.game.paused = False
            sims = data.get("sims")
            if isinstance(sims, int) and not isinstance(sims, bool) and 1 <= sims <= 2000:
                self.game.sims = sims
            return self._json({"ok": True, "paused": self.game.paused, "sims": self.game.sims})

    def log_message(self, *a):
        pass  # keep the console clean


def main():
    cfg = Config()
    net, trained_from = load_net(cfg)
    print(f"[serve] device={cfg.device}  weights={trained_from}", flush=True)

    game = Game(cfg, net, trained_from)
    Handler.game = game

    t = threading.Thread(target=game.loop, daemon=True)
    t.start()

    host, port = bind_host_port()
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[serve] self-play board live at  http://{host}:{port}/", flush=True)
    if host in ("0.0.0.0", "::", ""):
        print(f"[serve] WARNING: model API exposed beyond localhost on :{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
