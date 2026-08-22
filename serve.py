#!/usr/bin/env python3
"""Visual self-play server: the ChessNet plays itself, streamed to a browser tab.

Both sides are played by the same neural network (ResNet policy/value head)
guided by Monte-Carlo Tree Search -- the AlphaZero self-play loop, made visible.

Run:  .venv/bin/python serve.py   (serves on 0.0.0.0:8790)
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

from config import Config
from model import ChessNet
from mcts import MCTS

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8790
VIEWER = os.path.join(HERE, "viewer.html")


# --------------------------------------------------------------------------- #
#  Model loading                                                              #
# --------------------------------------------------------------------------- #

def load_net(cfg):
    """Load the best trained weights if present, else a fresh random net."""
    net = ChessNet(cfg).to(cfg.device)
    net.eval()
    best = os.path.join(cfg.checkpoint_dir, "best.pt")
    if os.path.exists(best):
        net.load_state_dict(torch.load(best, map_location=cfg.device))
        return net, "best.pt"
    latest = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if os.path.exists(latest):
        state = torch.load(latest, map_location=cfg.device, weights_only=False)
        net.load_state_dict(state.get("best_net", state.get("net")))
        return net, "latest.pt"
    return net, "untrained (random init)"


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
        self.lock = threading.Lock()
        self.stats = Stats(HERE)
        self.sims = cfg.num_simulations  # user-adjustable via /control; persists across games
        self.epoch = 0                   # bumped on reset; lets the loop discard a stale in-flight search
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
        if b.is_repetition(3):
            return "repetition", "Draw — threefold repetition"
        if b.is_fifty_moves():
            return "fifty", "Draw — fifty-move rule"
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

            # after a finished game, pause a beat then auto-restart
            with self.lock:
                finished = self.status != "playing"
            if finished:
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
        if self.path != "/control":
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            data = {}
        with self.game.lock:
            action = data.get("action")
            if action == "new":
                self.game.reset()
            elif action == "pause":
                self.game.paused = True
            elif action == "resume":
                self.game.paused = False
            if isinstance(data.get("sims"), int) and 1 <= data["sims"] <= 2000:
                self.game.sims = data["sims"]
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

    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[serve] self-play board live at  http://127.0.0.1:{PORT}/", flush=True)
    print(f"[serve] (also reachable on the Tailscale IP :{PORT})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
