#!/usr/bin/env python3
"""Start training and poll /api/status until it stops (or timeout), capturing the timeline."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8792"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())


def main():
    print("starting...", flush=True)
    print(post("/api/control", {"action": "start", "workers": 8, "resume": True}), flush=True)

    seen_pid = None
    t0 = time.time()
    last_running = True
    while time.time() - t0 < 360:
        s = get("/api/status")
        line = (f"t={int(time.time()-t0):4d}s running={s['running']} pid={s['pid']} "
                f"iter={s['iteration']} hist={s['history_len']} "
                f"cpu={s['resources']['cpu']} gpu={s['resources']['gpu']} "
                f"ram={s['resources']['ram']}")
        print(line, flush=True)
        if seen_pid is None and s["pid"]:
            seen_pid = s["pid"]
        if last_running and not s["running"]:
            print(">>> TRAINING STOPPED at t=%ds" % int(time.time() - t0), flush=True)
            print(">>> last log lines:", flush=True)
            for ln in s["log_tail"][-12:]:
                print("    | " + ln, flush=True)
            break
        last_running = s["running"]
        time.sleep(8)


if __name__ == "__main__":
    main()
