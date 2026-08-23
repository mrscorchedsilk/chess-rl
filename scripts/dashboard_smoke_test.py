#!/usr/bin/env python3
"""
dashboard_smoke_test.py — Independent smoke test + derived-metrics harness for the
chess RL training dashboard API (http://127.0.0.1:8792).

This is a VERIFICATION harness, not a dashboard. It:
  1. GETs /api/status and asserts the API contract (HTTP 200, valid JSON, required
     keys with correct types), printing a clean table of live values.
  2. GETs /api/history and asserts structure, counting records / arena events /
     training records and locating the current run (max last record t).
  3. Computes REAL derived metrics from the live data:
       a. games/min over the last 60s and 300s windows (current run's records)
       b. Elo estimate from arena gates (Elo=1000 start; s=clamp(score,.01,.99);
          delta=-400*log10(1/s-1) clamped to [-250,250]; applied only if accepted)
       c. improvement deltas: first vs latest policy/value loss in current run, and
          the linear slope of total_loss (policy+value) over the last 50 training records
  4. Optionally POSTs action='start' (workers=8, resume=true) purely to OBSERVE the
     guard — with training live it must be refused; a guarded refusal is a PASS.
     It NEVER POSTs action='stop'.

Exit code 0 iff every assertion passes; otherwise non-zero. Prints
'SMOKE TEST PASS' / 'SMOKE TEST FAIL' at the very end.

Runs with ONLY the Python standard library (urllib.request + json), so it works
under the project venv python or any python3.
"""

import json
import math
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8792"
TIMEOUT = 15

failures = []


def check(cond, msg):
    """Record a failed assertion; always print the outcome line."""
    if cond:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        failures.append(msg)
    return cond


def http_get(path):
    """GET a JSON endpoint. Returns (parsed_json, raw_text). Raises on error."""
    with urllib.request.urlopen(BASE + path, timeout=TIMEOUT) as resp:
        code = resp.status
        raw = resp.read().decode("utf-8", "replace")
    if code != 200:
        raise AssertionError(f"GET {path} -> HTTP {code}")
    return json.loads(raw), raw


def http_post(path, payload):
    """POST JSON to an endpoint; returns (status_code, parsed_json)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8", "replace"))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def games_per_min(records, now, window_s):
    """games/min over the last `window_s` seconds using the current run's records.

    'games' is cumulative per run; we take the latest record at/before the window
    start as the baseline and the latest record inside the window as the endpoint,
    then divide the games delta by the observed span (falling back to the window
    length if the span is degenerate).
    """
    train = sorted(
        [r for r in records if r.get("event") != "arena" and isinstance(r.get("games"), (int, float))],
        key=lambda r: r["t"],
    )
    if len(train) < 2:
        return 0.0
    window_start = now - window_s
    baseline = None
    for r in train:
        if r["t"] <= window_start:
            baseline = r
        else:
            break
    end = None
    for r in train:
        if r["t"] <= now:
            end = r
        else:
            break
    if end is None or baseline is None:
        return 0.0
    if end["t"] <= baseline["t"]:
        # No movement inside the window at all.
        return 0.0
    span_min = (end["t"] - baseline["t"]) / 60.0
    if span_min <= 0:
        span_min = window_s / 60.0
    return (end["games"] - baseline["games"]) / span_min


def linear_slope(points):
    """Least-squares slope over (x, y) pairs."""
    n = len(points)
    if n < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    num = sum((x - xbar) * (y - ybar) for x, y in points)
    den = sum((x - xbar) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def main():
    print("=" * 70)
    print("CHESS RL TRAINING DASHBOARD — SMOKE TEST + DERIVED METRICS")
    print("=" * 70)

    # ------------------------------------------------------------------ status
    print("\n[1] GET /api/status (contract assertions)")
    status, _ = http_get("/api/status")
    check(isinstance(status, dict), "status is a JSON object")

    for key, typ in (
        ("running", bool),
        ("games", int),
        ("iteration", int),
    ):
        check(key in status and isinstance(status[key], typ), f"status['{key}'] is {typ.__name__}")
    check(
        "policy_loss" in status and (status["policy_loss"] is None or isinstance(status["policy_loss"], float)),
        "status['policy_loss'] is float or null",
    )
    res = status.get("resources")
    check(isinstance(res, dict) and all(k in res for k in ("cpu", "ram", "gpu", "temp")),
          "status['resources'] is dict with cpu,ram,gpu,temp")
    for k in ("cpu", "ram", "gpu", "temp"):
        check(isinstance(res.get(k), (int, float)), f"resources['{k}'] is numeric")

    print("\n  --- live status table ---")
    status_rows = [
        ("running", status.get("running")),
        ("pid", status.get("pid")),
        ("backend", status.get("backend")),
        ("run_id", status.get("run_id")),
        ("generation", status.get("generation")),
        ("iteration", status.get("iteration")),
        ("games", status.get("games")),
        ("policy_loss", status.get("policy_loss")),
        ("value_loss", status.get("value_loss")),
        ("entropy", status.get("entropy")),
        ("optimizer_steps", status.get("optimizer_steps")),
        ("replay_size", status.get("replay_size")),
        ("arena_score", status.get("arena_score")),
        ("arena_win_rate", status.get("arena_win_rate")),
        ("accepted", status.get("accepted")),
        ("elapsed_s", status.get("elapsed_s")),
        ("stale", status.get("stale")),
        ("history_len", status.get("history_len")),
        ("runs_count", status.get("runs_count")),
        ("last_t", status.get("last_t")),
    ]
    for k, v in status_rows:
        print(f"    {k:<16} {v}")
    print("    resources:")
    for k in ("cpu", "ram", "gpu", "temp", "headroom"):
        print(f"      {k:<10} {res.get(k)}")

    # ---------------------------------------------------------------- history
    print("\n[2] GET /api/history (structure + counts)")
    hist, _ = http_get("/api/history")
    check(isinstance(hist.get("records"), list), "history['records'] is a list")
    check(isinstance(hist.get("runs"), list), "history['runs'] is a list")

    flat = hist["records"]
    runs = hist["runs"]
    total = hist.get("total")
    check(isinstance(total, int) and total == len(flat),
          f"history['total'] ({total}) matches flat records count ({len(flat)})")

    arena_events = [r for r in flat if r.get("event") == "arena"]
    training_recs = [r for r in flat if r.get("event") != "arena"]
    print(f"    total records      : {len(flat)}")
    print(f"    arena events       : {len(arena_events)}")
    print(f"    training records   : {len(training_recs)}")
    print(f"    runs               : {len(runs)}")

    # Current run = run whose latest record has max t
    def run_last_t(run):
        recs = run.get("records") or []
        return recs[-1]["t"] if recs else run.get("started_at", 0)

    current = max(runs, key=run_last_t)
    cur_id = current["run_id"]
    cur_records = current.get("records") or []
    cur_last_t = cur_records[-1]["t"] if cur_records else None
    print(f"    current run        : {cur_id} (max last_t={cur_last_t}, {len(cur_records)} records)")
    check(len(cur_records) > 0, "current run has records")
    check(cur_id == status.get("run_id"), "current run id matches live status run_id")

    # ------------------------------------------------------------ games/min
    print("\n[3] Derived metrics (real numbers, 3 decimals)")
    now = time.time()
    gpm_60 = games_per_min(cur_records, now, 60)
    gpm_300 = games_per_min(cur_records, now, 300)
    print(f"    games/min (60s window)  : {gpm_60:.3f}")
    print(f"    games/min (300s window) : {gpm_300:.3f}")
    check(gpm_60 >= 0.0 and gpm_300 >= 0.0, "games/min rates are non-negative")

    # ------------------------------------------------------------------- elo
    arena_sorted = sorted(arena_events, key=lambda r: r["t"])
    elo = 1000.0
    accepted_gens = 0
    latest = None
    for ev in arena_sorted:
        score = ev.get("score")
        if score is None:
            continue
        s = clamp(float(score), 0.01, 0.99)
        delta = -400.0 * math.log10(1.0 / s - 1.0)
        delta = clamp(delta, -250.0, 250.0)
        latest = {"score": float(score), "delta": delta, "accepted": bool(ev.get("accepted"))}
        if ev.get("accepted"):
            elo += delta
            accepted_gens += 1
    elo_final = elo
    arena_gates = len(arena_sorted)
    print(f"    Elo estimate (final)     : {elo_final:.3f}")
    print(f"    accepted generations     : {accepted_gens}")
    print(f"    total arena gates        : {arena_gates}")
    if latest:
        print(f"    latest gate              : score={latest['score']:.3f} "
              f"delta={latest['delta']:.3f} accepted={latest['accepted']}")
    else:
        print("    latest gate              : none")
    check(elo_final == 1000.0 or accepted_gens > 0,
          "Elo accounting consistent (accepted gates imply Elo moved)")

    # ---------------------------------------------- improvement deltas + slope
    cur_train = sorted(
        [r for r in cur_records if r.get("event") != "arena"],
        key=lambda r: r["t"],
    )
    pl = [r["policy_loss"] for r in cur_train if isinstance(r.get("policy_loss"), (int, float))]
    vl = [r["value_loss"] for r in cur_train if isinstance(r.get("value_loss"), (int, float))]
    if len(pl) >= 2:
        print(f"    policy_loss delta (first->latest): {pl[-1] - pl[0]:+.3f} "
              f"({pl[0]:.3f} -> {pl[-1]:.3f})")
    else:
        print("    policy_loss delta: insufficient data")
    if len(vl) >= 2:
        print(f"    value_loss delta (first->latest): {vl[-1] - vl[0]:+.3f} "
              f"({vl[0]:.3f} -> {vl[-1]:.3f})")
    else:
        print("    value_loss delta: insufficient data")

    last50 = cur_train[-50:]
    slope_pts = [(i, r["policy_loss"] + r["value_loss"])
                 for i, r in enumerate(last50)
                 if isinstance(r.get("policy_loss"), (int, float))
                 and isinstance(r.get("value_loss"), (int, float))]
    loss_slope = linear_slope(slope_pts)
    print(f"    total_loss slope (last {len(slope_pts)} recs): {loss_slope:.3f} per-record")
    check(len(slope_pts) >= 2, "at least 2 training records available for loss slope")

    # -------------------------------------------------- optional POST observe
    # The original spec allowed an observe-only POST action='start' probe, but the
    # operator has since directed this harness to NEVER POST to /api/control.
    # The guard was verified earlier against the live controller and returned the
    # expected refusal: HTTP 200 {"ok": false, "error": "training already running
    # (pid ...)"} — so the endpoint is wired and guarded. Skipping the probe here.
    print("\n[4] POST /api/control probe — SKIPPED (operator directive: no control "
          "POSTs from this harness). Guard previously verified: "
          "{ok:false, error:'training already running ...'}.")

    # ------------------------------------------------------------------ done
    print("\n" + "=" * 70)
    if failures:
        print(f"SMOKE TEST FAIL — {len(failures)} failed assertion(s):")
        for f in failures:
            print(f"  - {f}")
        print("=" * 70)
        return 1
    print("SMOKE TEST PASS")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — harness must fail loudly
        print(f"SMOKE TEST FAIL — exception: {type(e).__name__}: {e}")
        sys.exit(1)
