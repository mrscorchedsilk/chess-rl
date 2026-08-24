"""Arena benchmark: Python-MCTS arena vs native-MCTS arena (docs/tickets.md, Ticket D).

Two modes (exactly one required):

``--compare-python`` (CPU-only; run with ``CUDA_VISIBLE_DEVICES=""``): time the
full deterministic suite (``--games`` games at ``--sims`` simulations from
``--opening-plies``-ply openings generated from ``--seed``) through
``arena.play_match`` (python MCTS, ``benchmarks.native_mcts.FakeNet``) and
through ``native_arena.play_match`` (native MCTS, ``FakeEvaluator``),
``--repeats`` times each.  Reports per-suite wall time, the median, and the
native/python speedup.  Every native run also captures the ordered per-game
move transcripts and verifies their BLAKE2 digest against the Ticket-C golden
(``tests/test_native_arena.py::GOLDEN_TRANSCRIPT_HASH``), proving the
fake-evaluator path is byte-reproducible.

``--gpu`` (real ``InferenceRuntime`` on CUDA): time ``native_arena.play_match``
end-to-end with the production runtimes — a ``native_arena.NativeArenaEngine``
(two persistent ``InferenceRuntime``/``ChessNet`` pairs, candidate/champion
weights copied via ``state_dict``), ``--repeats`` times.  The one-time engine
construction (two ``torch.compile`` calls) is timed and reported separately.
The ``phase="arena"`` telemetry record emitted by ``play_match`` (Ticket A) is
verified to land and parse.

Both modes verify the arena result contract (``{a,b,draws}`` sums to
``num_games``), the score formula ``(wins + 0.5*draws)/games`` with the 0.55
accept threshold, and the phase telemetry record.

Exact benchmark commands (from the ticket)::

    CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmarks/arena_bench.py \\
        --sims 40 --games 20 --opening-plies 8 --seed 424242 --compare-python
    .venv/bin/python benchmarks/arena_bench.py \\
        --gpu --sims 40 --games 20 --opening-plies 8 --seed 424242

Exit codes: 0 = all applicable acceptance thresholds met; 1 = a threshold
failed; 2 = usage / environment error (e.g. ``--gpu`` without CUDA).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import arena  # noqa: E402
import native_arena  # noqa: E402
from benchmarks.native_mcts import FakeEvaluator, FakeNet  # noqa: E402
from config import Config  # noqa: E402

# Ticket-C golden: BLAKE2(digest_size=16) over the ordered per-game transcripts
# (opening + chosen moves) of the full 20-game / 40-sim / seed-424242 /
# 8-opening-plies fake-evaluator suite — recorded in
# tests/test_native_arena.py::GOLDEN_TRANSCRIPT_HASH and pinned there.
GOLDEN_TRANSCRIPT_HASH = "971b2c2eef88b803a30989ff2206abb9"

DEFAULT_TELEMETRY_DIR = REPO_ROOT / "benchmarks" / "results"
ACCEPT_THRESHOLD = 0.55  # cfg.arena_accept_threshold; semantics unchanged


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _arena_cfg(sims, games, opening_plies, seed, telemetry_path, device):
    """A Config with the arena-relevant fields forced to production values."""
    cfg = Config()
    cfg.device = device
    cfg.arena_games = int(games)
    cfg.arena_simulations = int(sims)
    cfg.max_game_length = 400
    cfg.arena_seed = int(seed)
    cfg.arena_opening_plies = int(opening_plies)
    cfg.arena_root_noise = False
    cfg.telemetry_enabled = True
    cfg.telemetry_path = str(telemetry_path)
    return cfg


def _transcripts_hash(transcripts) -> str:
    """BLAKE2(digest_size=16) over the ordered per-game transcripts — the exact
    digest algorithm from tests/test_native_arena.py::_transcripts_hash."""
    h = hashlib.blake2b(digest_size=16)
    for game in transcripts:
        for uci in game:
            h.update(uci.encode("utf-8"))
            h.update(b"\x00")
        h.update(b"\xff")
    return h.hexdigest()


class _TranscriptCapture:
    """Record the ordered per-game move transcripts (opening + chosen moves)
    exactly like the Ticket-C test helper, so the digest is comparable to the
    golden.  Patching is process-local, restored on exit, and adds only two
    python calls per ply — negligible against a 40-sim search."""

    def __init__(self):
        self.transcripts: list = []       # list of (opening, chosen) pairs
        self._current = None
        self._orig_play = None
        self._orig_select = None

    def __enter__(self):
        orig_play = native_arena._play_native_game
        orig_select = native_arena._select_move
        self._orig_play, self._orig_select = orig_play, orig_select

        def select_spy(policy):
            move = orig_select(policy)
            self._current[1].append(move)
            return move

        def play_spy(mcts_white, mcts_black, evaluate_white, evaluate_black,
                     cfg_, num_sims, opening_moves=(), max_batch=256):
            self._current = [list(opening_moves), []]
            self.transcripts.append(self._current)
            return orig_play(mcts_white, mcts_black, evaluate_white,
                             evaluate_black, cfg_, num_sims,
                             opening_moves=opening_moves, max_batch=max_batch)

        native_arena._play_native_game = play_spy
        native_arena._select_move = select_spy
        return self

    def __exit__(self, *exc):
        native_arena._play_native_game = self._orig_play
        native_arena._select_move = self._orig_select
        return None

    def game_transcripts(self):
        """One tuple per game: opening moves + chosen moves, in play order."""
        return [tuple(opening) + tuple(chosen) for opening, chosen in self.transcripts]


def _score_report(result, games):
    """Arena score with the exact production formula and 0.55 threshold."""
    wins, losses, draws = result["a"], result["b"], result["draws"]
    total = wins + losses + draws
    score = (wins + 0.5 * draws) / games
    return {
        "result": result,
        "total": total,
        "score": score,
        "accepted": score >= ACCEPT_THRESHOLD,
        "threshold": ACCEPT_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# timed suites                                                                #
# --------------------------------------------------------------------------- #

def _time_python_suite(cfg, games):
    """One python-arena suite through arena.play_match (FakeNet evaluators)."""
    evaluator = FakeEvaluator()
    net_a, net_b = FakeNet(evaluator), FakeNet(evaluator)
    t0 = time.perf_counter()
    result = arena.play_match(net_a, net_b, cfg, games)
    wall = time.perf_counter() - t0
    return wall, result, None


def _time_native_suite_fake(cfg, games, max_batch):
    """One native-arena suite through native_arena.play_match (FakeEvaluator),
    with transcript capture for the Ticket-C golden check."""
    evaluator = FakeEvaluator()
    with _TranscriptCapture() as cap:
        t0 = time.perf_counter()
        result = native_arena.play_match(
            None, None, cfg, games,
            evaluate_a=evaluator.logits_and_values,
            evaluate_b=evaluator.logits_and_values,
            max_batch=max_batch,
        )
        wall = time.perf_counter() - t0
    return wall, result, _transcripts_hash(cap.game_transcripts())


def _time_native_suite_gpu(engine, candidate_sd, best_sd, games, max_batch):
    """One production native-arena suite through NativeArenaEngine.play_match,
    with transcript capture for the within-process reproducibility check."""
    with _TranscriptCapture() as cap:
        t0 = time.perf_counter()
        result = engine.play_match(
            candidate_sd, best_sd, games, max_batch=max_batch
        )
        wall = time.perf_counter() - t0
    return wall, result, _transcripts_hash(cap.game_transcripts())


# --------------------------------------------------------------------------- #
# telemetry verification (Ticket A path)                                      #
# --------------------------------------------------------------------------- #

def _verify_telemetry(path, expected_arena_records, games, sims):
    """Every non-empty line parses as JSON; count the phase="arena" records and
    check their required fields.  Returns (ok, arena_records)."""
    records = []
    parsed = 0
    parse_ok = True
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_ok = False
                continue
            parsed += 1
    arena_records = [r for r in records
                     if r.get("type") == "phase" and r.get("phase") == "arena"]
    ok = parse_ok and parsed >= 1 and len(arena_records) >= expected_arena_records
    for rec in arena_records:
        ok &= rec.get("schema") == "telemetry/v1"
        ok &= rec.get("type") == "phase"
        ok &= rec.get("phase") == "arena"
        ok &= isinstance(rec.get("duration_s"), (int, float)) and rec["duration_s"] > 0
        ok &= rec.get("arena_games") == int(games)
        ok &= rec.get("arena_sims") == int(sims)
    return ok, arena_records


# --------------------------------------------------------------------------- #
# reporting                                                                   #
# --------------------------------------------------------------------------- #

class _Report:
    def __init__(self):
        self.checks = []

    def check(self, ok, label):
        self.checks.append((bool(ok), label))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    @property
    def all_pass(self):
        return all(ok for ok, _ in self.checks)


# --------------------------------------------------------------------------- #
# modes                                                                       #
# --------------------------------------------------------------------------- #

def run_compare_python(args) -> int:
    results_dir = Path(args.telemetry_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = results_dir / "arena-telemetry-compare-python.jsonl"
    if telemetry_path.exists():
        telemetry_path.unlink()

    cfg = _arena_cfg(args.sims, args.games, args.opening_plies, args.seed,
                     telemetry_path, device="cpu")
    print("=" * 78)
    print("arena_bench.py --compare-python  (CPU-only, deterministic FakeEvaluator)")
    print(f"  suite: games={args.games} sims={args.sims} "
          f"opening_plies={args.opening_plies} seed={args.seed} "
          f"repeats={args.repeats} max_batch={args.max_batch}")
    print("=" * 78)

    # Untimed warm-up (2 games each) so lazy imports / first-call init never
    # pollute the timed repeats; telemetry disabled so no stray records.
    if args.warmup:
        warm_cfg = _arena_cfg(args.sims, 2, args.opening_plies, args.seed,
                              telemetry_path, device="cpu")
        warm_cfg.telemetry_enabled = False
        arena.play_match(FakeNet(FakeEvaluator()), FakeNet(FakeEvaluator()),
                         warm_cfg, 2)
        warm_ev = FakeEvaluator()
        native_arena.play_match(None, None, warm_cfg, 2,
                                evaluate_a=warm_ev.logits_and_values,
                                evaluate_b=warm_ev.logits_and_values)
        print("warm-up: one 2-game run per engine (untimed)")

    py_times, py_results = [], []
    for i in range(args.repeats):
        wall, result, _ = _time_python_suite(cfg, args.games)
        py_times.append(wall)
        py_results.append(result)
        print(f"python suite run {i + 1}: {wall:8.2f}s  {result}")

    nat_times, nat_results, nat_hashes = [], [], []
    for i in range(args.repeats):
        wall, result, thash = _time_native_suite_fake(cfg, args.games,
                                                      args.max_batch)
        nat_times.append(wall)
        nat_results.append(result)
        nat_hashes.append(thash)
        print(f"native suite run {i + 1}: {wall:8.2f}s  {result}  "
              f"transcript_hash={thash}")

    py_median = statistics.median(py_times)
    nat_median = statistics.median(nat_times)
    speedup = py_median / nat_median if nat_median > 0 else float("inf")
    print("-" * 78)
    print(f"python median: {py_median:8.2f}s   "
          f"(reference ~125.49s, range 112.54-141.49s)")
    print(f"native median: {nat_median:8.2f}s   (acceptance: <= 60s)")
    print(f"speedup (python/native): {speedup:5.2f}x   (acceptance: >= 2.0x)")

    # Score semantics: production formula (wins + 0.5*draws)/games @ 0.55.
    print("-" * 78)
    print("score semantics (native fake-evaluator path):")
    for i, res in enumerate(nat_results):
        rep = _score_report(res, args.games)
        print(f"  run {i + 1}: {res} sum={rep['total']} "
              f"score={rep['score']:.3f} "
              f"accepted(>={rep['threshold']})={rep['accepted']}")

    ok_telemetry, arena_records = _verify_telemetry(
        telemetry_path, expected_arena_records=2 * args.repeats,
        games=args.games, sims=args.sims)
    print("-" * 78)
    print(f"telemetry: {len(arena_records)} phase=\"arena\" records "
          f"in {telemetry_path.name} (expected {2 * args.repeats})")
    for rec in arena_records:
        print(f"  {rec['phase']:>6} dur={rec['duration_s']:.3f}s "
              f"games={rec['arena_games']} sims={rec['arena_sims']} "
              f"schema={rec['schema']} type={rec['type']}")

    report = _Report()
    print("-" * 78)
    print("acceptance thresholds:")
    all_sums_ok = all(r["a"] + r["b"] + r["draws"] == args.games
                      for r in py_results + nat_results)
    report.check(all_sums_ok,
                 f"every suite's {{a,b,draws}} sums to {args.games}")
    golden_ok = all(h == GOLDEN_TRANSCRIPT_HASH for h in nat_hashes)
    report.check(golden_ok,
                 f"native transcript hash == Ticket-C golden "
                 f"{GOLDEN_TRANSCRIPT_HASH}")
    report.check(nat_median <= 60.0,
                 f"native full-suite median <= 60s ({nat_median:.2f}s)")
    report.check(speedup >= 2.0,
                 f"native/python speedup >= 2.0x ({speedup:.2f}x)")
    report.check(ok_telemetry,
                 'phase="arena" telemetry record emitted and parses')

    print("=" * 78)
    print(f"exit code: {0 if report.all_pass else 1}")
    return 0 if report.all_pass else 1


def run_gpu(args) -> int:
    if not torch.cuda.is_available():
        print("ERROR: --gpu requires a visible CUDA device (torch.cuda.is_available() "
              "is False). The benchmark must run outside the file sandbox so the "
              "RTX 2080 Ti (/dev/nvidia*) is reachable.", file=sys.stderr)
        return 2

    results_dir = Path(args.telemetry_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = results_dir / "arena-telemetry-gpu.jsonl"
    if telemetry_path.exists():
        telemetry_path.unlink()

    cfg = _arena_cfg(args.sims, args.games, args.opening_plies, args.seed,
                     telemetry_path, device="cuda")
    print("=" * 78)
    print("arena_bench.py --gpu  (production InferenceRuntime on CUDA)")
    print(f"  device: {torch.cuda.get_device_name(0)}  "
          f"mem={torch.cuda.get_device_properties(0).total_memory / 1e6:.0f}MB")
    print(f"  suite: games={args.games} sims={args.sims} "
          f"opening_plies={args.opening_plies} seed={args.seed} "
          f"repeats={args.repeats} max_batch={args.max_batch}")
    print("=" * 78)

    from model import ChessNet

    # Production path: one NativeArenaEngine (two persistent InferenceRuntimes,
    # weights copied via state_dict; the trainer's nets are never touched).
    candidate_net = ChessNet(cfg)
    best_net = ChessNet(cfg)
    candidate_sd = candidate_net.state_dict()
    best_sd = best_net.state_dict()
    t0 = time.perf_counter()
    engine = native_arena.NativeArenaEngine(cfg)
    engine_s = time.perf_counter() - t0
    print(f"engine construction (2x InferenceRuntime + torch.compile): "
          f"{engine_s:.2f}s (one-time, reported separately)")

    times, results, hashes = [], [], []
    try:
        for i in range(args.repeats):
            wall, result, thash = _time_native_suite_gpu(
                engine, candidate_sd, best_sd, args.games, args.max_batch)
            times.append(wall)
            results.append(result)
            hashes.append(thash)
            print(f"native gpu run {i + 1}: {wall:8.2f}s  {result}  "
                  f"transcript_hash={thash}")
    finally:
        try:
            engine.candidate_fn.runtime.close()
            engine.best_fn.runtime.close()
        except Exception:  # noqa: BLE001 - cleanup must not mask results
            pass

    median = statistics.median(times)
    print("-" * 78)
    print(f"native gpu median: {median:8.2f}s   (acceptance: <= 60s)")

    print("score semantics (production path):")
    for i, res in enumerate(results):
        score = (res["a"] + 0.5 * res["draws"]) / args.games
        print(f"  run {i + 1}: {res} sum={res['a'] + res['b'] + res['draws']} "
              f"score={score:.3f} accepted(>={ACCEPT_THRESHOLD})="
              f"{score >= ACCEPT_THRESHOLD}")

    ok_telemetry, arena_records = _verify_telemetry(
        telemetry_path, expected_arena_records=args.repeats,
        games=args.games, sims=args.sims)
    print("-" * 78)
    print(f"telemetry: {len(arena_records)} phase=\"arena\" records "
          f"in {telemetry_path.name} (expected {args.repeats})")
    for rec in arena_records:
        print(f"  {rec['phase']:>6} dur={rec['duration_s']:.3f}s "
              f"games={rec['arena_games']} sims={rec['arena_sims']} "
              f"schema={rec['schema']} type={rec['type']}")

    report = _Report()
    print("-" * 78)
    print("acceptance thresholds:")
    all_sums_ok = all(r["a"] + r["b"] + r["draws"] == args.games
                      for r in results)
    report.check(all_sums_ok,
                 f"every suite's {{a,b,draws}} sums to {args.games}")
    repro_ok = len(set(hashes)) == 1
    report.check(repro_ok,
                 f"reproducible: identical transcript hash across all "
                 f"{args.repeats} runs ({hashes[0] if hashes else '-'})")
    report.check(median <= 60.0,
                 f"native gpu full-suite median <= 60s ({median:.2f}s)")
    report.check(ok_telemetry,
                 'phase="arena" telemetry record emitted and parses')

    print("=" * 78)
    print(f"exit code: {0 if report.all_pass else 1}")
    return 0 if report.all_pass else 1


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Arena benchmark: python vs native (Ticket D).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compare-python", action="store_true",
                      help="CPU-only: time python arena vs native arena with the "
                           "deterministic FakeEvaluator")
    mode.add_argument("--gpu", action="store_true",
                      help="time native_arena.play_match with the production "
                           "InferenceRuntime runtimes on CUDA")
    parser.add_argument("--sims", type=int, default=40, help="simulations per search")
    parser.add_argument("--games", type=int, default=20, help="arena games (even)")
    parser.add_argument("--opening-plies", type=int, default=8,
                        help="opening depth of the deterministic suite")
    parser.add_argument("--seed", type=int, default=424242, help="arena suite seed")
    parser.add_argument("--repeats", type=int, default=3,
                        help="suite repetitions per engine (median over these)")
    parser.add_argument("--max-batch", type=int, default=256,
                        help="native gather_leaves batch cap (arena default)")
    parser.add_argument("--telemetry-dir", type=str,
                        default=str(DEFAULT_TELEMETRY_DIR),
                        help="directory for the telemetry JSONL evidence")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the untimed 2-game warm-up runs")
    args = parser.parse_args(argv)

    if args.games % 2 != 0:
        print(f"ERROR: arena games must be even (got {args.games})",
              file=sys.stderr)
        return 2
    if args.repeats < 1:
        print("ERROR: --repeats must be >= 1", file=sys.stderr)
        return 2
    args.warmup = not args.no_warmup

    if args.compare_python:
        return run_compare_python(args)
    return run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
