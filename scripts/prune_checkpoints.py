#!/usr/bin/env python3
"""Prune chess-RL checkpoints to a retention policy (DRY-RUN by default).

Retention (union of rules) applied to versioned snapshots ``ckpt-iter*.pt``:

  1. active files (``latest.pt``, ``best.pt``, ``checkpoint_meta.json``,
     ``best_meta.json``) — always kept;
  2. ``milestones/`` — always kept (arena-accepted champions);
  3. the last ``--keep`` snapshots by iteration;
  4. a sparse safety ladder: every ``--ladder`` iterations;
  5. snapshots within ``--neighbor`` iterations of each arena-accepted
     promotion (read from the run's ``training.jsonl``).

Everything else (ordinary per-iteration snapshots) is deletable.  Archived
``best-*``/``latest-*`` files are only touched with ``--prune-archives``.

Safety: this script NEVER deletes ``latest.pt`` / ``best.pt`` / the two meta
files / anything under ``milestones/``.  Versioned snapshots are independent
files (link count 1 once ``latest.pt`` is replaced on the next save), so
removing one frees real space.

Default is dry-run: it prints a manifest and a freed-space estimate without
deleting anything.  Pass ``--apply`` to actually remove the listed files.

Examples:
  python3 scripts/prune_checkpoints.py                      # dry-run, v2 dir
  python3 scripts/prune_checkpoints.py --keep 5 --ladder 200 --apply
"""
import argparse
import json
import os
import re
import sys

SNAPSHOT_RE = re.compile(r"^ckpt-iter(\d+)-")
ARCHIVE_RE = re.compile(r"^(best|latest)-\d{8}-\d{6}\.pt$")
ACTIVE_FILES = {"latest.pt", "best.pt", "checkpoint_meta.json", "best_meta.json"}
MILESTONE_DIR = "milestones"


def human(size_bytes):
    size_bytes = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0 or unit == "TB":
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0


def parse_snapshots(checkpoint_dir):
    """Return {iteration: [(path, size), ...]} for versioned snapshots."""
    snapshots = {}
    for name in os.listdir(checkpoint_dir):
        m = SNAPSHOT_RE.match(name)
        if not m:
            continue
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            continue
        it = int(m.group(1))
        snapshots.setdefault(it, []).append((path, os.path.getsize(path)))
    return snapshots


def arena_promotion_iters(training_jsonl):
    """Iterations at which the arena accepted a new champion."""
    iters = set()
    if not training_jsonl or not os.path.exists(training_jsonl):
        return iters
    with open(training_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "arena" and rec.get("accepted"):
                iters.add(int(rec["iteration"]))
    return iters


def read_checkpoint_interval(checkpoint_dir):
    """Best-effort checkpoint_every_iterations from checkpoint_meta.json."""
    meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return int(meta.get("config", {}).get("checkpoint_every_iterations", 20))
    except (OSError, ValueError, TypeError):
        return 20


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", default=None,
                   help="checkpoint dir to prune (default: checkpoints/v2)")
    p.add_argument("--training-jsonl", default=None,
                   help="training.jsonl with arena events (default: repo root)")
    p.add_argument("--keep", type=int, default=5,
                   help="keep the last N snapshots by iteration (default 5)")
    p.add_argument("--ladder", type=int, default=200,
                   help="also keep every Nth iteration (0 to disable; default 200)")
    p.add_argument("--neighbor", type=int, default=None,
                   help="keep snapshots within +/-N iters of a promotion "
                        "(default: checkpoint_every_iterations)")
    p.add_argument("--prune-archives", action="store_true",
                   help="also delete archived best-*/latest-* files")
    p.add_argument("--apply", action="store_true",
                   help="actually delete (default is dry-run)")
    args = p.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = args.checkpoint_dir or os.path.join(repo_root,
                                                         "checkpoints", "v2")
    training_jsonl = args.training_jsonl or os.path.join(repo_root,
                                                         "training.jsonl")
    if not os.path.isdir(checkpoint_dir):
        print(f"ERROR: checkpoint dir not found: {checkpoint_dir}", file=sys.stderr)
        return 2

    snapshots = parse_snapshots(checkpoint_dir)
    # Dedupe: keep only the newest file per iteration (duplicate writes happen
    # across restarts); the older duplicate is always deletable.
    newest = {}          # iteration -> (path, size)
    duplicates = []      # (path, size) redundant copies of an iteration
    for it, files in snapshots.items():
        newest_file = max(files, key=lambda t: os.path.getmtime(t[0]))
        newest[it] = newest_file
        duplicates.extend(f for f in files if f != newest_file)

    interval = read_checkpoint_interval(checkpoint_dir)
    neighbor = args.neighbor if args.neighbor is not None else interval
    promotions = arena_promotion_iters(training_jsonl)

    # ---- compute the keep set (union of rules) ----
    keep_iters = set()
    if snapshots:
        sorted_iters = sorted(newest)
        keep_iters.update(sorted_iters[-args.keep:])
        if args.ladder > 0:
            keep_iters.update(i for i in sorted_iters if i % args.ladder == 0)
        for promo in promotions:
            keep_iters.update(
                i for i in sorted_iters
                if abs(i - promo) <= neighbor
            )

    keep_files = {newest[i][0] for i in keep_iters}
    delete_files = [newest[i] for i in sorted(newest) if i not in keep_iters]
    delete_files.extend(duplicates)

    archives = []
    for name in os.listdir(checkpoint_dir):
        if ARCHIVE_RE.match(name):
            path = os.path.join(checkpoint_dir, name)
            if os.path.isfile(path):
                archives.append((path, os.path.getsize(path)))

    deletable_archives = archives if args.prune_archives else []
    freed = sum(size for _, size in delete_files) + \
        sum(size for _, size in deletable_archives)

    # ---- report ----
    print(f"checkpoint dir : {checkpoint_dir}")
    print(f"training.jsonl : {training_jsonl}")
    print(f"snapshots      : {len(newest)} distinct iterations "
          f"({len(duplicates)} duplicate files)")
    print(f"keep rules     : last {args.keep}"
          + (f" + every {args.ladder}" if args.ladder > 0 else "")
          + f" + within +-{neighbor} of {len(promotions)} promotions "
            f"(iters {sorted(promotions)})")
    print(f"keep snapshots : {len(keep_iters)}")
    print(f"delete snapshots: {len(delete_files)}")
    if archives:
        print(f"archives       : {len(archives)} "
              f"({'pruned' if args.prune_archives else 'kept'})")
    print(f"space to free  : {human(freed)}")
    print(f"mode           : {'APPLY' if args.apply else 'DRY-RUN (no changes)'}")
    print("-" * 72)

    for path, size in sorted(deletable_archives) + sorted(delete_files):
        print(f"{'DEL ' if args.apply else 'del '} {os.path.basename(path):<44} "
              f"{human(size):>9}")

    if not args.apply:
        print("\nNo files removed. Re-run with --apply to delete the 'del' list.")
    else:
        removed = 0
        for path, _ in deletable_archives + delete_files:
            try:
                os.remove(path)
                removed += 1
            except OSError as exc:
                print(f"  failed to remove {path}: {exc}", file=sys.stderr)
        print(f"\nRemoved {removed} files ({human(freed)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
