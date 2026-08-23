#!/usr/bin/env python3
"""Backfill immutable arena milestones from existing versioned snapshots.

For every arena-accepted event in ``training.jsonl``, locate the versioned
snapshot at that iteration and extract the accepted-best weights (the ``best``
key of the full resumable snapshot) into
``<checkpoint_dir>/milestones/best-genNNNN-iterNNNN-<run_id>.pt`` with a JSON
sidecar matching ``train._save_milestone``.

Retroactively captures historical champions (which previously lived only
inside ~99 MB full snapshots) as ~8 MB weights-only milestones so they survive
snapshot pruning.  Idempotent: already-present milestones are skipped.

Examples:
  python3 scripts/backfill_milestones.py --checkpoint-dir checkpoints/v2 \
      --training-jsonl training.jsonl --dry-run
  python3 scripts/backfill_milestones.py --checkpoint-dir checkpoints/v2 \
      --training-jsonl training.jsonl
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone


def read_accepted_events(training_jsonl):
    """Return ordered list of accepted-arena events (dicts)."""
    events = []
    if not os.path.exists(training_jsonl):
        return events
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
                events.append(rec)
    return events


def find_snapshot(checkpoint_dir, iteration):
    """Newest versioned snapshot for an iteration, or None."""
    candidates = sorted(glob.glob(
        os.path.join(checkpoint_dir, f"ckpt-iter{iteration:04d}-*.pt")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def read_config(checkpoint_dir):
    meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")
    try:
        with open(meta_path) as f:
            return json.load(f).get("config", {})
    except (OSError, ValueError):
        return {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--training-jsonl", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not os.path.isdir(args.checkpoint_dir):
        print(f"ERROR: {args.checkpoint_dir} not a directory", file=sys.stderr)
        return 2

    import torch  # local: keep the import cost out of --help

    events = read_accepted_events(args.training_jsonl)
    config = read_config(args.checkpoint_dir)
    milestones_dir = os.path.join(args.checkpoint_dir, "milestones")
    os.makedirs(milestones_dir, exist_ok=True)

    if not events:
        print("No accepted arena events found; nothing to backfill.")
        return 0

    done = skipped = missing = 0
    for ev in events:
        run_id = ev["run_id"]
        iteration = int(ev["iteration"])
        generation = int(ev["generation"])
        snap = find_snapshot(args.checkpoint_dir, iteration)
        if snap is None:
            print(f"  SKIP  gen {generation} iter {iteration}: no snapshot")
            missing += 1
            continue

        stem = f"best-gen{generation:04d}-iter{iteration:04d}-{run_id}"
        pt_path = os.path.join(milestones_dir, stem + ".pt")
        json_path = os.path.join(milestones_dir, stem + ".json")
        if os.path.exists(pt_path) and os.path.exists(json_path):
            print(f"  skip  {stem} (already present)")
            skipped += 1
            continue

        payload = torch.load(snap, map_location="cpu", weights_only=False)
        best_sd = payload.get("best")
        if best_sd is None:
            print(f"  SKIP  {stem}: snapshot has no 'best' key")
            missing += 1
            continue

        meta = {
            "schema_version": payload.get("schema_version", 2),
            "checkpoint_format": "schema-v3",
            "run_id": run_id,
            "iteration": iteration,
            "generation": generation,
            "architecture_id": payload.get("architecture_id"),
            "policy_size": int(payload.get("policy_size", 0)),
            "num_input_planes": int(payload.get("num_input_planes", 0)),
            "board_size": int(payload.get("board_size", 0)),
            "arena": {
                "wins": int(ev.get("wins", 0)),
                "draws": int(ev.get("draws", 0)),
                "losses": int(ev.get("losses", 0)),
                "score": float(ev.get("score", 0.0)),
                "accept_threshold": float(config.get(
                    "arena_accept_threshold", 0.55)),
            },
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        if args.dry_run:
            print(f"  WOULD WRITE  {stem}.pt  (from {os.path.basename(snap)})")
            done += 1
            continue

        torch.save(best_sd, pt_path + ".tmp")
        os.replace(pt_path + ".tmp", pt_path)
        with open(json_path + ".tmp", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(json_path + ".tmp", json_path)
        print(f"  wrote  {stem}.pt  (gen {generation}, iter {iteration}, "
              f"score {meta['arena']['score']})")
        done += 1

    print(f"\nbackfill complete: {done} written"
          + (", skipped " + str(skipped) if skipped else "")
          + (", missing " + str(missing) if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
