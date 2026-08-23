#!/usr/bin/env python3
"""Read-only replay-diversity audit for schema-v3 checkpoints.

Given a checkpoint (``--checkpoint /path/to/latest.pt``), emits machine-readable
JSON describing how much genuine variety the replay buffer actually holds.
This is the diagnostic that exposes the fixed-seed self-play failure mode:
tens of thousands of examples collapsing onto a handful of repeated
trajectories.

Properties:
  * READ-ONLY — the checkpoint is opened for reading only and never mutated.
  * Stable BLAKE2 hashes; "exact example" hashes include the packed state, the
    non-binary state extras, the sparse policy (indices + probabilities) AND z.
  * Inferred values (game-start / complete-game / trajectory counts) are clearly
    labelled and derived from the 8-step history encoding (all future-history
    planes are zero exactly at ply 0).
  * Model tensors are loaded with map_location="cpu" (never touched on CUDA).
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch


def _digest(b):
    return hashlib.blake2b(b, digest_size=16).digest()


def _example_digest(positions, i, offsets, legal, probs, zs):
    """Stable digest of example i's (state, policy, z) content."""
    a, b = int(offsets[i]), int(offsets[i + 1])
    h = hashlib.blake2b(digest_size=16)
    h.update(positions[i].tobytes())
    h.update(legal[a:b].tobytes())
    h.update(probs[a:b].tobytes())
    h.update(zs[i].tobytes())
    return h.digest()


def audit_replay(checkpoint_path, games_per_iteration=None):
    """Return a replay-diversity report dict for ``checkpoint_path``."""
    path = os.path.abspath(checkpoint_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)

    replay = payload.get("replay")
    if not isinstance(replay, dict):
        raise ValueError(f"checkpoint {path} has no replay state_dict")

    positions = np.asarray(replay["positions"], dtype=np.uint8)
    zs = np.asarray(replay["z"], dtype=np.float32)
    offsets = np.asarray(replay["offsets"], dtype=np.int64)
    legal = np.asarray(replay["legal_idx"], dtype=np.int32)
    probs = np.asarray(replay["probs"], dtype=np.float32)
    extra_idx = np.asarray(replay["state_extra_idx"], dtype=np.int32)
    extra_values = np.asarray(replay["state_extra_values"], dtype=np.float32)
    extra_offsets = np.asarray(replay["state_extra_offsets"], dtype=np.int64)

    num_input_planes = int(replay.get("num_input_planes", 104))
    board_size = int(replay.get("board_size", 8))
    n = positions.shape[0]

    config = payload.get("config") or {}
    if games_per_iteration is None:
        games_per_iteration = config.get("games_per_iteration")

    # Future-history planes (steps 1..history_steps-1) are planes
    # [12 : num_input_planes - 8]; all-zero ONLY at ply 0 (game start).
    future_start = 12
    future_end = num_input_planes - 8

    state_hashes = set()
    exact_hashes = set()
    game_starts = []

    for i in range(n):
        state_hashes.add(_digest(positions[i].tobytes()))

        a, b = int(offsets[i]), int(offsets[i + 1])
        ea, eb = int(extra_offsets[i]), int(extra_offsets[i + 1])
        h = hashlib.blake2b(digest_size=16)
        h.update(positions[i].tobytes())
        h.update(extra_idx[ea:eb].tobytes())
        h.update(extra_values[ea:eb].tobytes())
        h.update(legal[a:b].tobytes())
        h.update(probs[a:b].tobytes())
        h.update(zs[i].tobytes())
        exact_hashes.add(h.digest())

        state = np.unpackbits(positions[i], bitorder="big").astype(np.float32)
        state = state.reshape(num_input_planes, board_size, board_size)
        if state[future_start:future_end].sum() == 0.0:
            game_starts.append(i)

    # Trajectories: one game runs from a detected start to the next start (or
    # end of buffer).  The trailing game may be truncated at the ring-buffer
    # boundary; this is noted, not silently corrected.
    traj_counts = {}
    boundaries = game_starts + [n]
    for gi in range(len(game_starts)):
        s = game_starts[gi]
        e = boundaries[gi + 1]
        th = hashlib.blake2b(digest_size=16)
        for i in range(s, e):
            th.update(_example_digest(positions, i, offsets, legal, probs, zs))
        d = th.digest()
        traj_counts[d] = traj_counts.get(d, 0) + 1

    unique_traj = len(traj_counts)
    most_traj = max(traj_counts.values()) if traj_counts else 0

    # games_per_iteration-aligned blocks of complete games.
    unique_blocks = None
    most_block = None
    gpi = int(games_per_iteration) if games_per_iteration else 0
    if gpi > 0 and len(game_starts) >= gpi:
        block_counts = {}
        for b0 in range(0, len(game_starts) - gpi + 1, gpi):
            bh = hashlib.blake2b(digest_size=16)
            for gi in range(b0, b0 + gpi):
                s = game_starts[gi]
                e = (game_starts[gi + 1] if gi + 1 < len(game_starts) else n)
                for i in range(s, e):
                    bh.update(_example_digest(positions, i, offsets, legal,
                                              probs, zs))
            d = bh.digest()
            block_counts[d] = block_counts.get(d, 0) + 1
        unique_blocks = len(block_counts)
        most_block = max(block_counts.values())

    return {
        "checkpoint": path,
        "read_only": True,
        "iteration": payload.get("iteration"),
        "generation": payload.get("generation"),
        "run_id": payload.get("run_id"),
        "architecture_id": payload.get("architecture_id"),
        "games_per_iteration": gpi or None,
        "replay_example_count": int(n),
        "unique_packed_states": len(state_hashes),
        "unique_exact_examples": len(exact_hashes),
        "unique_state_fraction": (len(state_hashes) / n) if n else 0.0,
        "unique_example_fraction": (len(exact_hashes) / n) if n else 0.0,
        "inferred_game_start_count": len(game_starts),
        "inferred_complete_game_count": len(game_starts),
        "unique_full_game_trajectory_hashes": unique_traj,
        "most_repeated_trajectory_count": most_traj,
        "unique_12_game_blocks": unique_blocks,
        "most_repeated_12_game_block_count": most_block,
        "notes": [
            "read-only: the checkpoint was opened for reading only and not "
            "modified.",
            "game-start / complete-game / trajectory counts are INFERRED from "
            "the 8-step history encoding (future-history planes are all zero "
            "exactly at ply 0); the trailing game may be truncated at the "
            "ring-buffer boundary and is counted as a start, not as a "
            "verified-complete game.",
            "exact-example hashes include the packed state, non-binary state "
            "extras, sparse policy (indices + probabilities) and z.",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Read-only replay-diversity audit for a schema-v3 checkpoint"
    )
    ap.add_argument("--checkpoint", required=True,
                    help="path to a schema-v3 latest.pt")
    ap.add_argument("--games-per-iteration", type=int, default=None,
                    help="override the games-per-iteration block size "
                         "(else read from the checkpoint config)")
    args = ap.parse_args(argv)
    report = audit_replay(args.checkpoint,
                          games_per_iteration=args.games_per_iteration)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
