"""Verify the new checkpoint helpers in train.py without touching real data."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train
from config import Config

cfg = Config()
tmp = tempfile.mkdtemp(prefix="ckpt-test-")
cfg.checkpoint_dir = tmp

best = os.path.join(tmp, "best.pt")
latest = os.path.join(tmp, "latest.pt")
with open(best, "wb") as f:
    f.write(b"BEST")
with open(latest, "wb") as f:
    f.write(b"LATEST")

# --- _archive_best: moves best.pt -> best-<ts>.pt ---
train._archive_best(cfg)
archived = [f for f in os.listdir(tmp) if f.startswith("best-") and f.endswith(".pt")]
assert not os.path.exists(best), "best.pt still exists after archive"
assert len(archived) == 1, f"expected 1 archived best, got {archived}"
assert open(os.path.join(tmp, archived[0]), "rb").read() == b"BEST"
print("OK  _archive_best  ->", archived[0])

# --- _snapshot_checkpoint: hardlink latest.pt -> ckpt-iterNNNN-<ts>.pt ---
snap = train._snapshot_checkpoint(cfg, 7)
assert snap and os.path.exists(snap), "snapshot not created"
assert os.path.basename(snap).startswith("ckpt-iter0007-"), f"bad name: {snap}"
assert os.path.samefile(snap, latest), "snapshot is not a hardlink to latest.pt"
print("OK  _snapshot_checkpoint ->", os.path.basename(snap))

shutil.rmtree(tmp)
print("ALL HELPER TESTS PASSED")
