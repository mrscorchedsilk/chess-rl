import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from gpu_runtime import InferenceRuntime
from model import ChessNet

cfg = Config()
m = ChessNet(cfg).cuda().eval()
rt = InferenceRuntime(cfg=cfg, model=m, amp=True, compile=True)

rng = np.random.default_rng(0)
inputs = rng.standard_normal((1, 104, 8, 8)).astype(np.float32)
lengths = rng.integers(1, 41, size=1)
offsets = np.zeros(2, np.int32)
offsets[1] = int(lengths[0])
idx = np.sort(rng.choice(4672, int(lengths[0]), replace=False)).astype(np.int32)

for _ in range(5):
    rt.evaluate(inputs, offsets, idx)
torch.cuda.synchronize()
N = 300

t0 = time.perf_counter()
for _ in range(N):
    rt.evaluate(inputs, offsets, idx)
torch.cuda.synchronize()
wall = (time.perf_counter() - t0) / N * 1000
print(f"evaluate          {wall:.4f} ms")

t0 = time.perf_counter()
for _ in range(N):
    call = rt.prepare(inputs, offsets, idx)
print(f"prepare (cpu)     {(time.perf_counter() - t0) / N * 1000:.4f} ms")
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N):
    rt.forward_device(call)
torch.cuda.synchronize()
fwd = (time.perf_counter() - t0) / N * 1000
print(f"forward_device    {fwd:.4f} ms")

t0 = time.perf_counter()
for _ in range(N):
    rt.copy_back(call, call.legal, call.values)
print(f"copy_back (cpu)   {(time.perf_counter() - t0) / N * 1000:.4f} ms")
torch.cuda.synchronize()
print(f"-> transfer-side  {wall - fwd:.4f} ms  ({100*(wall-fwd)/wall:.1f}%)")
