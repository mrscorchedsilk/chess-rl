"""Native MCTS throughput benchmark + shared deterministic fake evaluator.

Task 5 (native cache-friendly MCTS core).  This module is the single home of
the deterministic fake policy/value head used by BOTH the pytest suite
(``tests/test_native_mcts.py``) and the CLI benchmark, mirroring the existing
``benchmarks/native_policy_encoder_parity.py`` harness pattern.

The fake evaluator removes all GPU / network variance from search throughput
measurements: logits are a fixed function of the action index
(``-idx / 1024``, strictly decreasing, so every legal move gets a distinct
prior and no selection ties occur), and values are a fixed function of the
encoded input planes (side to move, halfmove clock, repetition), so the same
position always yields the same value in the Python reference and the native
core.

CLI::

    python benchmarks/native_mcts.py --sims 100 --runs 100 --compare-python

reports native wall time per 100-simulation search, the Python ``mcts.py``
reference wall time, and the native/Python speedup (target >= 2x).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess  # noqa: E402
import chess_rl_native  # noqa: E402
import encoding  # noqa: E402
import mcts as mcts_reference  # noqa: E402
import torch  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
POLICY_SIZE = 4672
NUM_PLANES = 104
BATCH_SIZE = 32

# Deterministic corpus cycled by the benchmark (opening, middlegame, tactical,
# sparse endgame).
BENCH_POSITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (START_FEN, ()),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", ()),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ()),
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", ()),
    (START_FEN, ("e2e4", "e7e5", "g1f3", "b8c6")),
)


def fake_logits_row() -> np.ndarray:
    """[-0.0009766, -0.0019531, ...]: strictly decreasing float32 logits."""
    return -np.arange(POLICY_SIZE, dtype=np.float32) / 1024.0


def fake_values(inputs: np.ndarray) -> np.ndarray:
    """Deterministic float32 values in [-0.2, 0.3] from the encoded planes."""
    return (
        0.4 * inputs[:, 102, 0, 0]
        - 0.2
        + 0.1 * (inputs[:, 103, 0, 0] > 0.5)
    ).astype(np.float32)


class FakeEvaluator:
    """Deterministic fake policy/value head shared by native and Python.

    ``boost`` maps side-to-move ('w'/'b') to an action index whose logit is
    lifted to 10.0 (used by the mate-in-1 / claimable-draw tests to steer the
    first simulations deterministically).  ``None`` logits otherwise.
    """

    def __init__(self, boost: dict[str, int] | None = None) -> None:
        self.logits_row = fake_logits_row()
        self.boost = dict(boost or {})

    # ---- native two-phase API: (inputs, offsets, indices) -> (logits, values)

    def logits_and_values(
        self,
        inputs: np.ndarray,
        legal_offsets: np.ndarray,
        legal_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        logits = self.logits_row[legal_indices].copy()  # CSR order, float32
        for b in range(inputs.shape[0]):
            side = "w" if inputs[b, 96, 0, 0] > 0.5 else "b"
            idx = self.boost.get(side)
            if idx is None:
                continue
            lo, hi = int(legal_offsets[b]), int(legal_offsets[b + 1])
            row = legal_indices[lo:hi]
            pos = int(np.searchsorted(row, idx))
            if pos < row.size and row[pos] == idx:
                logits[lo + pos] = 10.0
        return logits, fake_values(inputs)

    # ---- python mcts.py interface: torch [B,104,8,8] -> (logits, values)

    def python_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        arr = x.detach().numpy()
        batch = arr.shape[0]
        logits = np.broadcast_to(self.logits_row, (batch, POLICY_SIZE)).copy()
        for b in range(batch):
            side = "w" if arr[b, 96, 0, 0] > 0.5 else "b"
            idx = self.boost.get(side)
            if idx is not None:
                logits[b, idx] = 10.0
        values = fake_values(arr)
        return torch.from_numpy(logits), torch.from_numpy(values[:, None])


class FakeNet(torch.nn.Module):
    """torch.nn.Module wrapper so the reference mcts.py can call it."""

    def __init__(self, evaluator: FakeEvaluator) -> None:
        super().__init__()
        self.evaluator = evaluator

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.evaluator.python_forward(x)


def make_python_mcts(
    evaluator: FakeEvaluator,
    num_simulations: int = 100,
    c_puct: float = 1.25,
    virtual_loss: float = 3.0,
) -> mcts_reference.MCTS:
    cfg = SimpleNamespace(
        c_puct=c_puct,
        virtual_loss=virtual_loss,
        batch_size=BATCH_SIZE,
        num_simulations=num_simulations,
        device="cpu",
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    return mcts_reference.MCTS(FakeNet(evaluator), cfg)


def make_native_mcts(
    num_simulations: int = 100,
    c_puct: float = 1.25,
    virtual_loss: float = 3.0,
    dirichlet_epsilon: float = 0.0,
    seed: int = 42,
    max_batch: int = BATCH_SIZE,
) -> chess_rl_native.MCTS:
    return chess_rl_native.MCTS(
        c_puct=c_puct,
        virtual_loss=virtual_loss,
        num_simulations=num_simulations,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=dirichlet_epsilon,
        seed=seed,
    )


def run_native_search(
    mcts: chess_rl_native.MCTS,
    start_fen: str,
    history_moves: tuple[str, ...],
    evaluator: FakeEvaluator,
    temperature: float = 1.0,
    max_batch: int = BATCH_SIZE,
) -> list[tuple[str, float]]:
    """Drive the two-phase API to completion; return policy(temperature)."""
    mcts.set_root(start_fen, list(history_moves))
    guard = mcts.num_simulations + 8
    while not mcts.is_complete():
        guard -= 1
        if guard < 0:
            raise RuntimeError("gather_leaves did not terminate the search")
        tokens, inputs, legal_offsets, legal_indices = mcts.gather_leaves(max_batch)
        if not tokens:
            continue  # batch resolved internally as terminals; sims still ran
        logits, values = evaluator.logits_and_values(
            np.asarray(inputs), np.asarray(legal_offsets), np.asarray(legal_indices)
        )
        mcts.apply_evaluations(tokens, legal_offsets, logits, values)
    return mcts.policy(temperature)


def run_python_search(
    board: chess.Board,
    evaluator: FakeEvaluator,
    num_simulations: int,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Reference mcts.py search (no root noise) -> {uci: prob}."""
    reference = make_python_mcts(evaluator, num_simulations=num_simulations)
    result = reference.search(
        board, temperature=temperature, num_sims=num_simulations, add_root_noise=False
    )
    return {move.uci(): float(prob) for move, prob in result.items()}


# ---------------------------------------------------------------------------
# CLI benchmark
# ---------------------------------------------------------------------------


def _time_native(sims: int, runs: int, evaluator: FakeEvaluator) -> float:
    total = 0.0
    for run in range(runs):
        fen, history = BENCH_POSITIONS[run % len(BENCH_POSITIONS)]
        mcts = make_native_mcts(num_simulations=sims)
        start = time.perf_counter()
        run_native_search(mcts, fen, history, evaluator, temperature=1.0)
        total += time.perf_counter() - start
    return total / runs


def _time_python(sims: int, runs: int, evaluator: FakeEvaluator) -> float:
    total = 0.0
    for run in range(runs):
        fen, history = BENCH_POSITIONS[run % len(BENCH_POSITIONS)]
        board = chess.Board(fen)
        for uci in history:
            board.push_uci(uci)
        reference = make_python_mcts(evaluator, num_simulations=sims)
        start = time.perf_counter()
        reference.search(board, temperature=1.0, num_sims=sims, add_root_noise=False)
        total += time.perf_counter() - start
    return total / runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=100, help="simulations per search")
    parser.add_argument("--runs", type=int, default=100, help="searches per engine")
    parser.add_argument("--compare-python", action="store_true",
                        help="also benchmark the Python mcts.py reference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    evaluator = FakeEvaluator()
    print(f"native MCTS benchmark: sims={args.sims} runs={args.runs} "
          f"compare_python={args.compare_python}")

    # warm-up (JIT / page faults / torch init)
    _time_native(args.sims, 3, evaluator)
    if args.compare_python:
        _time_python(args.sims, 2, evaluator)

    native_ms = 1000.0 * _time_native(args.sims, args.runs, evaluator)
    print(f"native  : {native_ms:8.3f} ms/search "
          f"({1000.0 / native_ms * args.sims:9.0f} sims/s)")

    if args.compare_python:
        python_ms = 1000.0 * _time_python(args.sims, args.runs, evaluator)
        print(f"python  : {python_ms:8.3f} ms/search "
              f"({1000.0 / python_ms * args.sims:9.0f} sims/s)")
        speedup = python_ms / native_ms
        print(f"speedup : {speedup:8.2f}x native/python (target >= 2.0x)")
        return 0 if speedup >= 2.0 else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
