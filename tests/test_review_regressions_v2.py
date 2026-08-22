"""Regression tests for independent-review findings in the v2 trainer."""

import os
import queue
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_collect_games_counts_envelopes_not_examples():
    import train
    from config import Config

    q = queue.Queue()
    q.put({"kind": "game", "examples": [("s", "p", 0.0)] * 30, "generation": 0})
    q.put({"kind": "game", "examples": [("s", "p", 0.0)] * 40, "generation": 0})
    cfg = Config()
    cfg.result_timeout_seconds = 0.1
    examples = train._collect_games(cfg, q, [], games_needed=2, expected_generation=0)
    assert len(examples) == 70


class RecordingBuffer:
    def __init__(self, n):
        self.n = n
        self.rows = []

    def __len__(self):
        return self.n

    def sample_indices(self, rows, device=None):
        rows = np.asarray(rows, dtype=np.int64)
        self.rows.extend(rows.tolist())
        b = len(rows)
        return (
            torch.zeros(b, 1, 1, 1),
            torch.tensor([[1.0, 0.0]] * b),
            torch.zeros(b, 1),
        )


def test_epoch_train_samples_across_full_replay_not_oldest_prefix():
    import train

    class Cfg:
        training_epochs = 1
        train_epoch_size = 8
        train_batch_size = 4

    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 3))

    class Heads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 3)

        def forward(self, x):
            y = self.fc(x.flatten(1))
            return y[:, :2], torch.tanh(y[:, 2:3])

    net = Heads()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer = RecordingBuffer(50)
    np.random.seed(123)
    train._epoch_train(Cfg(), net, optimizer, buffer, "cpu")
    assert len(buffer.rows) == 8
    assert max(buffer.rows) >= 8, buffer.rows


def test_collect_games_drops_stale_generation_envelopes():
    import train
    from config import Config

    q = queue.Queue()
    q.put({"kind": "game", "examples": [("old", "p", 0.0)], "generation": 1})
    q.put({"kind": "game", "examples": [("new", "p", 0.0)], "generation": 2})
    cfg = Config()
    cfg.result_timeout_seconds = 0.1
    examples = train._collect_games(cfg, q, [], games_needed=1, expected_generation=2)
    assert examples == [("new", "p", 0.0)]


def test_dirichlet_noise_obeys_numpy_seed():
    import chess
    from config import Config
    from mcts import MCTS, Node

    cfg = Config()
    cfg.dirichlet_epsilon = 1.0
    node1, node2 = Node(), Node()
    moves = list(chess.Board().legal_moves)[:4]
    for move in moves:
        node1.children[move] = Node(parent=node1, move=move, prior=0.25)
        node2.children[move] = Node(parent=node2, move=move, prior=0.25)
    mcts = MCTS.__new__(MCTS)
    mcts.cfg = cfg
    np.random.seed(77)
    mcts._apply_dirichlet_noise(node1)
    np.random.seed(77)
    mcts._apply_dirichlet_noise(node2)
    assert [c.P for c in node1.children.values()] == [c.P for c in node2.children.values()]


def test_mid_iteration_failure_does_not_checkpoint_partial_iteration(tmp_path, monkeypatch):
    import train
    from config import Config

    cfg = Config()
    cfg.device = "cpu"
    cfg.checkpoint_dir = str(tmp_path)
    cfg.num_res_blocks = 1
    cfg.num_filters = 4
    cfg.num_iterations = 1
    cfg.games_per_iteration = 1

    def fail_during_selfplay(net, cfg_):
        raise RuntimeError("injected mid-iteration failure")

    monkeypatch.setattr(train, "play_game", fail_during_selfplay)
    with pytest.raises(RuntimeError, match="mid-iteration"):
        train.run(cfg=cfg, resume=False)

    assert not (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "checkpoint_meta.json").exists()


def test_evaluation_summary_does_not_double_count_mirrored_records():
    import evaluate
    from config import Config

    cfg = Config()
    cfg.device = "cpu"
    cfg.max_game_length = 10
    out = evaluate.evaluate(
        cfg, seed=3, num_games=2, tactics_sims=1,
        players=("random", "greedy"), load_best=False,
    )
    assert len(out["results"]) == 2  # mirrored presentation records
    assert out["summary"]["total_games"] == 2  # only two games were played
    assert "score_rate" in out["summary"]


def test_best_weights_are_saved_via_atomic_replace(tmp_path):
    import train
    from config import Config
    from model import ChessNet

    cfg = Config()
    cfg.device = "cpu"
    cfg.checkpoint_dir = str(tmp_path)
    net = ChessNet(cfg)
    path = train._save_best_atomic(cfg, net)
    assert path == str(tmp_path / "best.pt")
    assert (tmp_path / "best.pt").exists()
    assert not (tmp_path / "best.pt.tmp").exists()


def test_viewer_terminal_detection_matches_claim_draw_search():
    import chess
    import serve
    from config import Config
    from model import ChessNet

    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    assert not board.is_repetition(3)
    assert board.can_claim_threefold_repetition()

    cfg = Config(); cfg.device = "cpu"; cfg.num_res_blocks = 1; cfg.num_filters = 4
    game = serve.Game(cfg, ChessNet(cfg).eval(), "test")
    game.board = board
    reason, text = game._terminal()
    assert reason == "repetition"
    assert text.startswith("Draw")
