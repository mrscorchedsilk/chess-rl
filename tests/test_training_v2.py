"""Sprint A vertical slice 1: compressed replay buffer (replay.py).

Only append / sample / state_dict-round-trip tests live here for now; the
train.py and parallel.py slices are added as separate vertical slices.
"""

import os
import queue
import sys
import threading

import chess
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import parallel  # noqa: E402
from config import Config  # noqa: E402
from encoding import encode_board, move_to_index, policy_to_vector  # noqa: E402
from mcts import MCTS  # noqa: E402
from model import ChessNet  # noqa: E402


def _synthetic_examples(n, policy_size=4096, planes=18, board=8, seed=0):
    """(state, pi, z) triples with realistic shapes; pi is sparse by construction."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        state = (rng.random((planes, board, board)) < 0.2).astype(np.float32)
        k = int(rng.integers(1, 12))
        idx = rng.choice(policy_size, size=k, replace=False)
        probs = rng.random(k)
        probs = probs / probs.sum()
        pi = np.zeros(policy_size, dtype=np.float32)
        pi[idx] = probs
        out.append((state, pi, float(rng.choice([-1.0, 0.0, 1.0]))))
    return out


def test_replay_add_and_len():
    from replay import ReplayBuffer
    buf = ReplayBuffer(64, 4096, 18, 8)
    assert len(buf) == 0
    for e in _synthetic_examples(5):
        buf.add(*e)
    assert len(buf) == 5


def test_replay_extend_real_selfplay_examples():
    from replay import ReplayBuffer
    cfg = Config()
    buf = ReplayBuffer(64, cfg.policy_size, cfg.num_input_planes, cfg.board_size)
    board = chess.Board()
    examples = []
    for _ in range(4):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = moves[0]
        state = encode_board(board)
        pi = policy_to_vector({move: 1.0})
        examples.append((state, pi, 1.0))
        board.push(move)
    assert len(examples) == 4
    buf.extend(examples)
    assert len(buf) == 4
    s, p, z = buf.sample_indices([0])
    assert s.shape == (1, cfg.num_input_planes, 8, 8) and s.dtype == torch.float32
    assert p.shape == (1, cfg.policy_size)
    assert np.array_equal(s[0].numpy(), examples[0][0]), "state round-trip lost bits"
    assert np.array_equal(p[0].numpy(), examples[0][1]), "policy round-trip lost mass"
    assert float(z[0, 0]) == 1.0


def test_replay_stores_state_compactly():
    from replay import ReplayBuffer
    buf = ReplayBuffer(100, 4096, 18, 8)
    ex = _synthetic_examples(10)
    buf.extend(ex)
    sd = buf.state_dict()
    # state: packed binary planes -> 144 bytes vs 4608 dense float32
    assert sd["positions"].shape == (10, 144)
    assert sd["positions"].dtype == np.uint8
    # policy: sparse (legal indices + probs) -> far below a dense 4096 float32
    dense_bytes = 10 * (18 * 8 * 8 * 4 + 4096 * 4 + 4)
    stored_bytes = sum(
        a.nbytes for a in (sd["positions"], sd["legal_idx"], sd["probs"], sd["z"])
    )
    assert stored_bytes < dense_bytes / 5, f"{stored_bytes} vs dense {dense_bytes}"


def test_replay_ring_eviction():
    from replay import ReplayBuffer
    buf = ReplayBuffer(4, 64, 3, 4)
    ex = _synthetic_examples(6, policy_size=64, planes=3, board=4, seed=3)
    buf.extend(ex)
    assert len(buf) == 4
    s, p, z = buf.sample_indices([0, 3])
    # ring eviction: the two oldest (ex[0], ex[1]) are gone; index 0 is the
    # oldest RETAINED example (ex[2]) and index 3 the newest (ex[5]).
    assert np.array_equal(s[0].numpy(), ex[2][0]), "oldest retained example"
    assert np.array_equal(s[1].numpy(), ex[5][0]), "newest example should survive"
    assert np.array_equal(p[0].numpy(), ex[2][1])


def test_replay_state_dict_roundtrip_deterministic():
    from replay import ReplayBuffer
    buf = ReplayBuffer(100, 4096, 18, 8)
    buf.extend(_synthetic_examples(25, seed=7))
    sd1 = buf.state_dict()
    buf2 = ReplayBuffer(100, 4096, 18, 8)
    buf2.load_state_dict(sd1)
    sd2 = buf2.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        if isinstance(sd1[k], np.ndarray):
            assert np.array_equal(sd1[k], sd2[k]), f"array mismatch in {k}"
        else:
            assert sd1[k] == sd2[k], f"scalar mismatch in {k}"
    rows = np.array([0, 3, 7, 11, 20])
    for a, b in zip(buf.sample_indices(rows), buf2.sample_indices(rows)):
        assert torch.equal(a, b), "sampled minibatch differs after round-trip"


def test_replay_sample_batch_shapes_and_zero():
    from replay import ReplayBuffer
    buf = ReplayBuffer(64, 4096, 18, 8)
    buf.extend(_synthetic_examples(8, seed=1))
    s, p, z = buf.sample_indices([0, 1, 2])
    assert s.shape == (3, 18, 8, 8) and s.dtype == torch.float32
    assert p.shape == (3, 4096) and p.dtype == torch.float32
    assert z.shape == (3, 1)
    assert torch.all((s >= 0) & (s <= 1))
    assert torch.all(p >= 0) and torch.all(p <= 1)


def test_replay_validation_errors():
    from replay import ReplayBuffer
    buf = ReplayBuffer(8, 4096, 18, 8)
    with pytest.raises(ValueError):
        buf.add(np.zeros((17, 8, 8), np.float32), np.zeros(4096, np.float32), 0.0)
    with pytest.raises(ValueError):
        buf.add(np.zeros((18, 8, 8), np.float32), np.zeros(100, np.float32), 0.0)
    with pytest.raises(ValueError):
        buf.load_state_dict({"capacity": 8, "policy_size": 4096,
                             "num_input_planes": 18, "board_size": 8})


# ==========================================================================
# parallel.py: sparse inference, client timeout, server error capture,
# worker liveness / error propagation
# ==========================================================================

def test_planes_to_board_inverts_encoding_over_random_games():
    from parallel import planes_to_board
    rng = np.random.default_rng(0)
    board = chess.Board()
    for step in range(150):
        dec = planes_to_board(encode_board(board))
        assert dec.piece_map() == board.piece_map(), f"piece mismatch at step {step}"
        assert dec.turn == board.turn
        assert dec.castling_rights == board.castling_rights
        assert dec.ep_square == board.ep_square
        assert set(dec.legal_moves) == set(board.legal_moves), f"legal mismatch at step {step}"
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))


def test_planes_to_board_specific_positions():
    from parallel import planes_to_board
    # castling rights both sides
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    b = chess.Board(fen)
    assert set(planes_to_board(encode_board(b)).legal_moves) == set(b.legal_moves)
    # en-passant available
    fen = "4k3/8/8/8/Pp6/8/8/4K3 b - b3 0 1"
    b = chess.Board(fen)
    assert set(planes_to_board(encode_board(b)).legal_moves) == set(b.legal_moves)
    # promotion setup
    fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    b = chess.Board(fen)
    assert set(planes_to_board(encode_board(b)).legal_moves) == set(b.legal_moves)


def test_inference_client_times_out_instead_of_hanging():
    client = parallel.InferenceClient(queue.Queue(), queue.Queue(),
                                      policy_size=64, timeout=0.1)
    x = torch.zeros(1, 18, 8, 8)
    result = {}

    def call():
        try:
            result["value"] = client(x)
        except Exception as exc:  # noqa: BLE001
            result["exc"] = exc

    t = threading.Thread(target=call)
    t.start()
    t.join(3)
    assert not t.is_alive(), "client blocked forever: no response timeout"
    assert isinstance(result.get("exc"), RuntimeError), "expected RuntimeError on timeout"


def test_inference_client_reconstructs_sparse_reply():
    req_q, resp_q = queue.Queue(), queue.Queue()
    client = parallel.InferenceClient(req_q, resp_q, policy_size=64, timeout=1.0)
    idx = np.array([1, 5, 9], dtype=np.int32)
    vals = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    resp_q.put(([(idx, vals)], np.array([[0.5]], dtype=np.float32), "sparse"))
    logits, values = client(torch.zeros(1, 18, 8, 8))
    dense = logits.numpy()[0]
    assert dense[1] == pytest.approx(0.1) and dense[5] == pytest.approx(0.2)
    assert dense[9] == pytest.approx(0.3)
    assert float(dense.sum()) == pytest.approx(0.6)
    assert float(values[0, 0]) == pytest.approx(0.5)


def test_inference_client_raises_on_server_error_reply():
    req_q, resp_q = queue.Queue(), queue.Queue()
    client = parallel.InferenceClient(req_q, resp_q, policy_size=64, timeout=1.0)
    resp_q.put((None, None, "error"))
    with pytest.raises(RuntimeError):
        client(torch.zeros(1, 18, 8, 8))


def test_inference_server_sparse_roundtrip_matches_dense():
    cfg = Config()
    cfg.num_res_blocks = 1
    cfg.num_filters = 4
    net = ChessNet(cfg).eval()
    req_q, resp_q = queue.Queue(), queue.Queue()
    server = parallel.InferenceServer(
        net, "cpu", [(req_q, resp_q)],
        max_batch=64, min_batch=1, wait_secs=0.001, sparse_response=True,
    )
    server.start()
    try:
        boards = [chess.Board() for _ in range(3)]
        xs = np.stack([encode_board(b) for b in boards])
        req_q.put(xs)
        payload, values, kind = resp_q.get(timeout=10)
        assert kind == "sparse", f"expected sparse reply, got {kind!r}"
        dense = np.zeros((len(boards), cfg.policy_size), dtype=np.float32)
        for i, (li, vals) in enumerate(payload):
            dense[i, li] = vals
        with torch.no_grad():
            logits_ref, _ = net(torch.from_numpy(xs))
        logits_ref = logits_ref.numpy()
        for i, b in enumerate(boards):
            mask = np.zeros(cfg.policy_size, dtype=bool)
            mask[[move_to_index(m) for m in b.legal_moves]] = True
            assert np.array_equal(dense[i][mask], logits_ref[i][mask]), \
                "sparse logits differ from dense at legal indices"
            assert not dense[i][~mask].any(), "illegal indices should be zero"
        assert values.shape == (3, 1)
    finally:
        server.stop()


def test_inference_server_dense_mode_fallback():
    cfg = Config()
    cfg.num_res_blocks = 1
    cfg.num_filters = 4
    net = ChessNet(cfg).eval()
    req_q, resp_q = queue.Queue(), queue.Queue()
    server = parallel.InferenceServer(
        net, "cpu", [(req_q, resp_q)],
        max_batch=64, min_batch=1, wait_secs=0.001, sparse_response=False,
    )
    server.start()
    try:
        xs = np.stack([encode_board(chess.Board())])
        req_q.put(xs)
        payload, values, kind = resp_q.get(timeout=10)
        assert kind == "dense"
        assert payload.shape == (1, cfg.policy_size)
    finally:
        server.stop()


def test_inference_server_captures_forward_exceptions():
    class _BrokenNet(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("boom in forward")

    req_q, resp_q = queue.Queue(), queue.Queue()
    server = parallel.InferenceServer(
        _BrokenNet(), "cpu", [(req_q, resp_q)],
        max_batch=64, min_batch=1, wait_secs=0.001, sparse_response=True,
    )
    server.start()
    try:
        req_q.put(np.zeros((1, 18, 8, 8), dtype=np.float32))
        payload, values, kind = resp_q.get(timeout=10)
        assert kind == "error"
        assert server.get_error() is not None
        assert "boom in forward" in server.get_error()
        assert not server.is_alive(), "server thread should have stopped after error"
    finally:
        server.stop()


def test_worker_loop_propagates_errors(monkeypatch):
    def boom(net, cfg):
        raise RuntimeError("worker boom")

    monkeypatch.setattr(parallel, "play_game", boom)
    req_q, resp_q, res_q = queue.Queue(), queue.Queue(), queue.Queue()
    stop = threading.Event()
    t = threading.Thread(
        target=parallel.worker_loop,
        args=(Config(), req_q, resp_q, res_q, 1, stop, 3),
    )
    t.start()
    t.join(5)
    msg = res_q.get(timeout=1)
    assert msg["kind"] == "error"
    assert msg["worker"] == 3
    assert "worker boom" in msg["traceback"]
    assert not t.is_alive()


def test_worker_loop_stops_cleanly(monkeypatch):
    stop = threading.Event()

    def fake_game(net, cfg):
        stop.set()
        return [("state", "pi", 0.0)]

    monkeypatch.setattr(parallel, "play_game", fake_game)
    req_q, resp_q, res_q = queue.Queue(), queue.Queue(), queue.Queue()
    t = threading.Thread(
        target=parallel.worker_loop,
        args=(Config(), req_q, resp_q, res_q, 1, stop, 0),
    )
    t.start()
    t.join(5)
    msg = res_q.get(timeout=1)
    assert msg["kind"] == "game"
    assert not t.is_alive(), "worker should exit when stop_event is set"


def test_sparse_inference_mcts_equivalent_to_dense():
    cfg = Config()
    cfg.device = "cpu"  # CPU equivalence test: server, client and reference all on CPU
    cfg.num_res_blocks = 1
    cfg.num_filters = 4
    cfg.num_simulations = 5
    cfg.batch_size = 4
    cfg.dirichlet_epsilon = 0.0  # kill noise: search becomes fully deterministic
    net = ChessNet(cfg).eval()
    req_q, resp_q = queue.Queue(), queue.Queue()
    server = parallel.InferenceServer(
        net, "cpu", [(req_q, resp_q)],
        max_batch=64, min_batch=1, wait_secs=0.001, sparse_response=True,
    )
    server.start()
    try:
        client = parallel.InferenceClient(req_q, resp_q,
                                          policy_size=cfg.policy_size, timeout=30)
        board = chess.Board()
        pi_sparse = MCTS(client, cfg).search(board, temperature=1.0, num_sims=5)
        pi_dense = MCTS(net, cfg).search(board.copy(), temperature=1.0, num_sims=5)
        assert set(pi_sparse.keys()) == set(pi_dense.keys())
        for move, prob in pi_dense.items():
            assert pi_sparse[move] == pytest.approx(prob, abs=1e-9), \
                f"sparse vs dense policy differ at {move}"
    finally:
        server.stop()
