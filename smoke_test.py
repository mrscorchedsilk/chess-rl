"""Smoke tests for the AlphaZero chess learner (option 1: CNN ResNet + MCTS).

Run from the project dir:  .venv/bin/python smoke_test.py
Sprint B: 104-plane 8-step-history encoding, 73-plane / 4672-action policy,
spatial Conv2d policy head with NHWC (from_square * 73 + plane) ordering.
"""
import sys
import chess
import numpy as np
import torch

from config import Config, get_device
import encoding
from model import ChessNet
from mcts import MCTS

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print("  ok  " + name)
    except Exception as e:
        FAIL.append(name)
        print("FAIL  " + name + "  ->  " + repr(e))


def main():
    torch.manual_seed(0)  # deterministic net init so mate-in-1 is reproducible
    print("== encoding ==")

    def test_planes():
        b = chess.Board()
        p = encoding.encode_board(b)
        assert p.shape == (104, 8, 8)
        assert p.dtype == np.float32
        # white king e1 (square 4 = rank0 file4) -> plane 5 of step 0
        assert p[5][0][4] == 1.0
        # black king e8 (square 60 = rank7 file4) -> plane 11 of step 0
        assert p[11][7][4] == 1.0
        # side-to-move plane all ones for white
        assert p[96].sum() == 64.0
        # history steps 1..7 zero-padded at game start
        assert p[12:96].sum() == 0.0
    check("104-plane encode (8-step history)", test_planes)

    def test_batch_equals_scalar():
        b = chess.Board()
        b.push_san("e4")
        for bb in (chess.Board(), b):
            assert np.array_equal(
                encoding.encode_batch([bb])[0], encoding.encode_board(bb)
            )
        assert encoding.encode_batch([chess.Board(), b]).shape == (2, 104, 8, 8)
    check("batch encoder == scalar encoder", test_batch_equals_scalar)

    def test_legal_mask():
        b = chess.Board()
        for san in ("e4", "e5", "Nf3"):
            b.push_san(san)
        mask = encoding.legal_moves_mask(b)
        for m in b.legal_moves:
            assert mask[encoding.move_to_index(m)] == 1.0
        distinct = len(set(encoding.move_to_index(m) for m in b.legal_moves))
        assert int(mask.sum()) == distinct
    check("legal-move mask matches python-chess", test_legal_mask)

    def test_index_roundtrip():
        b = chess.Board()
        b.push_san("e4")
        for m in b.legal_moves:
            m2 = encoding.index_to_move(b, encoding.move_to_index(m))
            assert m2.from_square == m.from_square and m2.to_square == m.to_square
    check("move <-> index roundtrip", test_index_roundtrip)

    def test_promotion():
        b = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        m = encoding.index_to_move(b, encoding.move_to_index(chess.Move.from_uci("a7a8")))
        assert m.promotion == chess.QUEEN
    check("promotion maps to queen", test_promotion)

    print("== model ==")
    cfg = Config()
    net = ChessNet(cfg).to(cfg.device)

    def test_forward():
        x = torch.randn(2, 104, 8, 8, device=cfg.device)
        p, v = net(x)
        assert tuple(p.shape) == (2, 4672), p.shape
        assert tuple(v.shape) == (2, 1), v.shape
        assert float(v.min().detach()) >= -1.0 and float(v.max().detach()) <= 1.0
    check("model forward shapes on " + cfg.device, test_forward)

    print("== mcts ==")
    mcts = MCTS(net, cfg)

    def test_legal_from_start():
        pi = mcts.search(
            chess.Board(), temperature=0.0, num_sims=20, add_root_noise=False
        )
        assert len(pi) == 1
        move = next(iter(pi))
        assert move in chess.Board().legal_moves
    check("mcts returns a legal move from start", test_legal_from_start)

    def test_mate_in_one():
        b = chess.Board("7k/6pp/8/8/8/8/8/R6K w - - 0 1")
        pi = mcts.search(b, temperature=0.0, num_sims=800, add_root_noise=False)
        move = max(pi, key=pi.get)
        assert move.uci() == "a1a8", move.uci()
    check("mcts finds mate-in-1 (a1a8#)", test_mate_in_one)

    print("== selfplay ==")
    from selfplay import play_game
    cfg_small = Config()
    cfg_small.num_simulations = 25

    def test_selfplay():
        ex = play_game(net, cfg_small)
        assert len(ex) > 0
        s, pi, z = ex[0]
        assert s.shape == (104, 8, 8)
        assert pi.shape == (4672,)
        assert abs(float(pi.sum()) - 1.0) < 1e-3
        assert z in (-1.0, 0.0, 1.0)
    check("selfplay produces valid training examples", test_selfplay)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
