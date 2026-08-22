import os, sys, cProfile, pstats, io
import chess, torch
from config import Config
from model import ChessNet
from mcts import MCTS

cfg = Config()
net = ChessNet(cfg).to(cfg.device); net.eval()
best = os.path.join(cfg.checkpoint_dir, "best.pt")
if os.path.exists(best): net.load_state_dict(torch.load(best, map_location=cfg.device))
board = chess.Board("r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8")
mcts = MCTS(net, cfg)
mcts.search(board, temperature=1.0, num_sims=10)  # warmup

pr = cProfile.Profile()
pr.enable()
for _ in range(3):
    mcts.search(board, temperature=1.0, num_sims=100)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
print(s.getvalue())
