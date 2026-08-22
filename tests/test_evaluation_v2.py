"""Sprint C + D test suite: fixed evaluation (evaluate.py) and the
integration-ready /move API + hot reload (serve.py).

Strict-TDD: this file is written FIRST; every assertion below targets
behaviour that does not exist yet. Run with:

    .venv/bin/python tests/test_evaluation_v2.py

Style matches the repo's existing assert-script tests
(test_checkpoint_helpers.py): plain asserts, exit code = pass/fail.
"""
import io
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import torch

from config import Config

PASSED = 0
FAILED = 0


def check(name, fn):
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"  ok  {name}")
    except Exception as e:  # noqa: BLE001
        FAILED += 1
        print(f"FAIL  {name}  ->  {type(e).__name__}: {e}")


def eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg or 'eq'}: got {a!r}, want {b!r}")


# --------------------------------------------------------------------------- #
#  Tiny fast config so network+MCTS baselines stay cheap on CPU               #
# --------------------------------------------------------------------------- #

class FastConfig(Config):
    num_res_blocks = 1
    num_filters = 16
    batch_size = 8
    num_simulations = 8
    c_puct = 1.25
    max_game_length = 60


def make_fast_cfg(**kw):
    cfg = FastConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
#  evaluate.py                                                                #
# --------------------------------------------------------------------------- #

def test_evaluate_module_imports():
    import evaluate  # noqa: F401
    assert hasattr(evaluate, "evaluate"), "evaluate.evaluate missing"
    assert hasattr(evaluate, "play_game"), "evaluate.play_game missing"
    assert hasattr(evaluate, "play_match"), "evaluate.play_match missing"
    assert hasattr(evaluate, "wilson_interval"), "evaluate.wilson_interval missing"


def test_wilson_interval():
    import evaluate
    lo, hi = evaluate.wilson_interval(10, 20)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0, f"CI {lo, hi} must bracket p=0.5"
    lo0, hi0 = evaluate.wilson_interval(0, 20)
    assert lo0 == 0.0, "zero-success CI lower bound must be 0"
    assert hi0 > 0.0, "zero-success CI upper bound must be > 0"
    lon, hin = evaluate.wilson_interval(20, 20)
    assert hin == 1.0, "full-success CI upper bound must be 1"
    assert lon < 1.0
    lo4, hi4 = evaluate.wilson_interval(4, 20)
    assert lo4 < lo, "fewer successes -> lower CI bound shifts down"


def test_wilson_interval_half_points():
    # draws count 0.5: a half-point success must be accepted by the formula
    import evaluate
    lo, hi = evaluate.wilson_interval(10.5, 20)
    assert 0.0 <= lo <= 0.525 <= hi <= 1.0, f"half-point CI {lo, hi}"


def test_random_player_legal_and_deterministic():
    import evaluate
    cfg = make_fast_cfg()
    board = chess.Board()
    # NOTE: fresh players every round.  The original version called
    # `p1.move(board)` once BEFORE the loop, which advanced p1's RNG one
    # draw ahead of its same-seed twin p2 — so the comparison compared
    # p1's *second* draw against p2's *first* and could never pass.
    for _ in range(30):
        p1 = evaluate.RandomPlayer(seed=7)
        p2 = evaluate.RandomPlayer(seed=7)
        p3 = evaluate.RandomPlayer(seed=8)
        ms = list(board.legal_moves)
        m1, m2, m3 = p1.move(board), p2.move(board), p3.move(board)
        assert m1 in ms and m2 in ms and m3 in ms, "random player must return a legal move"
        # same seed -> same move on the same position
        eq(m1, m2, "same seed must reproduce the same move")
        board.push(m1)
        ms = list(board.legal_moves)
        if not ms:
            break


def test_greedy_player_takes_hanging_queen():
    import evaluate
    p = evaluate.GreedyPlayer()
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1")
    mv = p.move(board)
    eq(mv.uci(), "f6e4", "one-ply material-greedy must capture the undefended queen")


def test_greedy_player_deterministic():
    import evaluate
    a, b = evaluate.GreedyPlayer(), evaluate.GreedyPlayer()
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1")
    eq(a.move(board), b.move(board), "greedy is fully deterministic")


def test_play_game_draw_scores_half():
    import evaluate
    cfg = make_fast_cfg()
    # K vs K: game is over before any move -> draw -> 0.5 from White's view
    board = chess.Board("8/8/8/8/8/8/8/k6K w - - 0 1")
    # play_game starts from the initial position, so drive it through the
    # evaluate-internal helper instead: use a player stub that returns None
    # and a starting board override.
    score, info = evaluate.play_game(
        evaluate.RandomPlayer(seed=1), evaluate.RandomPlayer(seed=2),
        cfg, start_fen="8/8/8/8/8/8/8/k6K w - - 0 1",
    )
    eq(score, 0.5, "K vs K must score 0.5 (draw)")
    assert info.get("result") == "draw", info


def test_play_game_checkmate_scores_one_or_zero():
    import evaluate
    cfg = make_fast_cfg()
    # Scholar's mate: white plays Qxf7# on move 4. Use a scripted stub player.
    seq = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    class Scripted:
        def __init__(self, moves): self.moves = list(moves)
        def move(self, board):
            return chess.Move.from_uci(self.moves.pop(0)) if self.moves else None
    score, info = evaluate.play_game(Scripted(seq[0::2]), Scripted(seq[1::2]), cfg)
    eq(score, 1.0, "white scholar's mate must score 1.0")
    eq(info.get("result"), "white", info)


def test_play_match_pairs_colors():
    import evaluate
    cfg = make_fast_cfg()
    seen = []

    class Stub:
        def __init__(self, name): self.name = name
        def move(self, board):
            return next(iter(board.legal_moves))

    orig = evaluate.play_game
    def spy(w, b, cfg_, **kw):
        seen.append((w.name, b.name))
        return 0.5, {"result": "draw", "plies": 0}
    evaluate.play_game = spy
    try:
        res = evaluate.play_match(Stub("A"), Stub("B"), cfg, num_games=4)
    finally:
        evaluate.play_game = orig
    eq(len(seen), 4, "one game per colour pair slot")
    eq(seen.count(("A", "B")), 2, "A must play White twice")
    eq(seen.count(("B", "A")), 2, "B must play White twice")
    # draws -> 0.5 each
    eq(res["score_a"], 2.0, "4 draws * 0.5")
    eq(res["score_b"], 2.0, "4 draws * 0.5")


def test_evaluate_json_schema_and_reproducibility():
    import evaluate
    cfg = make_fast_cfg()
    out1 = evaluate.evaluate(cfg, seed=1234, num_games=2, tactics_sims=2,
                             players=("random", "greedy"))
    out2 = evaluate.evaluate(cfg, seed=1234, num_games=2, tactics_sims=2,
                             players=("random", "greedy"))
    eq(out1, out2, "same seed must reproduce identical JSON output")
    for key in ("seed", "config", "results", "tactics", "summary"):
        assert key in out1, f"missing key {key}"
    json.dumps(out1)  # serialisable
    r0 = out1["results"][0]
    for key in ("white", "black", "games", "score_white", "score_black",
                "ci_white", "ci_black"):
        assert key in r0, f"result missing {key}"
    assert out1["seed"] == 1234
    # paired colours: every pairing appears twice, one per colour
    pairs = [(r["white"], r["black"]) for r in out1["results"]]
    for w, b in pairs:
        assert (b, w) in pairs, "paired-colour games must be symmetric"


def test_net_player_deterministic():
    import evaluate
    cfg = make_fast_cfg()
    import model
    net1 = model.ChessNet(cfg).eval()
    net2 = model.ChessNet(cfg).eval()
    net1.load_state_dict(net2.state_dict())  # identical weights
    p1 = evaluate.NetPlayer(net1, cfg, sims=6, seed=99)
    p2 = evaluate.NetPlayer(net2, cfg, sims=6, seed=99)
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 0 3")
    with evaluate.seeded_context(99):
        m1 = p1.move(board)
    with evaluate.seeded_context(99):
        m2 = p2.move(board)
    eq(m1, m2, "seeded net player must reproduce its move")
    assert m1 in list(board.legal_moves)


def test_tactical_positions_are_sane():
    import evaluate
    # every mate-in-one position really has a mating move
    for name, fen in evaluate.MATE_IN_ONE:
        b = chess.Board(fen)
        assert not b.is_game_over(), f"{name}: position must be live"
        mating = [m for m in b.legal_moves
                  if (lambda bb: bb.is_checkmate())(_push(b, m))]
        assert mating, f"{name}: no mate-in-one found in {fen}"
    # every avoid-mate position: side to move not in check, a blunder move
    # that allows mate-in-one exists, and at least one safe move exists
    for name, fen in evaluate.AVOID_MATE:
        b = chess.Board(fen)
        assert not b.is_check(), f"{name}: side to move must not be in check"
        blunders, safe = [], []
        for m in b.legal_moves:
            if _allows_mate_in_one(b, m):
                blunders.append(m)
            else:
                safe.append(m)
        assert blunders, f"{name}: no blunder move found"
        assert safe, f"{name}: no safe move found"


def test_tactics_suite_deterministic_and_scored():
    import evaluate
    cfg = make_fast_cfg()
    r1 = evaluate.run_tactics(evaluate.RandomPlayer(seed=5), cfg, seed=5)
    r2 = evaluate.run_tactics(evaluate.RandomPlayer(seed=5), cfg, seed=5)
    eq(r1, r2, "tactics run must be deterministic for a fixed seed")
    assert {"mate_in_one", "avoid_mate"} <= set(r1)
    for grp in ("mate_in_one", "avoid_mate"):
        for entry in r1[grp]:
            for key in ("name", "fen", "move", "pass"):
                assert key in entry, f"tactics entry missing {key}"


def _push(board, move):
    bb = board.copy()
    bb.push(move)
    return bb


def _allows_mate_in_one(board, move):
    bb = board.copy()
    bb.push(move)
    return any(_push(bb, m).is_checkmate() for m in bb.legal_moves)


# --------------------------------------------------------------------------- #
#  serve.py                                                                   #
# --------------------------------------------------------------------------- #

def test_serve_module_imports():
    import serve  # noqa: F401
    assert hasattr(serve, "compute_move"), "serve.compute_move missing"
    assert hasattr(serve, "bind_host_port"), "serve.bind_host_port missing"
    assert hasattr(serve, "reload_if_newer"), "serve.reload_if_newer missing"
    assert hasattr(serve, "infer_generation"), "serve.infer_generation missing"


def test_bind_host_port_defaults_and_env():
    import serve
    saved = {k: os.environ.pop(k, None) for k in ("CHESS_SERVE_HOST", "CHESS_SERVE_PORT")}
    try:
        host, port = serve.bind_host_port()
        eq(host, "127.0.0.1", "default bind host must be localhost")
        eq(port, 8790, "default port must be 8790")
        os.environ["CHESS_SERVE_HOST"] = "0.0.0.0"
        os.environ["CHESS_SERVE_PORT"] = "9999"
        host, port = serve.bind_host_port()
        eq(host, "0.0.0.0", "env host override")
        eq(port, 9999, "env port override")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_compute_move_returns_legal_move_and_meta():
    import serve
    cfg = make_fast_cfg(sims=6)
    import model
    net = model.ChessNet(cfg).eval()
    resp = serve.compute_move(net, cfg, chess.Board().fen(), sims=6, seed=42)
    for key in ("move", "value", "top_moves", "sims", "time_ms", "model"):
        assert key in resp, f"response missing {key}"
    mv = chess.Move.from_uci(resp["move"])
    b = chess.Board()
    assert mv in list(b.legal_moves), "returned move must be legal"
    assert -1.0 <= resp["value"] <= 1.0, "value must be in [-1, 1]"
    assert isinstance(resp["top_moves"], list) and resp["top_moves"]
    for tm in resp["top_moves"]:
        assert "uci" in tm and "visits" in tm, "top move entry incomplete"
        assert chess.Move.from_uci(tm["uci"]) in list(b.legal_moves)
    assert isinstance(resp["model"], dict)
    assert "source" in resp["model"], "model source required"
    assert "generation" in resp["model"], "model generation required"


def test_compute_move_rejects_bad_input():
    import serve
    cfg = make_fast_cfg(sims=4)
    import model
    net = model.ChessNet(cfg).eval()
    for bad in ("not a fen at all", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"):
        try:
            serve.compute_move(net, cfg, bad, sims=4, seed=1)
        except ValueError:
            continue
        raise AssertionError(f"compute_move must reject {bad!r}")
    # game-over FEN: black to move is mated (Qg7#); the original test data
    # (same board with "w" to move) was NOT game over — 26 legal moves — so
    # compute_move rightly refused to reject it.
    try:
        serve.compute_move(net, cfg, "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", sims=4, seed=1)
    except ValueError as e:
        assert "over" in str(e) or "legal" in str(e) or "mate" in str(e)
    else:
        raise AssertionError("compute_move must reject a game-over FEN")
    # sims bounds
    try:
        serve.compute_move(net, cfg, chess.Board().fen(), sims=0, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("sims=0 must be rejected")
    try:
        serve.compute_move(net, cfg, chess.Board().fen(), sims=99999, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("huge sims must be rejected")


def test_infer_generation_from_checkpoints():
    import serve
    cfg = make_fast_cfg()
    tmp = tempfile.mkdtemp(prefix="gen-test-")
    cfg.checkpoint_dir = tmp
    try:
        eq(serve.infer_generation(cfg, "best.pt"), None, "no checkpoints -> None")
        with open(os.path.join(tmp, "ckpt-iter0042-20260101-000000.pt"), "w") as f:
            f.write("x")
        with open(os.path.join(tmp, "ckpt-iter0037-20260101-000000.pt"), "w") as f:
            f.write("x")
        eq(serve.infer_generation(cfg, "best.pt"), 42, "max ckpt-iterNNNN wins")
    finally:
        import shutil
        shutil.rmtree(tmp)


def test_reload_if_newer_swaps_net_once():
    import serve
    cfg = make_fast_cfg()
    tmp = tempfile.mkdtemp(prefix="reload-test-")
    cfg.checkpoint_dir = tmp
    best = os.path.join(tmp, "best.pt")
    import model
    net1 = model.ChessNet(cfg).eval()
    torch.save(net1.state_dict(), best)
    try:
        game = serve.Game(cfg, net1, "best.pt")
        old_net = game.net
        eq(serve.reload_if_newer(game), False, "unchanged file -> no reload")
        # rewrite the file with new mtime + different weights
        time.sleep(0.05)
        net2 = model.ChessNet(cfg).eval()
        with torch.no_grad():
            for p in net2.parameters():
                p.add_(1.0)
        torch.save(net2.state_dict(), best)
        os.utime(best, (os.path.getmtime(best) + 2,) * 2)
        eq(serve.reload_if_newer(game), True, "newer file -> reload")
        assert game.net is not old_net, "net object must be replaced"
        assert game.trained_from == "best.pt"
        eq(serve.reload_if_newer(game), False, "no second reload for same file")
    finally:
        import shutil
        shutil.rmtree(tmp)


def test_explicit_reload_flag():
    import serve
    cfg = make_fast_cfg()
    tmp = tempfile.mkdtemp(prefix="reload-flag-")
    cfg.checkpoint_dir = tmp
    import model
    net1 = model.ChessNet(cfg).eval()
    torch.save(net1.state_dict(), os.path.join(tmp, "best.pt"))
    try:
        game = serve.Game(cfg, net1, "best.pt")
        eq(serve.reload_if_newer(game, force=True), True,
           "explicit reload must load even when the file is unchanged")
        eq(game.trained_from, "best.pt")
    finally:
        import shutil
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
#  Runner                                                                     #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import time
    t0 = time.time()
    check("evaluate module imports", test_evaluate_module_imports)
    check("wilson interval", test_wilson_interval)
    check("wilson interval half points", test_wilson_interval_half_points)
    check("random player legal + deterministic", test_random_player_legal_and_deterministic)
    check("greedy takes hanging queen", test_greedy_player_takes_hanging_queen)
    check("greedy deterministic", test_greedy_player_deterministic)
    check("draw scores 0.5", test_play_game_draw_scores_half)
    check("checkmate scores 1.0", test_play_game_checkmate_scores_one_or_zero)
    check("play_match pairs colours", test_play_match_pairs_colors)
    check("evaluate JSON schema + reproducibility", test_evaluate_json_schema_and_reproducibility)
    check("net player deterministic", test_net_player_deterministic)
    check("tactical positions sane", test_tactical_positions_are_sane)
    check("tactics suite deterministic + scored", test_tactics_suite_deterministic_and_scored)
    check("serve module imports", test_serve_module_imports)
    check("bind host/port defaults + env", test_bind_host_port_defaults_and_env)
    check("compute_move legal + meta", test_compute_move_returns_legal_move_and_meta)
    check("compute_move rejects bad input", test_compute_move_rejects_bad_input)
    check("infer generation", test_infer_generation_from_checkpoints)
    check("reload_if_newer swaps once", test_reload_if_newer_swaps_net_once)
    check("explicit reload flag", test_explicit_reload_flag)
    print(f"\n{FAILED} FAILED / {PASSED} passed  ({time.time() - t0:.1f}s)")
    sys.exit(1 if FAILED else 0)
