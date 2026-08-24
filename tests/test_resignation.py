"""Resignation and draw adjudication, with calibration.

Most self-play compute goes into positions that are already decided — a weak
network shuffles to the 400-ply cap, and those plies carry no learning signal.
Resignation reclaims that compute.

It is also the most dangerous optimisation in the pipeline: a threshold that
is too aggressive labels games with a result that never happened, poisoning
the value target in a way no loss curve reveals.  The guard is the PLAYOUT
FRACTION — a share of games have resignation suppressed and are played to a
real finish, so the false-positive rate is measured rather than assumed.  The
native Actor refuses to resign at all with a zero playout fraction.

Note on the streak rule: completed searches alternate sides, so the streak is
counted PER SIDE.  A single shared counter resets every other ply and could
never reach 2 unless both players were simultaneously lost — which is what an
earlier version of this did.

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_resignation.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess_rl_native as native   # noqa: E402
from config import Config          # noqa: E402
import native_selfplay as ns       # noqa: E402


def white_is_lost(inputs, offsets, indices):
    """Consistent evaluation: White is lost, from each node's own viewpoint."""
    planes = ns.expand_planes(inputs)
    white_to_move = planes[:, 96, 0, 0] > 0.5
    values = np.where(white_to_move, -0.99, 0.99).astype(np.float32)
    return (np.zeros(len(indices), dtype=np.float32), values.reshape(-1, 1))


def balanced(inputs, offsets, indices):
    """Dead-level evaluation, for draw adjudication."""
    planes = ns.expand_planes(inputs)
    return (np.zeros(len(indices), dtype=np.float32),
            np.zeros((planes.shape[0], 1), dtype=np.float32))


def drive(actor, evaluate):
    while not actor.is_done():
        tokens, inputs, offsets, indices = actor.gather_leaves(2048, 8)
        if len(tokens) == 0:
            actor.advance()
            continue
        logits, values = evaluate(inputs, offsets, indices)
        actor.apply_evaluations(tokens, offsets, logits, values)
        actor.advance()
    return actor.finished_games()


def run_actor(evaluate=white_is_lost, games=24, seed=4242, **kw):
    actor = native.Actor(games=games, c_puct=1.25, virtual_loss=3.0,
                         num_simulations=12, temperature=1.0,
                         temperature_threshold=30, max_game_length=80,
                         seed=seed, num_threads=8, **kw)
    actor.set_teacher(0, 0)
    return drive(actor, evaluate)


RESIGN = dict(resign_threshold=-0.85, resign_consecutive=2,
              resign_playout_fraction=0.10)


# --------------------------------------------------------------------------- #
#  off by default                                                             #
# --------------------------------------------------------------------------- #

def test_disabled_by_default_in_config():
    assert Config.resign_enabled is False
    assert Config.draw_adjudication_enabled is False
    assert Config.resign_playout_fraction == 0.10


def test_actor_does_not_resign_unless_configured():
    games = run_actor()
    assert all(not g["resigned"] for g in games)
    assert all(g["termination"] != "resignation" for g in games)


# --------------------------------------------------------------------------- #
#  resignation shortens games                                                 #
# --------------------------------------------------------------------------- #

def test_resignation_ends_lost_games_early():
    off = run_actor()
    on = run_actor(**RESIGN)
    mean_off = sum(g["plies"] for g in off) / len(off)
    mean_on = sum(g["plies"] for g in on) / len(on)
    assert sum(1 for g in on if g["resigned"]) > 0
    assert mean_on < mean_off / 2, (mean_off, mean_on)


def test_resigned_games_are_labelled_as_a_loss_for_the_mover():
    games = run_actor(**RESIGN)
    resigned = [g for g in games if g["resigned"]]
    assert resigned
    for g in resigned:
        assert g["termination"] == "resignation"
        # every example carries the game result from its own mover's view
        zs = {float(z) for _, _, z in g["examples"]}
        assert zs <= {-1.0, 1.0}


def test_per_side_streak_is_required():
    """consecutive=1 and consecutive=2 must BOTH be able to fire.

    With a single shared counter the alternating sides reset it every ply and
    consecutive=2 could never trigger; this pins the per-side rule.
    """
    one = run_actor(resign_threshold=-0.85, resign_consecutive=1,
                    resign_playout_fraction=0.10)
    two = run_actor(resign_threshold=-0.85, resign_consecutive=2,
                    resign_playout_fraction=0.10)
    assert sum(1 for g in one if g["resigned"]) > 0
    assert sum(1 for g in two if g["resigned"]) > 0


def test_a_stricter_threshold_resigns_no_more_often():
    lenient = run_actor(resign_threshold=-0.50, resign_consecutive=2,
                        resign_playout_fraction=0.10)
    strict = run_actor(resign_threshold=-0.995, resign_consecutive=2,
                       resign_playout_fraction=0.10)
    assert sum(1 for g in strict if g["resigned"]) <= \
        sum(1 for g in lenient if g["resigned"])


# --------------------------------------------------------------------------- #
#  calibration                                                                #
# --------------------------------------------------------------------------- #

def test_a_playout_sample_is_reserved_and_never_resigns():
    games = run_actor(games=60, **RESIGN)
    playout = [g for g in games if g["playout"]]
    assert playout, "no calibration sample was reserved"
    assert all(not g["resigned"] for g in playout)


def test_playout_fraction_is_approximately_honoured():
    games = run_actor(games=200, resign_threshold=-0.85,
                      resign_consecutive=2, resign_playout_fraction=0.25)
    frac = sum(1 for g in games if g["playout"]) / len(games)
    assert 0.15 < frac < 0.35, frac


def test_false_resignations_are_detected_on_playout_games():
    """The evaluator here is systematically WRONG about the opening.

    Games it claims are lost actually run to the length cap as draws, so every
    suppressed resignation is a false positive — and the calibration must say
    so rather than reporting a clean sheet.
    """
    games = run_actor(games=60, **RESIGN)
    would = [g for g in games if g["would_have_resigned"]]
    assert would, "no playout game reached the resign condition"
    assert all(g["playout"] for g in would)
    assert any(g["false_resignation"] for g in would)


def test_false_flags_are_never_set_on_resigned_games():
    """A resigned game has no ground truth; claiming one would be a lie."""
    games = run_actor(games=60, **RESIGN)
    for g in games:
        if g["resigned"]:
            assert not g["false_resignation"]
            assert not g["would_have_resigned"]


def test_resignation_without_a_playout_fraction_is_refused():
    """Uncalibratable resignation must not be silently allowed."""
    with pytest.raises(ValueError, match="calibrated"):
        native.Actor(games=4, num_simulations=4, max_game_length=10, seed=1,
                     resign_threshold=-0.9, resign_playout_fraction=0.0)


@pytest.mark.parametrize("kw,match", [
    (dict(resign_threshold=0.5), "resign_threshold"),
    (dict(resign_threshold=-0.9, resign_consecutive=0), "resign_consecutive"),
    (dict(resign_threshold=-0.9, resign_playout_fraction=1.5), "playout"),
    (dict(draw_consecutive=0), "draw_consecutive"),
    (dict(draw_min_ply=-1), "draw_min_ply"),
])
def test_invalid_settings_are_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        native.Actor(games=2, num_simulations=2, max_game_length=8, seed=1, **kw)


# --------------------------------------------------------------------------- #
#  draw adjudication                                                          #
# --------------------------------------------------------------------------- #

def test_draw_adjudication_ends_dead_level_games():
    off = run_actor(evaluate=balanced)
    on = run_actor(evaluate=balanced, draw_threshold=0.01,
                   draw_consecutive=3, draw_min_ply=10,
                   resign_playout_fraction=0.10)
    assert sum(1 for g in on if g["adjudicated_draw"]) > 0
    mean_off = sum(g["plies"] for g in off) / len(off)
    mean_on = sum(g["plies"] for g in on) / len(on)
    assert mean_on < mean_off


def test_adjudicated_draws_are_scored_as_draws():
    games = run_actor(evaluate=balanced, draw_threshold=0.01,
                      draw_consecutive=3, draw_min_ply=10,
                      resign_playout_fraction=0.10)
    drawn = [g for g in games if g["adjudicated_draw"]]
    assert drawn
    for g in drawn:
        assert g["termination"] == "adjudicated_draw"
        assert all(float(z) == 0.0 for _, _, z in g["examples"])


def test_draw_adjudication_respects_the_minimum_ply():
    games = run_actor(evaluate=balanced, draw_threshold=0.01,
                      draw_consecutive=2, draw_min_ply=40,
                      resign_playout_fraction=0.10)
    for g in games:
        if g["adjudicated_draw"]:
            assert g["plies"] >= 40


# --------------------------------------------------------------------------- #
#  aggregation                                                                #
# --------------------------------------------------------------------------- #

def test_summary_reports_rates_and_distinguishes_no_evidence():
    games = run_actor(games=60, **RESIGN)
    summary = ns.summarise_terminations(games)
    assert summary["games"] == 60
    assert summary["resigned"] > 0
    assert 0.0 < summary["playout_fraction"] < 1.0
    assert summary["false_resignation_rate"] is not None
    assert 0.0 <= summary["false_resignation_rate"] <= 1.0
    assert "resignation" in summary["terminations"]


def test_no_evidence_reports_none_not_zero():
    """'no playout game hit the condition' is NOT 'zero false positives'."""
    summary = ns.summarise_terminations(run_actor(games=12))
    assert summary["false_resignation_rate"] is None
    assert summary["false_draw_rate"] is None


def test_empty_round_summarises_to_empty():
    assert ns.summarise_terminations([]) == {}


def test_selfplay_drivers_expose_the_summary():
    cfg = Config()
    cfg.num_simulations = 8
    cfg.max_game_length = 60
    cfg.telemetry_enabled = False
    cfg.resign_enabled = True
    cfg.resign_threshold = -0.85
    sp = ns.ShardedSelfPlay(cfg, white_is_lost, games=12, shards=2, seed=7)
    sp.run()
    assert sp.termination_stats["games"] == 12
    assert sp.termination_stats["resigned"] > 0

    serial = ns.NativeSelfPlay(cfg, white_is_lost, games=6, seed=7)
    serial.run()
    assert serial.termination_stats["games"] == 6
