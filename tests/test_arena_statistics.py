"""Arena promotion gated on a confidence bound, not a point estimate.

The live run's arena history is the motivation: acceptances at 0.75 and a
rejection at 0.475, four promotions in 300 iterations, 13 draws in a 20-game
match.  At that sample size a 20-game score carries a 95% interval roughly
+/- 0.2 wide, so a gate on the point estimate promotes noise about as often as
it promotes strength.

Promotion now requires the LOWER bound of the interval to clear the threshold.
A practical consequence is pinned below: a 20-game arena can only ever confirm
LARGE improvements — a candidate whose true score is 0.60 needs ~176 games to
be promotable at threshold 0.55.

Run:  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_arena_statistics.py -q
"""
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config   # noqa: E402
import stats                # noqa: E402


# --------------------------------------------------------------------------- #
#  Wilson interval                                                            #
# --------------------------------------------------------------------------- #

def test_wilson_matches_a_known_reference():
    """n=100, k=60, 95% -> [0.50200, 0.69060].

    This is the UNCORRECTED Wilson score interval, computed by hand:
        denom  = 1 + z^2/n              = 1.038416
        centre = (p + z^2/2n) / denom   = 0.596301
        margin = (z/denom)*sqrt(p(1-p)/n + z^2/4n^2) = 0.094296
    (The continuity-corrected variant gives the slightly wider
    [0.4979, 0.6952]; this module deliberately uses the uncorrected form,
    and the promotion gate's conservatism comes from the doubled-trials
    transform instead.)
    """
    ci = stats.wilson_interval(60, 100, 0.95)
    assert ci["low"] == pytest.approx(0.502003, abs=1e-5)
    assert ci["high"] == pytest.approx(0.690599, abs=1e-5)


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    """A clean sweep must not produce a degenerate zero-width interval."""
    ci = stats.wilson_interval(20, 20, 0.95)
    assert 0.0 < ci["low"] < 1.0
    assert ci["high"] == 1.0
    zero = stats.wilson_interval(0, 20, 0.95)
    assert zero["low"] == 0.0
    assert 0.0 < zero["high"] < 1.0


def test_interval_narrows_as_games_grow():
    widths = [stats.wilson_interval(int(0.6 * n), n, 0.95)["high"]
              - stats.wilson_interval(int(0.6 * n), n, 0.95)["low"]
              for n in (20, 100, 1000)]
    assert widths[0] > widths[1] > widths[2]


def test_zero_trials_is_maximally_uncertain_not_an_error():
    ci = stats.wilson_interval(0, 0, 0.95)
    assert ci["low"] == 0.0 and ci["high"] == 1.0


@pytest.mark.parametrize("conf", [0.80, 0.90, 0.95, 0.99])
def test_higher_confidence_widens_the_interval(conf):
    base = stats.wilson_interval(60, 100, 0.80)
    ci = stats.wilson_interval(60, 100, conf)
    assert ci["low"] <= base["low"] + 1e-12


def test_untabulated_confidence_still_works():
    ci = stats.wilson_interval(60, 100, 0.975)
    assert 0.0 < ci["low"] < 0.6 < ci["high"] < 1.0


# --------------------------------------------------------------------------- #
#  match scores with draws                                                    #
# --------------------------------------------------------------------------- #

def test_score_counts_draws_as_half_a_point():
    ci = stats.score_interval(2, 1, 1)
    assert ci["score"] == pytest.approx(0.625)


def test_all_draws_scores_exactly_half():
    ci = stats.score_interval(0, 20, 0)
    assert ci["score"] == pytest.approx(0.5)
    assert ci["std_error"] == pytest.approx(0.0)


def test_doubled_trials_lower_bound_is_conservative_versus_empirical():
    """The gate reads the LOWER bound, and there it must not overstate.

    The doubled-trials transform treats a draw-heavy match as if it had the
    variance of coin flips, which it does not: for 10W/10D/0L the transform
    implies a per-game standard error of ~0.068 against an empirical ~0.056.
    So the lower bound sits BELOW the empirical one and the gate errs toward
    not promoting.  (Wilson is asymmetric — it shrinks toward 0.5 — so the
    upper bound is not correspondingly wider; only the lower bound is the
    gate, so only the lower bound is asserted.)
    """
    ci = stats.score_interval(10, 10, 0, 0.95)
    assert ci["low"] < ci["emp_low"], "gate must not be looser than the data"
    assert ci["std_error"] == pytest.approx(0.0559, abs=1e-4)


def test_empty_match_is_maximally_uncertain():
    ci = stats.score_interval(0, 0, 0)
    assert ci["games"] == 0 and ci["low"] == 0.0 and ci["high"] == 1.0


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError):
        stats.score_interval(-1, 0, 0)


# --------------------------------------------------------------------------- #
#  the promotion gate                                                         #
# --------------------------------------------------------------------------- #

def test_live_run_acceptance_still_promotes():
    """iter 2000: 10W 10D 0L, score 0.75 — a genuinely large margin."""
    d = stats.promotion_decision(10, 10, 0, threshold=0.55)
    assert d["score"] == pytest.approx(0.75)
    assert d["low"] > 0.55
    assert d["accepted"] is True


def test_live_run_rejection_still_rejects():
    """iter 2300: 3W 13D 4L, score 0.475."""
    d = stats.promotion_decision(3, 13, 4, threshold=0.55)
    assert d["accepted"] is False


def test_marginal_result_passes_the_point_estimate_but_fails_the_bound():
    """The case the old gate got wrong: 0.60 on 20 games is not evidence."""
    d = stats.promotion_decision(6, 12, 2, threshold=0.55)
    assert d["score"] == pytest.approx(0.60)
    assert d["point_pass"] is True
    assert d["lower_bound_pass"] is False
    assert d["accepted"] is False


def test_point_estimate_mode_can_be_restored():
    d = stats.promotion_decision(6, 12, 2, threshold=0.55,
                                 require_lower_bound=False)
    assert d["accepted"] is True


def test_the_same_margin_promotes_once_there_are_enough_games():
    """0.60 is promotable — it just needs the games to prove it."""
    small = stats.promotion_decision(6, 12, 2, threshold=0.55)
    big = stats.promotion_decision(120, 240, 40, threshold=0.55)
    assert big["score"] == pytest.approx(small["score"])
    assert small["accepted"] is False
    assert big["accepted"] is True


def test_a_draw_heavy_match_at_exactly_threshold_is_not_promoted():
    d = stats.promotion_decision(0, 20, 0, threshold=0.5)
    assert d["score"] == pytest.approx(0.5)
    assert d["accepted"] is False, "a dead-even match is not an improvement"


# --------------------------------------------------------------------------- #
#  sample-size guidance                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("true_score,at_most", [(0.75, 20), (0.65, 60), (0.60, 200)])
def test_games_needed_is_reported_and_monotone(true_score, at_most):
    n = stats.games_needed_for(0.55, true_score, 0.95)
    assert n <= at_most
    assert n % 2 == 0


def test_games_needed_is_unbounded_at_or_below_threshold():
    assert stats.games_needed_for(0.55, 0.55, 0.95, max_games=500) == 500
    assert stats.games_needed_for(0.55, 0.50, 0.95, max_games=500) == 500


def test_twenty_games_cannot_confirm_a_small_improvement():
    """Pins the practical limit of the default arena size."""
    assert stats.games_needed_for(0.55, 0.60, 0.95) > 20


# --------------------------------------------------------------------------- #
#  config surface                                                             #
# --------------------------------------------------------------------------- #

def test_config_defaults_enable_the_bound_gate():
    cfg = Config()
    assert cfg.arena_require_lower_bound is True
    assert cfg.arena_confidence == 0.95
    assert 0.5 <= cfg.arena_accept_threshold < 1.0
