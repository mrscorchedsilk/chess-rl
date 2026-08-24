"""Confidence intervals for head-to-head match results.

A 20-game arena that returns 0.475 one iteration and 0.75 the next is, at that
sample size, largely reporting noise: the 95% interval on a 20-game score is
roughly +/- 0.2 wide.  Promoting on the point estimate alone therefore
promotes noise about as often as it promotes strength, which is exactly the
behaviour the run's arena history shows.

The gate here is the LOWER confidence bound: a candidate is promoted only when
the interval says it is above the threshold, not merely when the point
estimate is.

Draws
-----
A match score ``(W + 0.5 D) / N`` is not a binomial proportion — outcomes take
three values.  The interval uses the standard "doubled trials" transform:
``2N`` trials with ``2W + D`` successes.  That is deliberately CONSERVATIVE
for a draw-heavy match.  With 10W/10D/0L the transform gives a per-game
standard error of 0.068 while the empirical standard error of the observed
{1, 0.5} outcomes is 0.056, so the interval is wider than the data strictly
requires and the gate errs toward not promoting.  ``score_interval`` also
returns the empirical-variance interval for information.
"""

from __future__ import annotations

import math
from typing import Dict

# Two-sided normal quantiles for the confidence levels worth offering.  Kept
# as a table so this module needs no scipy.
_Z = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.98: 2.3263478740408408,
    0.99: 2.5758293035489004,
}


def z_for(confidence: float) -> float:
    """Two-sided z for a confidence level; exact for the tabulated values."""
    conf = float(confidence)
    if conf in _Z:
        return _Z[conf]
    if not 0.0 < conf < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    # Acklam's inverse-normal approximation (|error| < 1.15e-9), so an
    # untabulated level still gets a usable z instead of an exception.
    p = 1.0 - (1.0 - conf) / 2.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def wilson_interval(successes: float, trials: float,
                    confidence: float = 0.95) -> Dict[str, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly at the extremes — a 20-0 sweep gets a lower bound below 1
    rather than a degenerate zero-width interval at 1.0.
    """
    n = float(trials)
    if n <= 0:
        return {"point": 0.0, "low": 0.0, "high": 1.0, "z": 0.0, "trials": 0.0}
    k = min(max(float(successes), 0.0), n)
    z = z_for(confidence)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {
        "point": p,
        "low": max(0.0, centre - margin),
        "high": min(1.0, centre + margin),
        "z": z,
        "trials": n,
    }


def score_interval(wins: int, draws: int, losses: int,
                   confidence: float = 0.95) -> Dict[str, float]:
    """Confidence interval for a match score ``(W + 0.5 D) / N``.

    Returns the Wilson interval on doubled trials (the gate), plus the
    empirical-variance normal interval (``emp_low`` / ``emp_high``) computed
    from the actual {1, 0.5, 0} outcomes, which is tighter when the match is
    draw-heavy.  ``games`` is the real game count, not the doubled one.
    """
    w, d, l = int(wins), int(draws), int(losses)
    if min(w, d, l) < 0:
        raise ValueError("wins/draws/losses must be non-negative")
    games = w + d + l
    if games == 0:
        return {"score": 0.0, "low": 0.0, "high": 1.0, "games": 0,
                "wins": 0, "draws": 0, "losses": 0,
                "emp_low": 0.0, "emp_high": 1.0, "std_error": 0.0,
                "confidence": float(confidence)}

    score = (w + 0.5 * d) / games
    wil = wilson_interval(2 * w + d, 2 * games, confidence)

    # Empirical variance of the per-game score.
    mean_sq = (w * 1.0 + d * 0.25) / games
    var = max(0.0, mean_sq - score * score)
    se = math.sqrt(var / games)
    z = z_for(confidence)
    return {
        "score": score,
        "low": wil["low"],
        "high": wil["high"],
        "games": games,
        "wins": w,
        "draws": d,
        "losses": l,
        "emp_low": max(0.0, score - z * se),
        "emp_high": min(1.0, score + z * se),
        "std_error": se,
        "confidence": float(confidence),
    }


def promotion_decision(wins: int, draws: int, losses: int, *,
                       threshold: float = 0.55, confidence: float = 0.95,
                       require_lower_bound: bool = True) -> Dict[str, object]:
    """Should this candidate be promoted?

    With ``require_lower_bound`` (the default) the candidate is promoted only
    when the LOWER bound of the interval clears ``threshold`` — i.e. the match
    is evidence of improvement, not merely consistent with it.  Setting it
    False restores point-estimate gating.
    """
    ci = score_interval(wins, draws, losses, confidence)
    thr = float(threshold)
    point_pass = ci["score"] >= thr
    bound_pass = ci["low"] >= thr
    accepted = bound_pass if require_lower_bound else point_pass
    return {
        **ci,
        "threshold": thr,
        "require_lower_bound": bool(require_lower_bound),
        "point_pass": point_pass,
        "lower_bound_pass": bound_pass,
        "accepted": accepted,
    }


def games_needed_for(threshold: float, expected_score: float,
                     confidence: float = 0.95, max_games: int = 100_000) -> int:
    """Smallest even game count whose lower bound clears ``threshold`` if the
    candidate really scores ``expected_score``.

    Answers the practical question directly: "how many games does this gate
    need to be able to promote at all?"  Returns ``max_games`` if unreachable.
    """
    if expected_score <= threshold:
        return max_games
    n = 2
    while n <= max_games:
        w = int(round(expected_score * n))
        ci = score_interval(w, 0, n - w, confidence)
        if ci["low"] >= threshold:
            return n
        n += 2
    return max_games
