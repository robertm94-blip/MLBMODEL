"""Dedicated totals prediction model for MLB games.

The moneyline model's total projection has systemic issues:
- +0.35 run over-projection bias
- Park factors compound too aggressively with other multipliers
- Correlation with actual totals is only 0.13

This dedicated model fixes these by:
1. Using additive adjustments instead of multiplicative (prevents compounding)
2. Recalibrated park factors specific to totals
3. Regressing all projections toward the league mean more heavily
4. Separate factor weights optimized for total runs, not win probability

Key insight: Totals in baseball are inherently noisy (std dev ~4.6 runs).
Even a perfect model can only reduce MAE to ~2.5 runs. The goal isn't
perfect prediction — it's identifying when the posted total is significantly
off from true expectation.
"""

from typing import Any

# Recalibrated park factors for totals (dampened from ML model)
# These are the ADDITIVE adjustment to runs per game, not multiplicative
# Derived from 2025 backtest venue bias analysis
TOTALS_PARK_ADJ = {
    # Hitter parks (add runs)
    19: +1.80,   # Coors Field
    2602: +0.60, # Great American Ball Park
    5325: +0.30, # Globe Life Field (retractable — dampened)
    17: +0.25,   # Wrigley Field
    3313: +0.25, # Yankee Stadium
    2681: +0.20, # Citizens Bank Park
    3: +0.15,    # Fenway Park
    2392: +0.10, # Minute Maid Park
    2: +0.10,    # Camden Yards

    # Neutral
    15: 0.0,     # Chase Field
    14: 0.0,     # Rogers Centre
    3309: 0.0,   # Nationals Park
    3312: 0.0,   # Target Field
    4705: -0.05, # Truist Park
    32: -0.05,   # American Family Field
    7: -0.05,    # Kauffman Stadium

    # Pitcher parks (subtract runs)
    2889: -0.10, # Busch Stadium
    5: -0.10,    # Progressive Field
    31: -0.10,   # PNC Park
    1: -0.10,    # Angel Stadium
    4: -0.10,    # Guaranteed Rate Field
    12: -0.15,   # Tropicana Field
    680: -0.20,  # T-Mobile Park
    22: -0.20,   # Dodger Stadium
    3289: -0.25, # Citi Field
    2680: -0.30, # Petco Park
    4169: -0.35, # loanDepot Park
    2395: -0.35, # Oracle Park
    10: -0.20,   # Oakland Coliseum
    2394: -0.15, # Comerica Park
}

# League average total runs per game.
# Calibrated 2026-04-27 via backtest_totals.py + live-slate verification:
# 2024-2025 raw actual mean was 8.84, but live projections built from
# lineup-specific wRC+ (which skews above team-season averages because active
# starters are above league avg) push the projected mean ~0.3 higher than
# this baseline. Anchoring at 8.50 brings the live slate mean (~8.6) in line
# with posted-market totals, which themselves shade ~0.3 below raw season actuals.
LEAGUE_AVG_TOTAL = 8.50

# Regression weight toward league mean
# Higher = more regression = more conservative (less overshoot)
REGRESSION_TO_MEAN = 0.55

# Multiplier on the offense + pitching adjustments. 1.0 = full scaling (legacy).
# Values < 1.0 dampen the contribution of team factors to the run-pool sum,
# countering the additive collision when both teams have above-average
# offenses or above-average pitching simultaneously.
# Calibrated against 2024-2025 actuals + live-slate verification on 2026-04-27:
# 0.55 cuts the additive scaling roughly in half, which collapses the
# typical above-average lineup pile-up from ~+0.6 raw runs to ~+0.3.
FACTOR_SCALE_DAMPENER = 0.55


def compute_totals_projection(
    away_off_factor: float,
    home_off_factor: float,
    away_pitching_factor: float,
    home_pitching_factor: float,
    venue_id: int | None = None,
    weather_factor: float = 1.0,
    ump_factor: float = 1.0,
) -> dict[str, Any]:
    """Compute dedicated totals projection.

    Uses additive adjustments instead of multiplicative to prevent
    compounding bias. All factors are expressed as deviations from
    the league-average total.

    Returns:
        {
            projected_total: float,
            away_runs: float,
            home_runs: float,
            confidence: str,  # "high", "medium", "low"
            factors_breakdown: dict,
        }
    """
    # Start from league average
    base_total = LEAGUE_AVG_TOTAL

    # ── OFFENSIVE ADJUSTMENTS (additive) ──
    # Convert multiplicative factors to additive deviations
    # off_factor of 1.05 means team scores 5% more than average
    # On a ~4.45 R/G base, that's +0.22 runs
    avg_rpg = base_total / 2

    away_off_adj = (away_off_factor - 1.0) * avg_rpg
    home_off_adj = (home_off_factor - 1.0) * avg_rpg
    offense_adj = (away_off_adj + home_off_adj) * FACTOR_SCALE_DAMPENER

    # ── PITCHING ADJUSTMENTS (additive) ──
    # Pitching factor > 1.0 means MORE runs allowed (worse pitching)
    # Factor applied to opposing team's offense
    away_pitch_adj = (away_pitching_factor - 1.0) * avg_rpg  # Away pitcher affects home runs
    home_pitch_adj = (home_pitching_factor - 1.0) * avg_rpg  # Home pitcher affects away runs
    pitching_adj = (away_pitch_adj + home_pitch_adj) * FACTOR_SCALE_DAMPENER

    # ── PARK ADJUSTMENT (additive, pre-calibrated) ──
    park_adj = TOTALS_PARK_ADJ.get(venue_id, 0.0) if venue_id else 0.0

    # ── WEATHER (additive, dampened from ML model) ──
    # Convert multiplicative weather factor to additive
    # But dampen it — weather effect on totals is smaller than on individual games
    weather_adj = (weather_factor - 1.0) * base_total * 0.5  # 50% dampening

    # ── UMPIRE (additive, dampened) ──
    ump_adj = (ump_factor - 1.0) * base_total * 0.6  # 60% dampening

    # ── RAW PROJECTION ──
    raw_total = base_total + offense_adj + pitching_adj + park_adj + weather_adj + ump_adj

    # ── DYNAMIC REGRESSION TO MEAN ──
    # Key insight: regression should be LIGHTER when multiple strong signals
    # agree in the same direction (e.g., two aces + pitcher's park = confident low).
    # Regression should be HEAVIER when signals conflict or are weak.

    # Count how many factors push in the same direction
    adjustments = [offense_adj, pitching_adj, park_adj, weather_adj, ump_adj]
    non_zero = [a for a in adjustments if abs(a) > 0.05]
    if non_zero:
        # What fraction of meaningful factors agree on direction?
        direction = 1 if (raw_total - base_total) > 0 else -1
        agreeing = sum(1 for a in non_zero if (a > 0) == (direction > 0))
        agreement_rate = agreeing / len(non_zero) if non_zero else 0.5

        # How strong is the total signal?
        total_signal = abs(raw_total - base_total)

        # Dynamic regression:
        # - Strong agreement (80%+) + strong signal (1.0+ run): regress only 30%
        # - Weak/conflicting signals: regress full 55%
        # - Middle ground: scale between 30-55%
        if agreement_rate >= 0.75 and total_signal >= 0.8:
            regression = 0.30  # High conviction — trust the signal more
        elif agreement_rate >= 0.60 and total_signal >= 0.5:
            regression = 0.40  # Moderate conviction
        else:
            regression = REGRESSION_TO_MEAN  # Default 55% — weak signal
    else:
        regression = REGRESSION_TO_MEAN

    projected_total = (raw_total * (1 - regression) +
                       LEAGUE_AVG_TOTAL * regression)

    # ── SPLIT INTO TEAM RUNS ──
    # Use the offensive factors to split the total between teams
    total_off = away_off_factor + home_off_factor
    away_share = away_off_factor / total_off if total_off > 0 else 0.5
    home_share = home_off_factor / total_off if total_off > 0 else 0.5

    # Adjust shares for pitching matchup
    # Away team faces home pitcher — if home pitcher is good, away gets fewer runs
    away_pitch_modifier = 1.0 - (home_pitching_factor - 1.0) * 0.3
    home_pitch_modifier = 1.0 - (away_pitching_factor - 1.0) * 0.3

    away_runs = projected_total * away_share * away_pitch_modifier
    home_runs = projected_total * home_share * home_pitch_modifier

    # Renormalize to match projected total
    run_sum = away_runs + home_runs
    if run_sum > 0:
        away_runs = away_runs / run_sum * projected_total
        home_runs = home_runs / run_sum * projected_total

    # Home field bump
    away_runs -= 0.12
    home_runs += 0.12

    # Floor
    away_runs = max(away_runs, 2.5)
    home_runs = max(home_runs, 2.5)
    projected_total = away_runs + home_runs

    # ── CONFIDENCE ──
    deviation = abs(projected_total - LEAGUE_AVG_TOTAL)
    if deviation >= 1.0:
        confidence = "high"
    elif deviation >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "projected_total": round(projected_total, 1),
        "away_runs": round(away_runs, 2),
        "home_runs": round(home_runs, 2),
        "confidence": confidence,
        "ou_line": round(projected_total * 2) / 2,
        "factors_breakdown": {
            "base": LEAGUE_AVG_TOTAL,
            "offense_adj": round(offense_adj, 2),
            "pitching_adj": round(pitching_adj, 2),
            "park_adj": round(park_adj, 2),
            "weather_adj": round(weather_adj, 2),
            "ump_adj": round(ump_adj, 2),
            "raw_total": round(raw_total, 2),
            "regression": regression,
            "final_total": round(projected_total, 1),
        },
    }


def evaluate_total_bet(
    model_total: float,
    posted_line: float,
    confidence: str,
    over_odds: int = -110,
    under_odds: int = -110,
) -> dict[str, Any]:
    """Evaluate if there's edge on an over/under bet.

    Only recommends bets when:
    1. Model disagrees with posted line by a meaningful amount
    2. Confidence is medium or high
    """
    diff = model_total - posted_line

    # Minimum disagreement thresholds by confidence
    min_diff = {"high": 0.7, "medium": 1.0, "low": 1.5}
    threshold = min_diff.get(confidence, 1.5)

    if abs(diff) < threshold:
        return {"recommendation": "NO BET", "reason": f"Model within {threshold} of line"}

    if diff > 0:
        side = "OVER"
        # Simple probability estimate based on deviation
        # Every 0.5 run deviation ≈ 3-4% probability shift
        edge_estimate = min(diff * 0.06, 0.15)  # Cap at 15%
    else:
        side = "UNDER"
        edge_estimate = min(abs(diff) * 0.06, 0.15)

    # Rating
    if abs(diff) >= 2.0:
        rating = "STRONG"
    elif abs(diff) >= 1.5:
        rating = "GOOD"
    else:
        rating = "LEAN"

    return {
        "recommendation": side,
        "diff": round(diff, 1),
        "edge_estimate": round(edge_estimate * 100, 1),
        "rating": rating,
        "confidence": confidence,
    }
