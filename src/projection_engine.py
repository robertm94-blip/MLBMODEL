"""Core projection engine.

Consumes the per-team feature rows produced by `src/feature_pipeline.py`
(stored in the SQLite `team_features` table) and produces, per game:
  - expected runs (lambda) for each side
  - win probabilities via the existing Negative Binomial model
  - a "fair line" in American odds for each side (vig-free price)
  - a "projected total" via the dedicated totals model
  - a totals confidence flag

The Kelly staking module already lives in `src/edge.py`. Market-odds /
edge / Kelly fields are intentionally not populated here: the user
deferred the odds source for now, so this engine emits fair lines only.
A later task can wire `src/edge.analyze_game` over the same per-game
output without touching this module.
"""

from __future__ import annotations

from typing import Any

from src.edge import american_to_decimal, decimal_to_american
from src.features import HOME_ADVANTAGE, compute_expected_runs
from src.model import generate_prediction
from src.projections import get_league_avg_rpg
from src.totals_model import compute_totals_projection


# League-average FIP. Used to convert per-team blended FIP from the
# features into the unitless "factor" shape expected by
# compute_expected_runs (1.0 = league-average pitching).
LEAGUE_AVG_FIP = 4.05


def fip_to_factor(fip: float | None) -> float:
    """Convert blended FIP to a runs-allowed factor (1.0 = league average).

    > 1.0  = pitcher worse than average (allows more runs)
    < 1.0  = pitcher better than average (suppresses runs)
    """
    if fip is None or fip <= 0:
        return 1.0
    return float(fip) / LEAGUE_AVG_FIP


def prob_to_fair_line(probability: float) -> tuple[float | None, int | None]:
    """Convert a model probability to (fair decimal odds, fair American line).

    Returns (None, None) at the boundary 0/1 where odds are undefined.
    """
    if probability is None or probability <= 0.0 or probability >= 1.0:
        return None, None
    decimal = 1.0 / float(probability)
    american = decimal_to_american(decimal)
    return round(decimal, 4), int(american)


def compute_game_projection(
    home: dict[str, Any],
    away: dict[str, Any],
    *,
    league_avg_rpg: float | None = None,
    park_runs_factor: float | None = None,
    weather_factor: float = 1.0,
    weather_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce the complete projection for one game.

    `home` and `away` are rows from the `team_features` table (or any
    dict with the same keys). The two rows are expected to share a
    venue and game date.

    `weather_factor` (default 1.0) is a multiplicative scoring effect
    derived from temperature/wind/humidity at the venue. Applied to both
    teams' lambdas (so the NB-derived total + win prob both reflect it)
    AND fed into the dedicated totals model. `weather_summary` is the
    breakdown dict from `src.weather.fetch_game_weather` for transparency.
    """
    league = league_avg_rpg if league_avg_rpg is not None else get_league_avg_rpg()
    park = park_runs_factor
    if park is None:
        park = home.get("park_runs_factor") or away.get("park_runs_factor") or 1.0
    park = float(park)
    wx = float(weather_factor or 1.0)

    home_off = float(home.get("offensive_factor") or 1.0)
    away_off = float(away.get("offensive_factor") or 1.0)
    home_pitch_factor = fip_to_factor(home.get("pitching_blended_fip"))
    away_pitch_factor = fip_to_factor(away.get("pitching_blended_fip"))

    # Away batting vs home pitching. We feed the same blended-FIP factor
    # into both the team-defense and starter slots, since the team_features
    # FIP already incorporates the IP-share weighted starter+bullpen blend
    # (handles openers smoothly via starter_share). compute_expected_runs
    # then collapses the 65/35 internal weighting to that single value.
    # Weather is applied multiplicatively after the floor/cap, mirroring
    # predict_full.py:184-200.
    away_lambda = compute_expected_runs(
        batting_team_off_factor=away_off,
        pitching_team_def_factor=home_pitch_factor,
        starter_factor=home_pitch_factor,
        park_factor=park,
        league_avg_rpg=league,
        is_home=False,
    ) * wx
    home_lambda = compute_expected_runs(
        batting_team_off_factor=home_off,
        pitching_team_def_factor=away_pitch_factor,
        starter_factor=away_pitch_factor,
        park_factor=park,
        league_avg_rpg=league,
        is_home=True,
    ) * wx

    away_team = away.get("team_name") or ""
    home_team = home.get("team_name") or ""
    prediction = generate_prediction(away_lambda, home_lambda, away_team, home_team)

    away_prob = prediction["win_probability"]["away"] / 100.0
    home_prob = prediction["win_probability"]["home"] / 100.0
    away_fair_decimal, away_fair_line = prob_to_fair_line(away_prob)
    home_fair_decimal, home_fair_line = prob_to_fair_line(home_prob)

    # Dedicated totals model (additive, with dynamic regression). Weather
    # contributes via the existing 50%-dampened additive adjustment in
    # compute_totals_projection (src/totals_model.py:121).
    totals = compute_totals_projection(
        away_off_factor=away_off,
        home_off_factor=home_off,
        away_pitching_factor=away_pitch_factor,
        home_pitching_factor=home_pitch_factor,
        venue_id=_int_or_none(home.get("venue_id") or away.get("venue_id")),
        weather_factor=wx,
    )

    return {
        "game_id": home.get("game_id") or away.get("game_id"),
        "game_date": home.get("game_date") or away.get("game_date"),
        "venue_id": home.get("venue_id") or away.get("venue_id"),
        "park_runs_factor": round(park, 4),

        "away_team_id": away.get("team_id"),
        "home_team_id": home.get("team_id"),
        "away_team_name": away_team,
        "home_team_name": home_team,

        "away_offensive_factor": round(away_off, 4),
        "home_offensive_factor": round(home_off, 4),
        "away_pitching_fip": _round(away.get("pitching_blended_fip"), 3),
        "home_pitching_fip": _round(home.get("pitching_blended_fip"), 3),
        "away_pitching_factor": round(away_pitch_factor, 4),
        "home_pitching_factor": round(home_pitch_factor, 4),
        "away_is_opener": bool(away.get("is_opener")),
        "home_is_opener": bool(home.get("is_opener")),
        "away_is_bullpen_heavy": bool(away.get("is_bullpen_heavy")),
        "home_is_bullpen_heavy": bool(home.get("is_bullpen_heavy")),
        "home_advantage_runs": HOME_ADVANTAGE,
        "weather_factor": round(wx, 4),
        "weather_summary": weather_summary or {},

        "away_lambda": round(float(away_lambda), 3),
        "home_lambda": round(float(home_lambda), 3),
        "away_win_pct": prediction["win_probability"]["away"],
        "home_win_pct": prediction["win_probability"]["home"],

        "away_fair_decimal": away_fair_decimal,
        "home_fair_decimal": home_fair_decimal,
        "away_fair_line": away_fair_line,
        "home_fair_line": home_fair_line,

        "expected_total_nb": prediction["expected_total"],
        "projected_total": totals["projected_total"],
        "ou_line": totals["ou_line"],
        "totals_confidence": totals["confidence"],
        "totals_breakdown": totals["factors_breakdown"],

        "predicted_score_away": prediction["predicted_score"]["away"],
        "predicted_score_home": prediction["predicted_score"]["home"],
        "score_probability_pct": prediction["score_probability"],
    }


def _round(value: Any, ndigits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# Re-export for convenience so callers don't need to import edge.py just
# to convert future market lines.
__all__ = [
    "LEAGUE_AVG_FIP",
    "fip_to_factor",
    "prob_to_fair_line",
    "compute_game_projection",
    "american_to_decimal",
    "decimal_to_american",
]
