"""Pure helpers for the totals model backtest / calibration.

No I/O — callers pass in the parsed JSON. Keeps `backtest_totals.py` thin and
makes these reusable for future grid searches or hyperparameter tuning.
"""

from __future__ import annotations

from typing import Any

from src import totals_model


def parse_team_stats(payload: dict) -> tuple[
    dict[int, tuple[float, int]],   # team_id -> (runs_scored, games_played)
    dict[int, tuple[float, int]],   # team_id -> (runs_allowed, games_played)
    float,                          # league_rpg
]:
    """Walk `data/team_stats_{year}.json` and extract per-team RS/RA + games.

    The file shape is `{"stats": [<hitting_block>, <pitching_block>]}`.
    Each block has `splits[]` with `team.id`, `stat.runs`, `stat.gamesPlayed`.
    """
    offense: dict[int, tuple[float, int]] = {}
    pitching: dict[int, tuple[float, int]] = {}
    total_runs = 0.0
    total_games = 0

    for block in payload.get("stats", []):
        group = (block.get("group") or {}).get("displayName", "").lower()
        for split in block.get("splits", []):
            team_id = (split.get("team") or {}).get("id")
            stat = split.get("stat") or {}
            runs = float(stat.get("runs", 0) or 0)
            games = int(stat.get("gamesPlayed", 0) or 0)
            if team_id is None or games <= 0:
                continue
            if group == "hitting":
                offense[team_id] = (runs, games)
                total_runs += runs
                total_games += games
            elif group == "pitching":
                pitching[team_id] = (runs, games)

    league_rpg = total_runs / total_games if total_games else 4.5
    return offense, pitching, league_rpg


def factors_for_game(
    game: dict[str, Any],
    offense: dict[int, tuple[float, int]],
    pitching: dict[int, tuple[float, int]],
    league_rpg: float,
) -> tuple[float, float, float, float] | None:
    """Return (away_off, home_off, away_pitch, home_pitch) factors or None
    if either team is missing season stats."""
    away_id = game.get("away_team_id")
    home_id = game.get("home_team_id")
    if away_id not in offense or home_id not in offense:
        return None
    if away_id not in pitching or home_id not in pitching:
        return None

    away_runs, away_games = offense[away_id]
    home_runs, home_games = offense[home_id]
    away_ra, _ = pitching[away_id]
    home_ra, _ = pitching[home_id]

    if league_rpg <= 0 or away_games == 0 or home_games == 0:
        return None

    away_off = (away_runs / away_games) / league_rpg
    home_off = (home_runs / home_games) / league_rpg
    away_pitch = (away_ra / away_games) / league_rpg
    home_pitch = (home_ra / home_games) / league_rpg
    return away_off, home_off, away_pitch, home_pitch


def project_with_constants(
    away_off: float,
    home_off: float,
    away_pitch: float,
    home_pitch: float,
    venue_id: int | None,
    *,
    league_avg_total: float,
    regression: float,
    factor_scale: float,
) -> float:
    """Run `compute_totals_projection` with patched constants and return the
    projected_total. We patch module-level constants so we don't have to
    duplicate the formula here.

    Patching is done in-place; the caller is expected to call
    `restore_defaults` after a calibration sweep.
    """
    totals_model.LEAGUE_AVG_TOTAL = league_avg_total
    totals_model.REGRESSION_TO_MEAN = regression
    totals_model.FACTOR_SCALE_DAMPENER = factor_scale
    return totals_model.compute_totals_projection(
        away_off_factor=away_off,
        home_off_factor=home_off,
        away_pitching_factor=away_pitch,
        home_pitching_factor=home_pitch,
        venue_id=venue_id,
    )["projected_total"]


# ---- snapshot / restore for safe in-process patching ---------------------

_DEFAULTS: dict[str, Any] = {}


def snapshot_defaults() -> None:
    """Save current totals_model constants so we can restore after sweeping."""
    global _DEFAULTS
    _DEFAULTS = {
        "LEAGUE_AVG_TOTAL": totals_model.LEAGUE_AVG_TOTAL,
        "REGRESSION_TO_MEAN": totals_model.REGRESSION_TO_MEAN,
        "FACTOR_SCALE_DAMPENER": getattr(totals_model, "FACTOR_SCALE_DAMPENER", 1.0),
    }


def restore_defaults() -> None:
    if not _DEFAULTS:
        return
    totals_model.LEAGUE_AVG_TOTAL = _DEFAULTS["LEAGUE_AVG_TOTAL"]
    totals_model.REGRESSION_TO_MEAN = _DEFAULTS["REGRESSION_TO_MEAN"]
    totals_model.FACTOR_SCALE_DAMPENER = _DEFAULTS["FACTOR_SCALE_DAMPENER"]


__all__ = [
    "parse_team_stats",
    "factors_for_game",
    "project_with_constants",
    "snapshot_defaults",
    "restore_defaults",
]
