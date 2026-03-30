"""Projection data aggregation from Steamer, ZiPS, and THE BAT.

Loads team-level projected standings (RS/RA per game) and individual
pitcher projections from FanGraphs API JSON exports. Blends all available
systems with equal weight for the most robust estimates.

Data files (in data/projections/2026/):
- fangraphs_projected_standings.json  — Depth Charts composite standings
- steamer_pitchers.json               — Steamer pitcher projections
- zips_pitchers.json                  — ZiPS pitcher projections
- thebat_pitchers.json                — THE BAT pitcher projections
- steamer_batters.json                — Steamer batter projections
- zips_batters.json                   — ZiPS batter projections
- thebat_batters.json                 — THE BAT batter projections
"""

import json
import os
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "projections", "2026")

# Team short name → MLB Stats API full name mapping
_TEAM_NAME_MAP = {
    "Dodgers": "Los Angeles Dodgers",
    "Yankees": "New York Yankees",
    "Mets": "New York Mets",
    "Mariners": "Seattle Mariners",
    "Braves": "Atlanta Braves",
    "Blue Jays": "Toronto Blue Jays",
    "Phillies": "Philadelphia Phillies",
    "Red Sox": "Boston Red Sox",
    "Tigers": "Detroit Tigers",
    "Orioles": "Baltimore Orioles",
    "Brewers": "Milwaukee Brewers",
    "Rangers": "Texas Rangers",
    "Cubs": "Chicago Cubs",
    "Pirates": "Pittsburgh Pirates",
    "Rays": "Tampa Bay Rays",
    "Astros": "Houston Astros",
    "Royals": "Kansas City Royals",
    "Padres": "San Diego Padres",
    "Diamondbacks": "Arizona Diamondbacks",
    "Giants": "San Francisco Giants",
    "Twins": "Minnesota Twins",
    "Reds": "Cincinnati Reds",
    "Marlins": "Miami Marlins",
    "Guardians": "Cleveland Guardians",
    "Athletics": "Athletics",
    "Cardinals": "St. Louis Cardinals",
    "Angels": "Los Angeles Angels",
    "Nationals": "Washington Nationals",
    "White Sox": "Chicago White Sox",
    "Rockies": "Colorado Rockies",
}

# Reverse: full name → short name
_FULL_TO_SHORT = {v: k for k, v in _TEAM_NAME_MAP.items()}


def _load_json(filename: str) -> Any:
    """Load a JSON file from the projections data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)


# =============================================================================
# TEAM PROJECTED STANDINGS (Depth Charts composite)
# =============================================================================

_standings_cache: dict[str, dict] | None = None


def _load_standings() -> dict[str, dict]:
    """Load and cache projected standings keyed by full team name."""
    global _standings_cache
    if _standings_cache is not None:
        return _standings_cache

    data = _load_json("fangraphs_projected_standings.json")
    if not data:
        _standings_cache = {}
        return _standings_cache

    standings = {}
    for team in data:
        short = team.get("shortName", "")
        full_name = _TEAM_NAME_MAP.get(short, short)
        standings[full_name] = {
            "xW": team.get("xW", 81),
            "xL": team.get("xL", 81),
            "xRpG": team.get("xRpG", 4.5),
            "xRApG": team.get("xRApG", 4.5),
            "xRS": team.get("xRS", 729),
            "xRA": team.get("xRA", 729),
        }

    _standings_cache = standings
    return _standings_cache


def get_league_avg_rpg() -> float:
    """Compute league average runs per game from projected standings."""
    standings = _load_standings()
    if not standings:
        return 4.50
    total_rpg = sum(t["xRpG"] for t in standings.values())
    return total_rpg / len(standings)


LEAGUE_AVG_RPG_2026 = None  # Set lazily


def _get_league_avg() -> float:
    global LEAGUE_AVG_RPG_2026
    if LEAGUE_AVG_RPG_2026 is None:
        LEAGUE_AVG_RPG_2026 = get_league_avg_rpg()
    return LEAGUE_AVG_RPG_2026


def get_team_projected_strength(team_name: str) -> dict[str, float]:
    """Get a team's projected offensive and defensive strength.

    Uses FanGraphs Depth Charts projected standings (composite of
    Steamer + ZiPS, prorated to actual roster depth).

    Returns projected RS/RA per game and strength factors normalized
    to the league average.
    """
    standings = _load_standings()
    league_avg = _get_league_avg()

    team = standings.get(team_name)
    if team is None:
        return {
            "proj_rs_per_game": league_avg,
            "proj_ra_per_game": league_avg,
            "proj_off_factor": 1.0,
            "proj_def_factor": 1.0,
            "proj_wins": 81.0,
        }

    rs_pg = team["xRpG"]
    ra_pg = team["xRApG"]

    return {
        "proj_rs_per_game": round(rs_pg, 3),
        "proj_ra_per_game": round(ra_pg, 3),
        "proj_off_factor": round(rs_pg / league_avg, 4),
        "proj_def_factor": round(ra_pg / league_avg, 4),
        "proj_wins": round(team["xW"], 1),
    }


# =============================================================================
# PITCHER PROJECTIONS — Blended from Steamer, ZiPS, THE BAT
# =============================================================================

PROJECTION_SYSTEMS = ["steamer", "zips", "thebat"]

_pitcher_cache: dict[str, dict] | None = None


def _load_pitcher_json(system: str) -> dict[str, dict[str, float]]:
    """Load pitcher projections from a FanGraphs API JSON export."""
    data = _load_json(f"{system}_pitchers.json")
    if not data:
        return {}

    pitchers = {}
    for row in data:
        name = row.get("PlayerName", "")
        if not name:
            continue

        ip = row.get("IP", 0) or 0
        era = row.get("ERA", 0) or 0
        if ip < 1 or era == 0:
            continue

        pitchers[name] = {
            "era": era,
            "fip": row.get("FIP", 0) or 0,
            "whip": row.get("WHIP", 0) or 0,
            "k9": row.get("K/9", 0) or 0,
            "bb9": row.get("BB/9", 0) or 0,
            "ip": ip,
            "war": row.get("WAR", 0) or 0,
            "k_pct": row.get("K%", 0) or 0,
            "bb_pct": row.get("BB%", 0) or 0,
            "gb_pct": row.get("GB%", 0) or 0,
            "babip": row.get("BABIP", 0) or 0,
            "lob_pct": row.get("LOB%", 0) or 0,
            "team": row.get("Team", ""),
            "mlbam_id": row.get("xMLBAMID"),
            "gs": row.get("GS", 0) or 0,
        }

    return pitchers


def load_all_pitcher_projections() -> dict[str, dict[str, Any]]:
    """Load and blend pitcher projections from all available systems.

    Equal-weights Steamer, ZiPS, and THE BAT for each pitcher.
    Returns: {pitcher_name: {era, fip, whip, k9, bb9, ip, war, ...}}
    """
    global _pitcher_cache
    if _pitcher_cache is not None:
        return _pitcher_cache

    all_systems = {}
    systems_loaded = []

    for system in PROJECTION_SYSTEMS:
        data = _load_pitcher_json(system)
        if data:
            all_systems[system] = data
            systems_loaded.append(system)

    if not all_systems:
        _pitcher_cache = {}
        return _pitcher_cache

    # Collect all pitcher names
    all_names = set()
    for system_data in all_systems.values():
        all_names.update(system_data.keys())

    # Blend with equal weight
    stat_keys = ["era", "fip", "whip", "k9", "bb9", "ip", "war",
                 "k_pct", "bb_pct", "gb_pct", "babip", "lob_pct", "gs"]
    blended = {}

    for name in all_names:
        available = []
        for system in systems_loaded:
            if name in all_systems[system]:
                available.append(all_systems[system][name])

        if not available:
            continue

        blended_stats = {}
        for key in stat_keys:
            values = [s[key] for s in available if s.get(key, 0)]
            blended_stats[key] = sum(values) / len(values) if values else 0.0

        blended_stats["systems_count"] = len(available)
        blended_stats["systems"] = [
            sys for sys in systems_loaded if name in all_systems[sys]
        ]
        blended_stats["team"] = available[0].get("team", "")
        blended_stats["mlbam_id"] = available[0].get("mlbam_id")

        # Per-system breakdown for transparency
        blended_stats["by_system"] = {}
        for system in systems_loaded:
            if name in all_systems[system]:
                blended_stats["by_system"][system] = {
                    "era": all_systems[system][name]["era"],
                    "fip": all_systems[system][name]["fip"],
                    "ip": all_systems[system][name]["ip"],
                }

        blended[name] = blended_stats

    _pitcher_cache = blended
    return _pitcher_cache


# =============================================================================
# BATTER PROJECTIONS — Aggregate team offense from individual projections
# =============================================================================

_batter_cache: dict[str, dict] | None = None


def _load_batter_json(system: str) -> list[dict]:
    """Load batter projections from a FanGraphs API JSON export."""
    data = _load_json(f"{system}_batters.json")
    return data if data else []


def load_team_batting_projections() -> dict[str, dict[str, float]]:
    """Aggregate individual batter projections to team level.

    Sums projected PA, R, HR, and computes weighted-average OPS/wOBA
    across all systems, then averages per team.

    Returns: {team_short_name: {ops, woba, runs, hr, pa, ...}}
    """
    global _batter_cache
    if _batter_cache is not None:
        return _batter_cache

    team_totals: dict[str, dict[str, list[float]]] = {}

    for system in PROJECTION_SYSTEMS:
        batters = _load_batter_json(system)
        if not batters:
            continue

        # Aggregate by team for this system
        sys_teams: dict[str, dict[str, float]] = {}
        for b in batters:
            team = b.get("Team", "")
            if not team:
                continue
            pa = b.get("PA", 0) or 0
            if pa < 10:
                continue

            if team not in sys_teams:
                sys_teams[team] = {"pa": 0, "r": 0, "hr": 0, "ops_sum": 0,
                                   "woba_sum": 0, "pa_weight": 0}

            t = sys_teams[team]
            t["pa"] += pa
            t["r"] += b.get("R", 0) or 0
            t["hr"] += b.get("HR", 0) or 0
            ops = b.get("OPS", 0) or 0
            woba = b.get("wOBA", 0) or 0
            t["ops_sum"] += ops * pa
            t["woba_sum"] += woba * pa
            t["pa_weight"] += pa

        # Store per-system results
        for team, t in sys_teams.items():
            if team not in team_totals:
                team_totals[team] = {"ops": [], "woba": [], "r_per_pa": []}
            if t["pa_weight"] > 0:
                team_totals[team]["ops"].append(t["ops_sum"] / t["pa_weight"])
                team_totals[team]["woba"].append(t["woba_sum"] / t["pa_weight"])
            if t["pa"] > 0:
                team_totals[team]["r_per_pa"].append(t["r"] / t["pa"])

    # Average across systems
    result = {}
    for team, totals in team_totals.items():
        full_name = _TEAM_NAME_MAP.get(team, team)
        result[full_name] = {
            "ops": sum(totals["ops"]) / len(totals["ops"]) if totals["ops"] else 0.720,
            "woba": sum(totals["woba"]) / len(totals["woba"]) if totals["woba"] else 0.320,
            "r_per_pa": (sum(totals["r_per_pa"]) / len(totals["r_per_pa"])
                         if totals["r_per_pa"] else 0.11),
        }

    _batter_cache = result
    return _batter_cache
