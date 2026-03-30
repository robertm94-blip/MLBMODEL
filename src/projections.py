"""Pre-season projection data from major projection systems.

Sources:
- FanGraphs Depth Charts (50/50 blend of Steamer + ZiPS, prorated to
  RosterResource playing time) — projected records + WAR breakdowns
- ZiPS standalone projected standings
- ESPN projected records

Team projections are blended across systems for more robust estimates.
Individual pitcher projections can be loaded from CSV exports (see load_pitcher_csv).
"""

from typing import Any

# =============================================================================
# TEAM PROJECTIONS — 2026 Pre-Season
# =============================================================================
# Format: team_name -> {
#   "dc_w": Depth Charts projected wins,
#   "dc_l": Depth Charts projected losses,
#   "zips_w": ZiPS projected wins,
#   "zips_l": ZiPS projected losses,
#   "bat_war": Depth Charts projected batter WAR,
#   "pitch_war": Depth Charts projected pitcher WAR,
# }

TEAM_PROJECTIONS_2026 = {
    "Los Angeles Dodgers": {
        "dc_w": 99, "dc_l": 63,
        "zips_w": 97, "zips_l": 65,
        "bat_war": 35.3, "pitch_war": 20.6,
    },
    "New York Mets": {
        "dc_w": 90, "dc_l": 72,
        "zips_w": 89, "zips_l": 73,
        "bat_war": 31.5, "pitch_war": 15.6,
    },
    "Atlanta Braves": {
        "dc_w": 89, "dc_l": 73,
        "zips_w": 84, "zips_l": 78,
        "bat_war": 27.6, "pitch_war": 17.3,
    },
    "Seattle Mariners": {
        "dc_w": 88, "dc_l": 74,
        "zips_w": 88, "zips_l": 74,
        "bat_war": 29.1, "pitch_war": 17.8,
    },
    "Philadelphia Phillies": {
        "dc_w": 88, "dc_l": 74,
        "zips_w": 91, "zips_l": 71,
        "bat_war": 25.9, "pitch_war": 20.9,
    },
    "New York Yankees": {
        "dc_w": 87, "dc_l": 75,
        "zips_w": 88, "zips_l": 74,
        "bat_war": 30.2, "pitch_war": 17.0,
    },
    "Detroit Tigers": {
        "dc_w": 86, "dc_l": 76,
        "zips_w": 86, "zips_l": 76,
        "bat_war": 24.7, "pitch_war": 20.0,
    },
    "Chicago Cubs": {
        "dc_w": 86, "dc_l": 76,
        "zips_w": 87, "zips_l": 75,
        "bat_war": 29.4, "pitch_war": 14.1,
    },
    "Toronto Blue Jays": {
        "dc_w": 85, "dc_l": 77,
        "zips_w": 89, "zips_l": 73,
        "bat_war": 30.2, "pitch_war": 17.7,
    },
    "Boston Red Sox": {
        "dc_w": 85, "dc_l": 77,
        "zips_w": 90, "zips_l": 72,
        "bat_war": 23.8, "pitch_war": 22.4,
    },
    "Baltimore Orioles": {
        "dc_w": 84, "dc_l": 78,
        "zips_w": 88, "zips_l": 74,
        "bat_war": 30.3, "pitch_war": 14.7,
    },
    "Pittsburgh Pirates": {
        "dc_w": 84, "dc_l": 78,
        "zips_w": 79, "zips_l": 83,
        "bat_war": 20.2, "pitch_war": 17.2,
    },
    "San Francisco Giants": {
        "dc_w": 82, "dc_l": 80,
        "zips_w": 84, "zips_l": 78,
        "bat_war": 26.4, "pitch_war": 12.5,
    },
    "Milwaukee Brewers": {
        "dc_w": 82, "dc_l": 80,
        "zips_w": 85, "zips_l": 77,
        "bat_war": 22.6, "pitch_war": 15.8,
    },
    "Arizona Diamondbacks": {
        "dc_w": 82, "dc_l": 80,
        "zips_w": 82, "zips_l": 80,
        "bat_war": 25.9, "pitch_war": 12.1,
    },
    "Kansas City Royals": {
        "dc_w": 81, "dc_l": 81,
        "zips_w": 82, "zips_l": 80,
        "bat_war": 22.5, "pitch_war": 16.0,
    },
    "Texas Rangers": {
        "dc_w": 81, "dc_l": 81,
        "zips_w": 81, "zips_l": 81,
        "bat_war": 23.1, "pitch_war": 16.7,
    },
    "Houston Astros": {
        "dc_w": 80, "dc_l": 82,
        "zips_w": 84, "zips_l": 78,
        "bat_war": 26.1, "pitch_war": 14.8,
    },
    "Tampa Bay Rays": {
        "dc_w": 80, "dc_l": 82,
        "zips_w": 74, "zips_l": 88,
        "bat_war": 19.9, "pitch_war": 19.1,
    },
    "San Diego Padres": {
        "dc_w": 80, "dc_l": 82,
        "zips_w": 83, "zips_l": 79,
        "bat_war": 25.7, "pitch_war": 14.8,
    },
    "Athletics": {
        "dc_w": 79, "dc_l": 83,
        "zips_w": 74, "zips_l": 88,
        "bat_war": 25.7, "pitch_war": 11.7,
    },
    "Minnesota Twins": {
        "dc_w": 78, "dc_l": 84,
        "zips_w": 77, "zips_l": 85,
        "bat_war": 21.9, "pitch_war": 14.8,
    },
    "Cincinnati Reds": {
        "dc_w": 77, "dc_l": 85,
        "zips_w": 76, "zips_l": 86,
        "bat_war": 19.0, "pitch_war": 15.7,
    },
    "Cleveland Guardians": {
        "dc_w": 76, "dc_l": 86,
        "zips_w": 78, "zips_l": 84,
        "bat_war": 22.0, "pitch_war": 12.9,
    },
    "St. Louis Cardinals": {
        "dc_w": 75, "dc_l": 87,
        "zips_w": 76, "zips_l": 86,
        "bat_war": 22.1, "pitch_war": 9.8,
    },
    "Miami Marlins": {
        "dc_w": 75, "dc_l": 87,
        "zips_w": 76, "zips_l": 86,
        "bat_war": 17.4, "pitch_war": 13.6,
    },
    "Los Angeles Angels": {
        "dc_w": 72, "dc_l": 90,
        "zips_w": 67, "zips_l": 95,
        "bat_war": 16.4, "pitch_war": 13.1,
    },
    "Washington Nationals": {
        "dc_w": 68, "dc_l": 94,
        "zips_w": 63, "zips_l": 99,
        "bat_war": 16.9, "pitch_war": 8.7,
    },
    "Chicago White Sox": {
        "dc_w": 67, "dc_l": 95,
        "zips_w": 72, "zips_l": 90,
        "bat_war": 16.1, "pitch_war": 11.5,
    },
    "Colorado Rockies": {
        "dc_w": 65, "dc_l": 97,
        "zips_w": 60, "zips_l": 102,
        "bat_war": 14.8, "pitch_war": 7.9,
    },
}


# =============================================================================
# DERIVED TEAM STRENGTH — Runs Scored / Runs Allowed
# =============================================================================
# We derive projected RS/RA from the blended projected records using
# the inverse PythagenPat formula, anchored to league-average run environment.

LEAGUE_AVG_RPG_2026 = 4.50  # Projected league-average runs per game
REPLACEMENT_LEVEL_WINS = 47.7  # Replacement-level wins over 162 games
RUNS_PER_WAR = 10.0  # ~10 runs of value per 1 WAR


def _compute_projected_rs_ra(proj: dict) -> tuple[float, float]:
    """Derive projected runs scored and allowed from WAR + blended record.

    Uses two signals:
    1. Blended W-L record → total run differential via PythagenPat inverse
    2. WAR split (bat vs pitch) → apportion differential to offense vs defense

    Returns (projected_rs_per_game, projected_ra_per_game).
    """
    # Blend Depth Charts and ZiPS records (60/40 weight — DC is more current)
    blended_w = proj["dc_w"] * 0.6 + proj["zips_w"] * 0.4
    blended_l = proj["dc_l"] * 0.6 + proj["zips_l"] * 0.4
    win_pct = blended_w / (blended_w + blended_l)

    # Total projected WAR
    total_war = proj["bat_war"] + proj["pitch_war"]

    # WAR-based run differential (above replacement)
    # Replacement team: ~47.7 wins → needs ~(81 - 47.7) * 10 = 333 runs above repl
    war_above_avg = total_war - 22.7  # ~22.7 WAR = average team (81-win pace)
    run_diff_from_war = war_above_avg * RUNS_PER_WAR

    # Split differential: offensive WAR → runs scored boost, pitching WAR → runs allowed reduction
    avg_bat_war = 23.0  # approximate league average batter WAR
    avg_pitch_war = 14.9  # approximate league average pitcher WAR
    offensive_boost = (proj["bat_war"] - avg_bat_war) * RUNS_PER_WAR
    defensive_boost = (proj["pitch_war"] - avg_pitch_war) * RUNS_PER_WAR

    # Projected runs per game
    rs_per_game = LEAGUE_AVG_RPG_2026 + (offensive_boost / 162)
    ra_per_game = LEAGUE_AVG_RPG_2026 - (defensive_boost / 162)

    # Sanity bounds
    rs_per_game = max(3.2, min(rs_per_game, 5.8))
    ra_per_game = max(3.2, min(ra_per_game, 5.8))

    return rs_per_game, ra_per_game


def get_team_projected_strength(team_name: str) -> dict[str, float]:
    """Get a team's projected offensive and defensive strength.

    Returns {
        "proj_rs_per_game": float,  # Projected runs scored per game
        "proj_ra_per_game": float,  # Projected runs allowed per game
        "proj_off_factor": float,   # Offensive factor (1.0 = league avg)
        "proj_def_factor": float,   # Defensive factor (<1.0 = better defense)
        "blended_wins": float,      # Blended projected wins
        "bat_war": float,
        "pitch_war": float,
    }
    """
    proj = TEAM_PROJECTIONS_2026.get(team_name)
    if proj is None:
        return {
            "proj_rs_per_game": LEAGUE_AVG_RPG_2026,
            "proj_ra_per_game": LEAGUE_AVG_RPG_2026,
            "proj_off_factor": 1.0,
            "proj_def_factor": 1.0,
            "blended_wins": 81.0,
            "bat_war": 23.0,
            "pitch_war": 14.9,
        }

    rs, ra = _compute_projected_rs_ra(proj)
    blended_w = proj["dc_w"] * 0.6 + proj["zips_w"] * 0.4

    return {
        "proj_rs_per_game": round(rs, 3),
        "proj_ra_per_game": round(ra, 3),
        "proj_off_factor": round(rs / LEAGUE_AVG_RPG_2026, 4),
        "proj_def_factor": round(ra / LEAGUE_AVG_RPG_2026, 4),
        "blended_wins": round(blended_w, 1),
        "bat_war": proj["bat_war"],
        "pitch_war": proj["pitch_war"],
    }


# =============================================================================
# PITCHER PROJECTIONS — CSV Import
# =============================================================================
# Users can export pitcher projections from FanGraphs as CSV:
#   - Steamer: projections?type=steamer&stats=pit
#   - ZiPS: projections?type=zips&stats=pit
#   - THE BAT: projections?type=thebat&stats=pit
#
# Place CSVs in data/ directory:
#   data/steamer_pitchers.csv
#   data/zips_pitchers.csv
#   data/thebat_pitchers.csv

import csv
import os

PROJECTION_SYSTEMS = ["steamer", "zips", "thebat"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_pitcher_csv(filepath: str) -> dict[str, dict[str, float]]:
    """Load pitcher projections from a FanGraphs CSV export.

    Expected columns: Name, Team, ERA, FIP, WHIP, K/9, BB/9, IP, WAR
    (FanGraphs exports use these column names)

    Returns: {pitcher_name: {era, fip, whip, k9, bb9, ip, war}}
    """
    pitchers = {}
    if not os.path.exists(filepath):
        return pitchers

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name") or row.get("PlayerName") or row.get("name", "")
            if not name:
                continue

            def _float(key: str, *alt_keys: str) -> float:
                for k in (key, *alt_keys):
                    val = row.get(k, "")
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
                return 0.0

            pitchers[name] = {
                "era": _float("ERA"),
                "fip": _float("FIP"),
                "whip": _float("WHIP"),
                "k9": _float("K/9", "K9"),
                "bb9": _float("BB/9", "BB9"),
                "ip": _float("IP"),
                "war": _float("WAR"),
                "team": row.get("Team", row.get("team", "")),
            }

    return pitchers


def load_all_pitcher_projections() -> dict[str, dict[str, Any]]:
    """Load and blend pitcher projections from all available CSV systems.

    Blends Steamer, ZiPS, and THE BAT with equal weight for available systems.
    Returns: {pitcher_name: {era, fip, whip, k9, bb9, ip, war, systems_count}}
    """
    all_systems = {}
    systems_loaded = []

    for system in PROJECTION_SYSTEMS:
        filepath = os.path.join(DATA_DIR, f"{system}_pitchers.csv")
        data = load_pitcher_csv(filepath)
        if data:
            all_systems[system] = data
            systems_loaded.append(system)

    if not all_systems:
        return {}

    # Collect all pitcher names across systems
    all_names = set()
    for system_data in all_systems.values():
        all_names.update(system_data.keys())

    # Blend projections with equal weight
    blended = {}
    stat_keys = ["era", "fip", "whip", "k9", "bb9", "ip", "war"]

    for name in all_names:
        available = []
        for system in systems_loaded:
            if name in all_systems[system]:
                available.append(all_systems[system][name])

        if not available:
            continue

        blended_stats = {}
        for key in stat_keys:
            values = [s[key] for s in available if s[key] > 0]
            blended_stats[key] = sum(values) / len(values) if values else 0.0

        blended_stats["systems_count"] = len(available)
        blended_stats["team"] = available[0].get("team", "")
        blended[name] = blended_stats

    return blended
