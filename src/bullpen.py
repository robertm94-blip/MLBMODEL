"""Bullpen quality module for MLB score predictions.

Aggregates reliever projections (Steamer/ZiPS/THE BAT) per team and
computes bullpen quality factors. Combined with starter projected
innings to determine the starter/bullpen split for each game.

Key concepts:
- Starter length: IP/GS from projections determines how many innings
  the starter is expected to cover
- Bullpen factor: IP-weighted ERA/FIP blend of all projected relievers
  on the team, normalized to league average
- Game pitching factor: weighted blend of starter quality (for their
  projected innings) and bullpen quality (for the remainder)
"""

import json
import os
from collections import defaultdict
from typing import Any

PROJ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "projections", "2026")
PROJECTION_SYSTEMS = ["steamer", "zips", "thebat"]

# Team abbreviation → full name mapping
_TEAM_ABBR_TO_FULL = {
    "LAD": "Los Angeles Dodgers", "NYY": "New York Yankees", "NYM": "New York Mets",
    "SEA": "Seattle Mariners", "ATL": "Atlanta Braves", "TOR": "Toronto Blue Jays",
    "PHI": "Philadelphia Phillies", "BOS": "Boston Red Sox", "DET": "Detroit Tigers",
    "BAL": "Baltimore Orioles", "MIL": "Milwaukee Brewers", "TEX": "Texas Rangers",
    "CHC": "Chicago Cubs", "PIT": "Pittsburgh Pirates", "TBR": "Tampa Bay Rays",
    "HOU": "Houston Astros", "KCR": "Kansas City Royals", "SDP": "San Diego Padres",
    "ARI": "Arizona Diamondbacks", "SFG": "San Francisco Giants", "MIN": "Minnesota Twins",
    "CIN": "Cincinnati Reds", "MIA": "Miami Marlins", "CLE": "Cleveland Guardians",
    "ATH": "Athletics", "STL": "St. Louis Cardinals", "LAA": "Los Angeles Angels",
    "WSN": "Washington Nationals", "CHW": "Chicago White Sox", "COL": "Colorado Rockies",
}

_bullpen_cache: dict[str, dict] | None = None
_starter_length_cache: dict[str, float] | None = None


def _load_pitchers_json(system: str) -> list[dict]:
    """Load pitcher projections for a system."""
    filepath = os.path.join(PROJ_DIR, f"{system}_pitchers.json")
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return json.load(f)


def _is_reliever(pitcher: dict) -> bool:
    """Determine if a pitcher is a reliever based on GS vs G."""
    gs = pitcher.get("GS", 0) or 0
    g = pitcher.get("G", 0) or 0
    if g == 0:
        return False
    # Pure reliever: 0-2 starts, or starts < 30% of appearances
    return gs <= 2 or (gs / g < 0.3)


def load_team_bullpen_projections() -> dict[str, dict[str, float]]:
    """Load and blend bullpen projections per team.

    Aggregates all relievers' ERA/FIP/WHIP/K9/BB9 weighted by projected IP
    across all available projection systems.

    Returns: {team_full_name: {era, fip, whip, k9, bb9, total_ip, n_relievers, factor}}
    """
    global _bullpen_cache
    if _bullpen_cache is not None:
        return _bullpen_cache

    # Accumulate per-team bullpen stats across systems
    # Key: team_abbr -> list of {era, fip, whip, k9, bb9, ip} per system
    team_systems: dict[str, list[dict]] = defaultdict(list)

    for system in PROJECTION_SYSTEMS:
        pitchers = _load_pitchers_json(system)
        if not pitchers:
            continue

        # Aggregate relievers by team for this system
        team_rp: dict[str, dict] = {}
        for p in pitchers:
            team = p.get("Team", "")
            if not team:
                continue
            ip = p.get("IP", 0) or 0
            if ip < 5:
                continue
            if not _is_reliever(p):
                continue

            era = p.get("ERA", 0) or 0
            fip = p.get("FIP", 0) or 0
            whip = p.get("WHIP", 0) or 0
            k9 = p.get("K/9", 0) or 0
            bb9 = p.get("BB/9", 0) or 0

            if team not in team_rp:
                team_rp[team] = {
                    "era_ip": 0, "fip_ip": 0, "whip_ip": 0,
                    "k9_ip": 0, "bb9_ip": 0, "total_ip": 0, "count": 0,
                }

            t = team_rp[team]
            t["era_ip"] += era * ip
            t["fip_ip"] += fip * ip
            t["whip_ip"] += whip * ip
            t["k9_ip"] += k9 * ip
            t["bb9_ip"] += bb9 * ip
            t["total_ip"] += ip
            t["count"] += 1

        # Convert to averages for this system
        for team, t in team_rp.items():
            if t["total_ip"] > 0:
                team_systems[team].append({
                    "era": t["era_ip"] / t["total_ip"],
                    "fip": t["fip_ip"] / t["total_ip"],
                    "whip": t["whip_ip"] / t["total_ip"],
                    "k9": t["k9_ip"] / t["total_ip"],
                    "bb9": t["bb9_ip"] / t["total_ip"],
                    "total_ip": t["total_ip"],
                    "count": t["count"],
                })

    if not team_systems:
        _bullpen_cache = {}
        return _bullpen_cache

    # Compute league average bullpen ERA/FIP for normalization
    all_eras = []
    all_fips = []
    for team, systems in team_systems.items():
        for s in systems:
            all_eras.append(s["era"])
            all_fips.append(s["fip"])
    league_bp_era = sum(all_eras) / len(all_eras) if all_eras else 4.10
    league_bp_fip = sum(all_fips) / len(all_fips) if all_fips else 4.10

    # Blend across systems
    result = {}
    for team_abbr, systems in team_systems.items():
        n = len(systems)
        blended = {
            "era": sum(s["era"] for s in systems) / n,
            "fip": sum(s["fip"] for s in systems) / n,
            "whip": sum(s["whip"] for s in systems) / n,
            "k9": sum(s["k9"] for s in systems) / n,
            "bb9": sum(s["bb9"] for s in systems) / n,
            "total_ip": sum(s["total_ip"] for s in systems) / n,
            "n_relievers": int(sum(s["count"] for s in systems) / n),
            "systems_count": n,
        }

        # Compute bullpen quality factor (ERA/FIP blend normalized to league avg)
        blended_rate = blended["era"] * 0.4 + blended["fip"] * 0.6
        league_rate = league_bp_era * 0.4 + league_bp_fip * 0.6
        blended["factor"] = blended_rate / league_rate if league_rate > 0 else 1.0

        full_name = _TEAM_ABBR_TO_FULL.get(team_abbr, team_abbr)
        result[full_name] = blended

    _bullpen_cache = result
    return _bullpen_cache


def get_starter_projected_length(pitcher_name: str, pitcher_projections: dict) -> float:
    """Get projected average innings per start for a pitcher.

    Uses IP/GS from blended projections. Falls back to 5.0 IP for unknowns.
    """
    proj = pitcher_projections.get(pitcher_name)
    if not proj:
        return 5.0

    ip = proj.get("ip", 0)
    gs = proj.get("gs", 0)

    if gs >= 3 and ip > 0:
        avg_ip = ip / gs
        # Clamp to realistic range (3.5 to 7.5 IP per start)
        return max(3.5, min(avg_ip, 7.5))

    # If no GS data, estimate from total IP
    if ip >= 100:
        return 5.5  # Likely full-season starter
    elif ip >= 50:
        return 5.0
    else:
        return 4.5  # Short outings / opener


def compute_game_pitching_factor(
    starter_factor: float,
    bullpen_factor: float,
    starter_innings: float,
) -> float:
    """Compute blended pitching factor based on starter/bullpen split.

    Instead of a fixed 55/45 split, uses the starter's projected length
    to dynamically weight starter vs bullpen.

    A starter projected for 6.5 IP covers ~72% of the game.
    A starter projected for 4.5 IP covers ~50% of the game.
    """
    total_innings = 9.0
    starter_share = min(starter_innings / total_innings, 0.85)  # Cap at 85%
    bullpen_share = 1.0 - starter_share

    return (starter_factor * starter_share) + (bullpen_factor * bullpen_share)


def get_bullpen_factor(team_name: str) -> float:
    """Get a team's bullpen quality factor. 1.0 = league average."""
    bullpens = load_team_bullpen_projections()
    bp = bullpens.get(team_name)
    if bp is None:
        return 1.0
    return bp["factor"]


def get_bullpen_summary(team_name: str) -> dict[str, Any]:
    """Get full bullpen summary for display."""
    bullpens = load_team_bullpen_projections()
    bp = bullpens.get(team_name)
    if bp is None:
        return {"era": 0, "fip": 0, "factor": 1.0, "n_relievers": 0}
    return bp
