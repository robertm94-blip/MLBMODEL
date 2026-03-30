"""Feature engineering for MLB score predictions.

Computes offensive/defensive strength factors for each team using
team-level and pitcher-level stats, normalized against league averages.
"""

from typing import Any

# Park factors: venue_id -> (batting_factor, pitching_factor)
# Values > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly
# Based on historical multi-year park factors
PARK_FACTORS = {
    # Coors Field (COL)
    19: 1.38,
    # Great American Ball Park (CIN)
    2602: 1.12,
    # Fenway Park (BOS)
    3: 1.08,
    # Globe Life Field (TEX)
    5325: 1.06,
    # Yankee Stadium (NYY)
    3313: 1.05,
    # Wrigley Field (CHC)
    17: 1.05,
    # Citizens Bank Park (PHI)
    2681: 1.04,
    # Nationals Park (WSH)
    3309: 1.02,
    # Minute Maid Park (HOU)
    2392: 1.02,
    # Camden Yards (BAL)
    2: 1.02,
    # Target Field (MIN)
    3312: 1.01,
    # Chase Field (ARI)
    15: 1.01,
    # Rogers Centre (TOR)
    14: 1.00,
    # Progressive Field (CLE)
    5: 0.99,
    # Angel Stadium (LAA)
    1: 0.99,
    # PNC Park (PIT)
    31: 0.99,
    # Busch Stadium (STL)
    2889: 0.98,
    # Truist Park (ATL)
    4705: 0.98,
    # American Family Field (MIL)
    32: 0.98,
    # Kauffman Stadium (KC)
    7: 0.97,
    # Tropicana Field (TB)
    12: 0.97,
    # Guaranteed Rate Field (CWS)
    4: 0.97,
    # T-Mobile Park (SEA)
    680: 0.96,
    # Dodger Stadium (LAD)
    22: 0.96,
    # Citi Field (NYM)
    3289: 0.95,
    # Petco Park (SD)
    2680: 0.94,
    # loanDepot Park (MIA)
    4169: 0.93,
    # Oracle Park (SF)
    2395: 0.93,
    # Oakland Coliseum / Athletics
    10: 0.96,
    # Comerica Park (DET)
    2394: 0.97,
}

# Home field advantage in runs (historical MLB average ~0.25 runs)
HOME_ADVANTAGE = 0.25


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def compute_team_offensive_factor(
    team_hitting: dict, league_avg_rpg: float
) -> float:
    """Compute a team's offensive strength factor relative to league average.

    Returns a multiplier: 1.0 = league average, >1.0 = above average offense.
    """
    runs = safe_float(team_hitting.get("runs", 0))
    games = safe_float(team_hitting.get("gamesPlayed", 0))

    if games == 0 or league_avg_rpg == 0:
        return 1.0

    team_rpg = runs / games
    return team_rpg / league_avg_rpg


def compute_team_defensive_factor(
    team_pitching: dict, league_avg_rpg: float
) -> float:
    """Compute a team's defensive/pitching strength factor.

    Returns a multiplier: <1.0 = better than average defense (allows fewer runs).
    """
    runs_allowed = safe_float(team_pitching.get("runs", 0))
    games = safe_float(team_pitching.get("gamesPlayed", 0))

    if games == 0 or league_avg_rpg == 0:
        return 1.0

    team_rapg = runs_allowed / games
    return team_rapg / league_avg_rpg


def compute_pitcher_factor(
    pitcher_stats: dict, team_pitching: dict, league_avg_rpg: float
) -> float:
    """Compute starting pitcher quality factor.

    Blends the pitcher's ERA against team ERA to measure how much better/worse
    the starter is compared to the team's average arm. Also incorporates
    WHIP and K/9 as secondary signals.

    Returns a multiplier applied to the opposing team's expected runs.
    < 1.0 = pitcher suppresses scoring, > 1.0 = pitcher inflates scoring.
    """
    pitcher_era = safe_float(pitcher_stats.get("era", 0))
    pitcher_whip = safe_float(pitcher_stats.get("whip", 0))
    pitcher_ip = safe_float(pitcher_stats.get("inningsPitched", 0))

    team_era = safe_float(team_pitching.get("era", 0))

    # If no pitcher data, assume league-average starter
    if pitcher_era == 0 or pitcher_ip < 5:
        return 1.0

    # If no team ERA to compare against, use league average ~4.20
    if team_era == 0:
        team_era = 4.20

    # ERA-based factor (primary signal, 70% weight)
    league_era = league_avg_rpg * 9  # rough conversion: rpg * 9 innings ≈ ERA scale
    if league_era == 0:
        league_era = 4.20
    era_factor = pitcher_era / league_era

    # WHIP-based factor (secondary signal, 20% weight)
    league_whip = 1.30  # approximate league average
    whip_factor = pitcher_whip / league_whip if pitcher_whip > 0 else 1.0

    # Innings-based reliability weight (more IP = more reliable)
    reliability = min(pitcher_ip / 80.0, 1.0)  # Full weight at 80+ IP

    # Blend: weighted combination of ERA and WHIP factors
    raw_factor = (0.75 * era_factor) + (0.25 * whip_factor)

    # Regress toward 1.0 based on sample size
    return (raw_factor * reliability) + (1.0 * (1 - reliability))


def compute_expected_runs(
    batting_team_off_factor: float,
    pitching_team_def_factor: float,
    starter_factor: float,
    park_factor: float,
    league_avg_rpg: float,
    is_home: bool,
) -> float:
    """Compute expected runs for a team in a game.

    Formula:
        E[runs] = league_avg * off_factor * def_factor * starter_adj * park * home_adj

    The starter factor adjusts the defensive factor to account for the specific
    pitcher on the mound (rather than the team's season-long average).
    """
    # Starter adjustment: blend starter quality with team pitching
    # Starters typically pitch ~5.5 innings (61% of game), bullpen covers rest
    starter_weight = 0.55
    pitching_factor = (
        starter_factor * starter_weight + pitching_team_def_factor * (1 - starter_weight)
    )

    expected = league_avg_rpg * batting_team_off_factor * pitching_factor * park_factor

    if is_home:
        expected += HOME_ADVANTAGE

    # Floor at 1.5 runs, cap at 12 runs for reasonable bounds
    return max(1.5, min(expected, 12.0))


def get_park_factor(venue_id: int | None) -> float:
    """Look up park factor for a venue. Defaults to 1.0 (neutral)."""
    if venue_id is None:
        return 1.0
    return PARK_FACTORS.get(venue_id, 1.0)
