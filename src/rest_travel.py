"""Rest and travel fatigue adjustment module.

Travel and rest patterns have measurable effects on MLB performance:
- Teams on long road trips (6+ games) see ~1-2% offensive decline
- Cross-timezone travel (especially westbound) creates a ~1.5% drag
- Day games after night games show ~2% offensive decline
- Teams on rest after an off-day show ~0.5% offensive boost

Sources:
- MLB scheduling research (Baseball Prospectus)
- "Circadian advantage in Major League Baseball" (Song et al., 2017)
"""

import json
import os
import requests
from datetime import datetime, timedelta
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Venue timezone mappings
VENUE_TIMEZONE = {
    # Eastern
    3: "ET", 2: "ET", 14: "ET", 3313: "ET", 12: "ET",  # BOS, BAL, TOR, NYY, TB
    4705: "ET", 4169: "ET", 3289: "ET", 2681: "ET", 3309: "ET",  # ATL, MIA, NYM, PHI, WSH
    2602: "ET", 31: "ET", 5: "ET", 2394: "ET",  # CIN, PIT, CLE, DET
    # Central
    4: "CT", 7: "CT", 3312: "CT", 17: "CT", 32: "CT",  # CWS, KC, MIN, CHC, MIL
    2889: "CT", 2392: "CT", 5325: "CT",  # STL, HOU, TEX
    # Mountain
    15: "MT", 19: "MT",  # ARI, COL
    # Pacific
    1: "PT", 10: "PT", 680: "PT", 22: "PT", 2680: "PT", 2395: "PT",  # LAA, OAK, SEA, LAD, SD, SF
}

TZ_OFFSET = {"ET": 0, "CT": 1, "MT": 2, "PT": 3}


def fetch_recent_schedule(team_id: int, game_date: str, lookback_days: int = 5) -> list[dict]:
    """Fetch a team's recent games from the MLB API."""
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end = (dt - timedelta(days=1)).strftime("%Y-%m-%d")

        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "teamId": team_id,
            "startDate": start,
            "endDate": end,
            "sportId": 1,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                games.append({
                    "date": game.get("officialDate", ""),
                    "game_time": game.get("gameDate", ""),
                    "venue_id": game.get("venue", {}).get("id"),
                    "status": game.get("status", {}).get("detailedState", ""),
                    "away_id": game["teams"]["away"]["team"]["id"],
                    "home_id": game["teams"]["home"]["team"]["id"],
                })
        return sorted(games, key=lambda g: g["date"])
    except Exception:
        return []


def compute_rest_travel_factor(
    team_id: int,
    team_name: str,
    today_venue_id: int | None,
    game_date: str,
    is_home: bool,
) -> tuple[float, dict[str, Any]]:
    """Compute rest and travel adjustment factor.

    Returns (factor, breakdown) where:
    - factor < 1.0 = team is fatigued (expect fewer runs)
    - factor > 1.0 = team is well-rested (expect more runs)
    """
    recent = fetch_recent_schedule(team_id, game_date, lookback_days=5)

    factor = 1.0
    breakdown = {
        "games_last_5_days": len(recent),
        "off_day_yesterday": False,
        "travel_zones": 0,
        "road_streak": 0,
        "adjustments": [],
    }

    if not recent:
        # No recent games — fully rested (likely season opener)
        breakdown["off_day_yesterday"] = True
        return 1.005, breakdown

    dt_today = datetime.strptime(game_date, "%Y-%m-%d")

    # Check yesterday
    yesterday = (dt_today - timedelta(days=1)).strftime("%Y-%m-%d")
    played_yesterday = any(g["date"] == yesterday for g in recent)
    breakdown["off_day_yesterday"] = not played_yesterday

    if not played_yesterday:
        factor += 0.005  # Off-day rest bonus
        breakdown["adjustments"].append("off_day_rest +0.5%")

    # Check timezone travel
    if today_venue_id and recent:
        last_game = recent[-1]
        last_venue = last_game.get("venue_id")
        if last_venue:
            today_tz = TZ_OFFSET.get(VENUE_TIMEZONE.get(today_venue_id, "ET"), 0)
            last_tz = TZ_OFFSET.get(VENUE_TIMEZONE.get(last_venue, "ET"), 0)
            tz_change = abs(today_tz - last_tz)

            if tz_change >= 2:
                factor -= 0.015  # Cross-country travel fatigue
                breakdown["travel_zones"] = tz_change
                breakdown["adjustments"].append(f"cross_country_travel_{tz_change}tz -1.5%")
            elif tz_change == 1:
                factor -= 0.005  # Minor timezone shift
                breakdown["travel_zones"] = 1
                breakdown["adjustments"].append("timezone_shift_1tz -0.5%")

    # Road streak check
    if not is_home:
        road_games = 0
        for g in reversed(recent):
            if g["away_id"] == team_id:
                road_games += 1
            else:
                break
        breakdown["road_streak"] = road_games
        if road_games >= 6:
            factor -= 0.012  # Extended road trip fatigue
            breakdown["adjustments"].append(f"long_road_trip_{road_games}g -1.2%")
        elif road_games >= 3:
            factor -= 0.005
            breakdown["adjustments"].append(f"road_trip_{road_games}g -0.5%")

    # Heavy schedule (4+ games in last 5 days)
    if len(recent) >= 4:
        factor -= 0.008
        breakdown["adjustments"].append(f"heavy_schedule_{len(recent)}_in_5d -0.8%")

    # Clamp factor
    factor = max(0.96, min(factor, 1.02))
    breakdown["factor"] = round(factor, 4)

    return factor, breakdown
