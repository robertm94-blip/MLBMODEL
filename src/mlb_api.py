"""MLB Stats API client for fetching team stats, pitcher stats, and schedules."""

import requests
from typing import Any

BASE_URL = "https://statsapi.mlb.com/api/v1"


def get_schedule(date: str) -> list[dict[str, Any]]:
    """Fetch the MLB schedule for a given date (YYYY-MM-DD)."""
    url = f"{BASE_URL}/schedule"
    params = {
        "date": date,
        "sportId": 1,
        "hydrate": "probablePitcher,team,linescore",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]
            game_info = {
                "game_id": game["gamePk"],
                "game_time": game.get("gameDate", ""),
                "status": game.get("status", {}).get("detailedState", ""),
                "away_team_id": away["team"]["id"],
                "away_team_name": away["team"]["name"],
                "home_team_id": home["team"]["id"],
                "home_team_name": home["team"]["name"],
                "away_pitcher_id": away.get("probablePitcher", {}).get("id"),
                "away_pitcher_name": away.get("probablePitcher", {}).get("fullName", "TBD"),
                "home_pitcher_id": home.get("probablePitcher", {}).get("id"),
                "home_pitcher_name": home.get("probablePitcher", {}).get("fullName", "TBD"),
                "venue_id": game.get("venue", {}).get("id"),
                "venue_name": game.get("venue", {}).get("name", "Unknown"),
            }
            games.append(game_info)
    return games


def get_team_stats(team_id: int, season: int) -> dict[str, Any]:
    """Fetch team-level batting and pitching stats for a season."""
    url = f"{BASE_URL}/teams/{team_id}/stats"
    params = {
        "stats": "season",
        "season": season,
        "group": "hitting,pitching",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result = {"hitting": {}, "pitching": {}}
    for stat_group in data.get("stats", []):
        group_name = stat_group.get("group", {}).get("displayName", "").lower()
        splits = stat_group.get("splits", [])
        if splits:
            stats = splits[0].get("stat", {})
            if group_name == "hitting":
                result["hitting"] = stats
            elif group_name == "pitching":
                result["pitching"] = stats
    return result


def get_pitcher_stats(pitcher_id: int, season: int) -> dict[str, Any]:
    """Fetch individual pitcher season stats."""
    if pitcher_id is None:
        return {}
    url = f"{BASE_URL}/people/{pitcher_id}/stats"
    params = {
        "stats": "season",
        "season": season,
        "group": "pitching",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    for stat_group in data.get("stats", []):
        splits = stat_group.get("splits", [])
        if splits:
            return splits[0].get("stat", {})
    return {}


def get_league_averages(season: int) -> dict[str, float]:
    """Fetch league-wide averages for normalization."""
    url = f"{BASE_URL}/teams/stats"
    params = {
        "stats": "season",
        "season": season,
        "group": "hitting",
        "sportIds": 1,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    total_runs = 0
    total_games = 0
    total_ops = []

    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            stats = split.get("stat", {})
            games = int(stats.get("gamesPlayed", 0))
            runs = int(stats.get("runs", 0))
            ops = float(stats.get("ops", ".000").replace(".", "0.", 1) if isinstance(stats.get("ops"), str) and stats.get("ops", "").startswith(".") else stats.get("ops", 0))
            total_runs += runs
            total_games += games
            if ops > 0:
                total_ops.append(ops)

    avg_rpg = total_runs / max(total_games, 1)
    avg_ops = sum(total_ops) / max(len(total_ops), 1) if total_ops else 0.720

    return {
        "avg_runs_per_game": avg_rpg,
        "avg_ops": avg_ops,
        "total_runs": total_runs,
        "total_games": total_games,
    }
