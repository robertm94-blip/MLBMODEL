"""MLB Stats API source.

Implements the `DataSource` contract against statsapi.mlb.com:
- /api/v1/schedule           -> fetch_schedule
- /api/v1/game/{id}/boxscore -> fetch_lineups
- /api/v1.1/game/{id}/feed/live -> fetch_boxscore (per-player stats + final)

This is the **most critical** source for the MLB model and is the
implementation against which the `DataSource` interface was designed.
"""

from __future__ import annotations

import logging
from typing import Any

from src.data_ingestion.base import (
    BoxscoreEntry,
    DataSource,
    FetchResult,
    Game,
    LineupEntry,
    Player,
    Team,
    Venue,
)
from src.data_ingestion.http import HttpClient

log = logging.getLogger(__name__)

SPORT = "mlb"
BASE_URL = "https://statsapi.mlb.com/api/v1"
LIVE_BASE_URL = "https://statsapi.mlb.com/api/v1.1"


class MLBStatsSource(DataSource):
    """Pull MLB schedule, lineups, and box scores from statsapi.mlb.com."""

    sport = SPORT

    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient()

    # ---- schedule -----------------------------------------------------

    def fetch_schedule(self, game_date: str) -> list[Game]:
        payload = self.http.get_json(
            f"{BASE_URL}/schedule",
            host_key="mlb",
            params={
                "date": game_date,
                "sportId": 1,
                "hydrate": "probablePitcher,team,linescore",
            },
        )
        out: list[Game] = []
        for date_entry in payload.get("dates", []):
            for raw in date_entry.get("games", []):
                game = _parse_schedule_game(raw, game_date)
                if game is not None:
                    out.append(game)
        return out

    # ---- per-game -----------------------------------------------------

    def fetch_lineups(self, game: Game) -> list[LineupEntry]:
        try:
            payload = self.http.get_json(
                f"{BASE_URL}/game/{game.game_id}/boxscore",
                host_key="mlb",
            )
        except Exception as exc:
            log.warning("MLB boxscore fetch failed for %s: %s", game.game_id, exc)
            return []
        return _extract_lineup_entries(game, payload)

    def fetch_boxscore(self, game: Game) -> FetchResult:
        try:
            payload = self.http.get_json(
                f"{LIVE_BASE_URL}/game/{game.game_id}/feed/live",
                host_key="mlb",
            )
        except Exception as exc:
            log.warning("MLB live feed fetch failed for %s: %s", game.game_id, exc)
            return FetchResult(game=game)
        box_entries = _extract_boxscore_entries(game, payload)
        lineup_entries = _extract_lineup_entries_from_live(game, payload)
        updated_game = _update_game_with_final(game, payload)
        return FetchResult(game=updated_game, lineups=lineup_entries, box_scores=box_entries)


# =====================================================================
# Parsers (private)
# =====================================================================

def _parse_schedule_game(raw: dict[str, Any], game_date: str) -> Game | None:
    if "gamePk" not in raw:
        return None
    away = raw.get("teams", {}).get("away", {})
    home = raw.get("teams", {}).get("home", {})
    away_team_raw = away.get("team", {}) or {}
    home_team_raw = home.get("team", {}) or {}
    venue_raw = raw.get("venue") or {}

    home_team = Team(
        sport=SPORT,
        team_id=str(home_team_raw.get("id")),
        team_name=home_team_raw.get("name"),
        team_abbr=home_team_raw.get("abbreviation"),
    )
    away_team = Team(
        sport=SPORT,
        team_id=str(away_team_raw.get("id")),
        team_name=away_team_raw.get("name"),
        team_abbr=away_team_raw.get("abbreviation"),
    )
    venue = None
    if venue_raw.get("id") is not None:
        venue = Venue(
            sport=SPORT,
            venue_id=str(venue_raw.get("id")),
            venue_name=venue_raw.get("name"),
        )

    away_pitcher_id = (away.get("probablePitcher") or {}).get("id")
    home_pitcher_id = (home.get("probablePitcher") or {}).get("id")

    raw_summary = {
        "game_id": raw["gamePk"],
        "game_time": raw.get("gameDate", ""),
        "status": raw.get("status", {}).get("detailedState", ""),
        "away_team_id": away_team_raw.get("id"),
        "away_team_name": away_team_raw.get("name"),
        "home_team_id": home_team_raw.get("id"),
        "home_team_name": home_team_raw.get("name"),
        "venue_id": venue_raw.get("id"),
        "venue_name": venue_raw.get("name"),
        "away_pitcher_id": away_pitcher_id,
        "away_pitcher_name": (away.get("probablePitcher") or {}).get("fullName", "TBD"),
        "home_pitcher_id": home_pitcher_id,
        "home_pitcher_name": (home.get("probablePitcher") or {}).get("fullName", "TBD"),
    }

    return Game(
        sport=SPORT,
        game_id=str(raw["gamePk"]),
        game_date=game_date,
        status=raw.get("status", {}).get("detailedState"),
        home_team=home_team,
        away_team=away_team,
        venue=venue,
        home_score=None,
        away_score=None,
        raw=raw_summary,
    )


def _extract_lineup_entries(game: Game, boxscore_payload: dict[str, Any]) -> list[LineupEntry]:
    rows: list[LineupEntry] = []
    teams = boxscore_payload.get("teams", {})
    for side in ("home", "away"):
        team_block = teams.get(side, {})
        team_id = (team_block.get("team") or {}).get("id")
        if team_id is None:
            continue
        team_id_s = str(team_id)
        order_ids = team_block.get("battingOrder", []) or []
        order_map = {pid: idx + 1 for idx, pid in enumerate(order_ids)}
        for player in (team_block.get("players") or {}).values():
            person = player.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            position = (player.get("position") or {}).get("abbreviation")
            rows.append(LineupEntry(
                sport=SPORT,
                game_id=game.game_id,
                team_id=team_id_s,
                player=Player(
                    sport=SPORT,
                    player_id=str(pid),
                    player_name=person.get("fullName"),
                    primary_position=position,
                ),
                batting_order=order_map.get(pid),
                starter_slot=None,
                position=position,
                source="official_lineup" if order_map else "roster",
            ))
    return rows


def _extract_lineup_entries_from_live(game: Game, payload: dict[str, Any]) -> list[LineupEntry]:
    """Live feed embeds the boxscore inside `liveData.boxscore`."""
    box = payload.get("liveData", {}).get("boxscore")
    if not box:
        return []
    return _extract_lineup_entries(game, box)


def _extract_boxscore_entries(game: Game, payload: dict[str, Any]) -> list[BoxscoreEntry]:
    rows: list[BoxscoreEntry] = []
    box = payload.get("liveData", {}).get("boxscore", {})
    teams = box.get("teams", {})
    for side in ("home", "away"):
        team_block = teams.get(side, {})
        team_id = (team_block.get("team") or {}).get("id")
        if team_id is None:
            continue
        team_id_s = str(team_id)
        for player in (team_block.get("players") or {}).values():
            person = player.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            position = (player.get("position") or {}).get("abbreviation")
            rows.append(BoxscoreEntry(
                sport=SPORT,
                game_id=game.game_id,
                team_id=team_id_s,
                player=Player(
                    sport=SPORT,
                    player_id=str(pid),
                    player_name=person.get("fullName"),
                    primary_position=position,
                ),
                position=position,
                stats={
                    "stats": player.get("stats", {}),
                    "seasonStats": player.get("seasonStats", {}),
                },
            ))
    return rows


def _update_game_with_final(game: Game, payload: dict[str, Any]) -> Game:
    live = payload.get("liveData") or {}
    linescore = live.get("linescore") or {}
    status = (payload.get("gameData") or {}).get("status", {}).get("detailedState") or game.status
    teams_score = linescore.get("teams") or {}
    home_runs = (teams_score.get("home") or {}).get("runs")
    away_runs = (teams_score.get("away") or {}).get("runs")
    if home_runs is None and away_runs is None and status == game.status:
        return game
    return Game(
        sport=game.sport,
        game_id=game.game_id,
        game_date=game.game_date,
        status=status,
        home_team=game.home_team,
        away_team=game.away_team,
        venue=game.venue,
        home_score=home_runs if home_runs is not None else game.home_score,
        away_score=away_runs if away_runs is not None else game.away_score,
        raw=game.raw,
    )


__all__ = ["MLBStatsSource"]
