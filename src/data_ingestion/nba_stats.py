"""NBA Stats API source (stats.nba.com).

Implements the same `DataSource` contract as `MLBStatsSource`. The full
parser implementation is intentionally brief here — MLB is the
critical source for Phase 1 and was implemented end-to-end. NBA fetches
the schedule and box score; lineups are derived from the boxscore's
position field (starters have a position set).
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

SPORT = "nba"
SCOREBOARD_URL = "https://stats.nba.com/stats/scoreboardv3"
BOXSCORE_URL = "https://stats.nba.com/stats/boxscoretraditionalv3"


def _format_nba_date(iso_date: str) -> str:
    y, m, d = iso_date.split("-")
    return f"{int(m):02d}/{int(d):02d}/{int(y):04d}"


class NBAStatsSource(DataSource):
    """Pull NBA schedule, lineups (starters), and box scores from stats.nba.com."""

    sport = SPORT

    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient()

    def fetch_schedule(self, game_date: str) -> list[Game]:
        payload = self.http.get_json(
            SCOREBOARD_URL,
            host_key="nba",
            params={"GameDate": _format_nba_date(game_date), "LeagueID": "00"},
        )
        out: list[Game] = []
        scoreboard = payload.get("scoreboard", {})
        for raw in scoreboard.get("games", []):
            home = raw.get("homeTeam", {}) or {}
            away = raw.get("awayTeam", {}) or {}
            arena = raw.get("arena") or {}
            home_team = Team(
                sport=SPORT, team_id=str(home.get("teamId")),
                team_name=f"{home.get('teamCity', '')} {home.get('teamName', '')}".strip(),
                team_abbr=home.get("teamTricode"),
            )
            away_team = Team(
                sport=SPORT, team_id=str(away.get("teamId")),
                team_name=f"{away.get('teamCity', '')} {away.get('teamName', '')}".strip(),
                team_abbr=away.get("teamTricode"),
            )
            venue = None
            arena_id = arena.get("arenaId")
            if arena_id is not None:
                venue = Venue(sport=SPORT, venue_id=str(arena_id), venue_name=arena.get("arenaName"))
            out.append(Game(
                sport=SPORT,
                game_id=str(raw.get("gameId")),
                game_date=game_date,
                status=raw.get("gameStatusText"),
                home_team=home_team,
                away_team=away_team,
                venue=venue,
                home_score=home.get("score"),
                away_score=away.get("score"),
                raw=raw,
            ))
        return out

    def fetch_lineups(self, game: Game) -> list[LineupEntry]:
        payload = self._fetch_box_payload(game)
        if payload is None:
            return []
        return _extract_starters(game, payload)

    def fetch_boxscore(self, game: Game) -> FetchResult:
        payload = self._fetch_box_payload(game)
        if payload is None:
            return FetchResult(game=game)
        return FetchResult(
            game=game,
            lineups=_extract_starters(game, payload),
            box_scores=_extract_box_entries(game, payload),
        )

    def _fetch_box_payload(self, game: Game) -> dict[str, Any] | None:
        try:
            return self.http.get_json(
                BOXSCORE_URL,
                host_key="nba",
                params={
                    "GameID": game.game_id,
                    "LeagueID": "00",
                    "endPeriod": 0,
                    "startPeriod": 0,
                    "endRange": 28800,
                    "startRange": 0,
                    "rangeType": 0,
                },
            )
        except Exception as exc:
            log.warning("NBA boxscore fetch failed for %s: %s", game.game_id, exc)
            return None


def _iter_team_blocks(payload: dict[str, Any]):
    box = payload.get("boxScoreTraditional", {})
    for side_key in ("homeTeam", "awayTeam"):
        team = box.get(side_key, {})
        team_id = team.get("teamId")
        if team_id is None:
            continue
        yield str(team_id), team.get("players", []) or []


def _extract_starters(game: Game, payload: dict[str, Any]) -> list[LineupEntry]:
    rows: list[LineupEntry] = []
    for team_id, players in _iter_team_blocks(payload):
        slot = 0
        for p in players:
            if not p.get("position"):
                continue  # bench player
            slot += 1
            pid = p.get("personId")
            if pid is None:
                continue
            name = p.get("nameI") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
            rows.append(LineupEntry(
                sport=SPORT,
                game_id=game.game_id,
                team_id=team_id,
                player=Player(sport=SPORT, player_id=str(pid), player_name=name,
                              primary_position=p.get("position")),
                batting_order=None,
                starter_slot=slot,
                position=p.get("position"),
                source="official",
            ))
    return rows


def _extract_box_entries(game: Game, payload: dict[str, Any]) -> list[BoxscoreEntry]:
    rows: list[BoxscoreEntry] = []
    for team_id, players in _iter_team_blocks(payload):
        for p in players:
            pid = p.get("personId")
            if pid is None:
                continue
            name = p.get("nameI") or f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
            rows.append(BoxscoreEntry(
                sport=SPORT,
                game_id=game.game_id,
                team_id=team_id,
                player=Player(sport=SPORT, player_id=str(pid), player_name=name,
                              primary_position=p.get("position")),
                position=p.get("position"),
                stats={
                    "starter": bool(p.get("position")),
                    "statistics": p.get("statistics", {}),
                },
            ))
    return rows


__all__ = ["NBAStatsSource"]
