"""NBA ingestion: schedule, lineups (starters), and box scores via stats.nba.com."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from src.ingest import db
from src.ingest.http import HttpClient

log = logging.getLogger(__name__)

SPORT = "nba"
SCOREBOARD_URL = "https://stats.nba.com/stats/scoreboardv3"
BOXSCORE_URL = "https://stats.nba.com/stats/boxscoretraditionalv3"


def _format_nba_date(iso_date: str) -> str:
    y, m, d = iso_date.split("-")
    return f"{int(m):02d}/{int(d):02d}/{int(y):04d}"


def _fetch_scoreboard(http: HttpClient, date: str) -> list[dict[str, Any]]:
    payload = http.get_json(
        SCOREBOARD_URL,
        host_key="nba",
        params={"GameDate": _format_nba_date(date), "LeagueID": "00"},
    )
    games_out: list[dict[str, Any]] = []
    scoreboard = payload.get("scoreboard", {})
    for game in scoreboard.get("games", []):
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        games_out.append({
            "game_id": game.get("gameId"),
            "status": game.get("gameStatusText"),
            "home_team_id": home.get("teamId"),
            "home_team_name": f"{home.get('teamCity', '')} {home.get('teamName', '')}".strip(),
            "away_team_id": away.get("teamId"),
            "away_team_name": f"{away.get('teamCity', '')} {away.get('teamName', '')}".strip(),
            "venue": (game.get("arena") or {}).get("arenaName"),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "raw": game,
        })
    return games_out


def _ingest_schedule(
    conn: sqlite3.Connection, date: str, http: HttpClient
) -> tuple[list[dict[str, Any]], int, int]:
    games = _fetch_scoreboard(http, date)
    inserted = updated = 0
    for game in games:
        if game.get("game_id") is None:
            continue
        was_new = db.upsert_game(
            conn,
            {
                "sport": SPORT,
                "game_id": game["game_id"],
                "game_date": date,
                "status": game.get("status"),
                "home_team_id": game.get("home_team_id"),
                "home_team_name": game.get("home_team_name"),
                "away_team_id": game.get("away_team_id"),
                "away_team_name": game.get("away_team_name"),
                "venue": game.get("venue"),
                "home_score": game.get("home_score"),
                "away_score": game.get("away_score"),
                "raw": game.get("raw"),
            },
        )
        inserted += was_new
        updated += 1 - was_new
    conn.commit()
    return games, inserted, updated


def _fetch_boxscore(http: HttpClient, game_id: str) -> dict[str, Any]:
    return http.get_json(
        BOXSCORE_URL,
        host_key="nba",
        params={
            "GameID": game_id,
            "LeagueID": "00",
            "endPeriod": 0,
            "startPeriod": 0,
            "endRange": 28800,
            "startRange": 0,
            "rangeType": 0,
        },
    )


def ingest_lineups(conn: sqlite3.Connection, date: str, http: HttpClient) -> tuple[int, int]:
    """Pull starters for every NBA game on `date`. starter_slot 1..5 by appearance."""
    run_id = db.record_run_start(conn, SPORT, "lineups", date)
    try:
        games, _, _ = _ingest_schedule(conn, date, http)
        total_inserted = total_updated = 0
        for game in games:
            game_id = game.get("game_id")
            if game_id is None:
                continue
            try:
                payload = _fetch_boxscore(http, game_id)
            except Exception as exc:
                log.warning("nba boxscore fetch failed for %s: %s", game_id, exc)
                continue
            rows = _extract_lineup_rows(game_id, payload)
            ins, upd = db.upsert_lineup_rows(conn, rows)
            total_inserted += ins
            total_updated += upd
        conn.commit()
        db.record_run_finish(conn, run_id, total_inserted, total_updated)
        return total_inserted, total_updated
    except Exception as exc:
        db.record_run_finish(conn, run_id, 0, 0, error=str(exc))
        raise


def ingest_boxscores(conn: sqlite3.Connection, date: str, http: HttpClient) -> tuple[int, int]:
    run_id = db.record_run_start(conn, SPORT, "box_scores", date)
    try:
        games, _, _ = _ingest_schedule(conn, date, http)
        total_inserted = total_updated = 0
        for game in games:
            game_id = game.get("game_id")
            if game_id is None:
                continue
            try:
                payload = _fetch_boxscore(http, game_id)
            except Exception as exc:
                log.warning("nba boxscore fetch failed for %s: %s", game_id, exc)
                continue
            rows = _extract_boxscore_rows(game_id, payload)
            ins, upd = db.upsert_boxscore_rows(conn, rows)
            total_inserted += ins
            total_updated += upd
        conn.commit()
        db.record_run_finish(conn, run_id, total_inserted, total_updated)
        return total_inserted, total_updated
    except Exception as exc:
        db.record_run_finish(conn, run_id, 0, 0, error=str(exc))
        raise


def _iter_team_blocks(payload: dict[str, Any]):
    box = payload.get("boxScoreTraditional", {})
    for side_key in ("homeTeam", "awayTeam"):
        team = box.get(side_key, {})
        team_id = team.get("teamId")
        if team_id is None:
            continue
        yield team_id, team.get("players", []) or []


def _extract_lineup_rows(game_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_id, players in _iter_team_blocks(payload):
        starter_slot = 0
        for player in players:
            if player.get("position"):
                starter_slot += 1
                rows.append({
                    "sport": SPORT,
                    "game_id": game_id,
                    "team_id": team_id,
                    "player_id": player.get("personId"),
                    "player_name": (
                        player.get("nameI")
                        or f"{player.get('firstName', '')} {player.get('familyName', '')}".strip()
                    ),
                    "position": player.get("position"),
                    "batting_order": None,
                    "starter_slot": starter_slot,
                    "source": "official",
                })
    return rows


def _extract_boxscore_rows(game_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_id, players in _iter_team_blocks(payload):
        for player in players:
            pid = player.get("personId")
            if pid is None:
                continue
            rows.append({
                "sport": SPORT,
                "game_id": game_id,
                "team_id": team_id,
                "player_id": pid,
                "stats": {
                    "name": (
                        player.get("nameI")
                        or f"{player.get('firstName', '')} {player.get('familyName', '')}".strip()
                    ),
                    "position": player.get("position"),
                    "starter": bool(player.get("position")),
                    "statistics": player.get("statistics", {}),
                },
            })
    return rows
