"""MLB ingestion: schedule, lineups, and box scores via the MLB Stats API."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from src.ingest import db
from src.ingest.http import HttpClient
from src.mlb_api import get_schedule

log = logging.getLogger(__name__)

SPORT = "mlb"
BASE_URL = "https://statsapi.mlb.com/api/v1"
LIVE_BASE_URL = "https://statsapi.mlb.com/api/v1.1"


def _ingest_schedule(conn: sqlite3.Connection, date: str) -> tuple[list[dict[str, Any]], int, int]:
    games = get_schedule(date)
    inserted = updated = 0
    for game in games:
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
                "venue": game.get("venue_name"),
                "home_score": None,
                "away_score": None,
                "raw": game,
            },
        )
        inserted += was_new
        updated += 1 - was_new
    conn.commit()
    return games, inserted, updated


def ingest_lineups(conn: sqlite3.Connection, date: str, http: HttpClient) -> tuple[int, int]:
    """Pull boxscore for every scheduled game on `date` and upsert lineup rows."""
    run_id = db.record_run_start(conn, SPORT, "lineups", date)
    try:
        games, _, _ = _ingest_schedule(conn, date)
        total_inserted = total_updated = 0
        for game in games:
            game_id = game["game_id"]
            try:
                payload = http.get_json(
                    f"{BASE_URL}/game/{game_id}/boxscore",
                    host_key="mlb",
                )
            except Exception as exc:
                log.warning("boxscore fetch failed for game %s: %s", game_id, exc)
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
    """Pull live feed for every game on `date` and upsert box-score rows + final scores."""
    run_id = db.record_run_start(conn, SPORT, "box_scores", date)
    try:
        games, _, _ = _ingest_schedule(conn, date)
        total_inserted = total_updated = 0
        for game in games:
            game_id = game["game_id"]
            try:
                payload = http.get_json(
                    f"{LIVE_BASE_URL}/game/{game_id}/feed/live",
                    host_key="mlb",
                )
            except Exception as exc:
                log.warning("live feed fetch failed for game %s: %s", game_id, exc)
                continue
            rows, final = _extract_boxscore_rows(game_id, payload)
            ins, upd = db.upsert_boxscore_rows(conn, rows)
            total_inserted += ins
            total_updated += upd
            if final is not None:
                db.upsert_game(
                    conn,
                    {
                        "sport": SPORT,
                        "game_id": game_id,
                        "game_date": date,
                        "status": final["status"],
                        "home_team_id": game.get("home_team_id"),
                        "home_team_name": game.get("home_team_name"),
                        "away_team_id": game.get("away_team_id"),
                        "away_team_name": game.get("away_team_name"),
                        "venue": game.get("venue_name"),
                        "home_score": final["home_score"],
                        "away_score": final["away_score"],
                        "raw": game,
                    },
                )
        conn.commit()
        db.record_run_finish(conn, run_id, total_inserted, total_updated)
        return total_inserted, total_updated
    except Exception as exc:
        db.record_run_finish(conn, run_id, 0, 0, error=str(exc))
        raise


def _extract_lineup_rows(game_id: int, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    teams = payload.get("teams", {})
    for side in ("home", "away"):
        team_block = teams.get(side, {})
        team_id = team_block.get("team", {}).get("id")
        if team_id is None:
            continue
        players = team_block.get("players", {})
        batting_order_ids = team_block.get("battingOrder", []) or []
        order_map = {pid: idx + 1 for idx, pid in enumerate(batting_order_ids)}
        for key, player in players.items():
            person = player.get("person", {})
            pid = person.get("id")
            if pid is None:
                continue
            position = (player.get("position") or {}).get("abbreviation")
            rows.append({
                "sport": SPORT,
                "game_id": game_id,
                "team_id": team_id,
                "player_id": pid,
                "player_name": person.get("fullName"),
                "position": position,
                "batting_order": order_map.get(pid),
                "starter_slot": None,
                "source": "official" if order_map else "roster",
            })
    return rows


def _extract_boxscore_rows(
    game_id: int, payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    box = payload.get("liveData", {}).get("boxscore", {})
    teams = box.get("teams", {})
    for side in ("home", "away"):
        team_block = teams.get(side, {})
        team_id = team_block.get("team", {}).get("id")
        if team_id is None:
            continue
        for key, player in team_block.get("players", {}).items():
            person = player.get("person", {})
            pid = person.get("id")
            if pid is None:
                continue
            rows.append({
                "sport": SPORT,
                "game_id": game_id,
                "team_id": team_id,
                "player_id": pid,
                "stats": {
                    "name": person.get("fullName"),
                    "position": (player.get("position") or {}).get("abbreviation"),
                    "stats": player.get("stats", {}),
                    "seasonStats": player.get("seasonStats", {}),
                },
            })
    final: dict[str, Any] | None = None
    linescore = payload.get("liveData", {}).get("linescore", {})
    status = payload.get("gameData", {}).get("status", {}).get("detailedState")
    if linescore:
        final = {
            "status": status,
            "home_score": linescore.get("teams", {}).get("home", {}).get("runs"),
            "away_score": linescore.get("teams", {}).get("away", {}).get("runs"),
        }
    return rows, final
