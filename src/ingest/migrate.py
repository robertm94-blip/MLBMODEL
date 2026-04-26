"""One-shot import of legacy JSON files into the SQLite ingest DB.

Reads:
- data/results_*.json   -> games (sport='mlb')
- lineups_*.json (root) -> games + lineups (sport='mlb')
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import sqlite3
from typing import Any

from src.ingest import db

log = logging.getLogger(__name__)

SPORT = "mlb"

_LINEUP_FILE_RE = re.compile(r"lineups_(\d{4})_(\d{2})_(\d{2})\.json$")
_RESULTS_FILE_RE = re.compile(r"results_(\d{4})\.json$")


def migrate_all(conn: sqlite3.Connection, repo_root: str = ".") -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    out["results"] = migrate_results(conn, repo_root)
    out["lineups"] = migrate_lineups(conn, repo_root)
    return out


def migrate_results(conn: sqlite3.Connection, repo_root: str = ".") -> tuple[int, int]:
    pattern = os.path.join(repo_root, "data", "results_*.json")
    files = sorted(glob.glob(pattern))
    inserted = updated = 0
    for path in files:
        if not _RESULTS_FILE_RE.search(os.path.basename(path)):
            continue
        with open(path) as fh:
            results = json.load(fh)
        for game in results:
            was_new = db.upsert_game(
                conn,
                {
                    "sport": SPORT,
                    "game_id": game["game_id"],
                    "game_date": game["date"],
                    "status": game.get("status"),
                    "home_team_id": game.get("home_team_id"),
                    "home_team_name": game.get("home_team_name"),
                    "away_team_id": game.get("away_team_id"),
                    "away_team_name": game.get("away_team_name"),
                    "venue": game.get("venue_id"),
                    "home_score": game.get("home_score"),
                    "away_score": game.get("away_score"),
                    "raw": game,
                },
            )
            inserted += was_new
            updated += 1 - was_new
        conn.commit()
        log.info("migrated results file %s", path)
    return inserted, updated


def migrate_lineups(conn: sqlite3.Connection, repo_root: str = ".") -> tuple[int, int]:
    pattern = os.path.join(repo_root, "lineups_*.json")
    files = sorted(glob.glob(pattern))
    inserted = updated = 0
    for path in files:
        m = _LINEUP_FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        with open(path) as fh:
            payload = json.load(fh)
        rows: list[dict[str, Any]] = []
        for game_id, game in payload.get("games", {}).items():
            db.upsert_game(
                conn,
                {
                    "sport": SPORT,
                    "game_id": game_id,
                    "game_date": date,
                    "status": game.get("status"),
                    "home_team_id": game.get("home_team_id"),
                    "home_team_name": game.get("home_team"),
                    "away_team_id": game.get("away_team_id"),
                    "away_team_name": game.get("away_team"),
                    "venue": None,
                    "home_score": None,
                    "away_score": None,
                    "raw": game,
                },
            )
            for side in ("away", "home"):
                lineup = game.get(f"{side}_lineup") or []
                team_id = game.get(f"{side}_team_id")
                source = game.get(f"{side}_lineup_source")
                if team_id is None:
                    continue
                for entry in lineup:
                    rows.append({
                        "sport": SPORT,
                        "game_id": game_id,
                        "team_id": team_id,
                        "player_id": entry["id"],
                        "player_name": entry.get("name"),
                        "position": entry.get("position"),
                        "batting_order": entry.get("batting_order"),
                        "starter_slot": None,
                        "source": source,
                    })
        ins, upd = db.upsert_lineup_rows(conn, rows)
        inserted += ins
        updated += upd
        conn.commit()
        log.info("migrated lineup file %s (+%d/~%d)", path, ins, upd)
    return inserted, updated
