"""One-shot migrator that imports legacy JSON files into the SQLite store.

Reads the historical artifacts that pre-date this module:
- data/results_*.json   -> games (sport='mlb')
- lineups_*.json (root) -> games + lineups (sport='mlb')
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Any

from src.data_ingestion.base import (
    Game,
    LineupEntry,
    Player,
    Team,
    Venue,
)
from src.data_ingestion.storage import SQLiteStore

log = logging.getLogger(__name__)

SPORT = "mlb"
_LINEUP_FILE_RE = re.compile(r"lineups_(\d{4})_(\d{2})_(\d{2})\.json$")
_RESULTS_FILE_RE = re.compile(r"results_(\d{4})\.json$")


def migrate_all(store: SQLiteStore, repo_root: str = ".") -> dict[str, tuple[int, int]]:
    return {
        "results": migrate_results(store, repo_root),
        "lineups": migrate_lineups(store, repo_root),
    }


def migrate_results(store: SQLiteStore, repo_root: str = ".") -> tuple[int, int]:
    files = sorted(glob.glob(os.path.join(repo_root, "data", "results_*.json")))
    inserted = updated = 0
    for path in files:
        if not _RESULTS_FILE_RE.search(os.path.basename(path)):
            continue
        with open(path) as fh:
            results = json.load(fh)
        for raw in results:
            game = _result_to_game(raw)
            new = store.upsert_game(game)
            inserted += int(new)
            updated += int(not new)
        store.conn.commit()
        log.info("migrated %s (%d games)", path, len(results))
    return inserted, updated


def migrate_lineups(store: SQLiteStore, repo_root: str = ".") -> tuple[int, int]:
    files = sorted(glob.glob(os.path.join(repo_root, "lineups_*.json")))
    inserted = updated = 0
    for path in files:
        m = _LINEUP_FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        game_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        with open(path) as fh:
            payload = json.load(fh)
        rows: list[LineupEntry] = []
        for game_id, raw in (payload.get("games") or {}).items():
            game = _lineup_payload_to_game(game_id, raw, game_date)
            store.upsert_game(game)
            for side in ("away", "home"):
                team_id = raw.get(f"{side}_team_id")
                source = raw.get(f"{side}_lineup_source")
                if team_id is None:
                    continue
                team_id_s = str(team_id)
                for entry in (raw.get(f"{side}_lineup") or []):
                    pid = entry.get("id")
                    if pid is None:
                        continue
                    rows.append(LineupEntry(
                        sport=SPORT,
                        game_id=str(game_id),
                        team_id=team_id_s,
                        player=Player(
                            sport=SPORT,
                            player_id=str(pid),
                            player_name=entry.get("name"),
                            primary_position=entry.get("position"),
                        ),
                        batting_order=entry.get("batting_order"),
                        starter_slot=None,
                        position=entry.get("position"),
                        source=source,
                    ))
        ins, upd = store.upsert_lineups(rows)
        inserted += ins
        updated += upd
        store.conn.commit()
        log.info("migrated %s (+%d/~%d)", path, ins, upd)
    return inserted, updated


def _result_to_game(raw: dict[str, Any]) -> Game:
    home = Team(
        sport=SPORT, team_id=str(raw.get("home_team_id")),
        team_name=raw.get("home_team_name"),
    )
    away = Team(
        sport=SPORT, team_id=str(raw.get("away_team_id")),
        team_name=raw.get("away_team_name"),
    )
    venue = None
    if raw.get("venue_id") is not None:
        venue = Venue(sport=SPORT, venue_id=str(raw["venue_id"]))
    return Game(
        sport=SPORT,
        game_id=str(raw["game_id"]),
        game_date=raw["date"],
        status=raw.get("status"),
        home_team=home,
        away_team=away,
        venue=venue,
        home_score=raw.get("home_score"),
        away_score=raw.get("away_score"),
        raw=raw,
    )


def _lineup_payload_to_game(game_id: str, raw: dict[str, Any], game_date: str) -> Game:
    home = Team(
        sport=SPORT, team_id=str(raw.get("home_team_id")),
        team_name=raw.get("home_team"),
    )
    away = Team(
        sport=SPORT, team_id=str(raw.get("away_team_id")),
        team_name=raw.get("away_team"),
    )
    return Game(
        sport=SPORT,
        game_id=str(game_id),
        game_date=game_date,
        status=raw.get("status"),
        home_team=home,
        away_team=away,
        venue=None,
        home_score=None,
        away_score=None,
        raw=raw,
    )


__all__ = ["migrate_all", "migrate_results", "migrate_lineups"]
