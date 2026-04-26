"""CLI entry point for the MLB feature engineering pipeline.

Reads schedule + lineups from the SQLite store (`data/data_ingestion.db`),
falls back to legacy lineups_*.json or a live MLB Stats fetch when the
DB is empty for the date, composes per-team feature rows, and writes
them to the `team_features` table and/or data/features/{date}.json.

Examples:
    python feature_engineer.py --date 2026-03-30
    python feature_engineer.py --date 2026-03-30 --write-db
    python feature_engineer.py --backfill 2025-04-01:2025-04-30 --write-db
    python feature_engineer.py --game 824135 --date 2026-03-30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from typing import Any

from src.data_ingestion.storage import DEFAULT_DB_PATH, SQLiteStore
from src.feature_pipeline import build_game_features
from src.mlb_api import get_schedule

log = logging.getLogger("feature_engineer")

FEATURES_DIR = os.path.join("data", "features")
LEGACY_LINEUP_TEMPLATE = "lineups_{y}_{m:02d}_{d:02d}.json"


def _parse_range(value: str) -> tuple[date, date]:
    start_str, _, end_str = value.partition(":")
    if not end_str:
        raise argparse.ArgumentTypeError("--backfill must be START:END (YYYY-MM-DD:YYYY-MM-DD)")
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    if end < start:
        raise argparse.ArgumentTypeError("--backfill END must be >= START")
    return start, end


def _daterange(start: date, end: date):
    cur = start
    one = timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


def _load_schedule_from_store(store: SQLiteStore, target_date: str) -> list[dict[str, Any]]:
    games = store.query_games_for_date("mlb", target_date)
    out = []
    for g in games:
        raw = json.loads(g["raw_json"]) if g.get("raw_json") else {}
        out.append({
            "game_id": g["game_id"],
            "home_team_id": g.get("home_team_id"),
            "home_team_name": g.get("home_team_name") or raw.get("home_team_name"),
            "away_team_id": g.get("away_team_id"),
            "away_team_name": g.get("away_team_name") or raw.get("away_team_name"),
            "venue_id": g.get("venue_id") or raw.get("venue_id"),
            "home_pitcher_id": raw.get("home_pitcher_id"),
            "home_pitcher_name": raw.get("home_pitcher_name"),
            "away_pitcher_id": raw.get("away_pitcher_id"),
            "away_pitcher_name": raw.get("away_pitcher_name"),
        })
    return out


def _load_schedule_live(target_date: str) -> list[dict[str, Any]]:
    games = get_schedule(target_date)
    return [
        {
            "game_id": g["game_id"],
            "home_team_id": g["home_team_id"],
            "home_team_name": g["home_team_name"],
            "away_team_id": g["away_team_id"],
            "away_team_name": g["away_team_name"],
            "venue_id": g.get("venue_id"),
            "home_pitcher_id": g.get("home_pitcher_id"),
            "home_pitcher_name": g.get("home_pitcher_name"),
            "away_pitcher_id": g.get("away_pitcher_id"),
            "away_pitcher_name": g.get("away_pitcher_name"),
        }
        for g in games
    ]


def _load_lineups_from_store(
    store: SQLiteStore, target_date: str
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Returns {game_id: {team_id: [lineup row dicts...]}} from the SQLite store."""
    raw = store.query_lineups_for_date("mlb", target_date)
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for game_id, by_team in raw.items():
        for team_id, entries in by_team.items():
            for e in entries:
                if e.get("batting_order") is None:
                    continue
                out.setdefault(game_id, {}).setdefault(team_id, []).append({
                    "batting_order": e["batting_order"],
                    "id": e["id"],
                    "name": e["name"],
                    "position": e["position"],
                    "source": e["source"],
                })
    return out


def _load_lineups_legacy(target_date: str) -> dict[str, dict[str, dict[str, Any]]]:
    y, m, d = target_date.split("-")
    path = LEGACY_LINEUP_TEMPLATE.format(y=int(y), m=int(m), d=int(d))
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        payload = json.load(fh)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for game_id, game in payload.get("games", {}).items():
        out[str(game_id)] = {
            str(game["home_team_id"]): {
                "lineup": game.get("home_lineup") or [],
                "source": game.get("home_lineup_source"),
            },
            str(game["away_team_id"]): {
                "lineup": game.get("away_lineup") or [],
                "source": game.get("away_lineup_source"),
            },
        }
    return out


def _resolve_lineup(
    game_id: str,
    team_id: str,
    store_lineups: dict[str, dict[str, list[dict[str, Any]]]],
    legacy_lineups: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str | None]:
    db_lu = store_lineups.get(game_id, {}).get(team_id, [])
    if db_lu:
        return db_lu, db_lu[0].get("source")
    legacy = legacy_lineups.get(game_id, {}).get(team_id, {})
    return legacy.get("lineup", []) or [], legacy.get("source")


def process_date(
    target_date: str,
    store: SQLiteStore,
    write_db: bool,
    out_dir: str,
    only_game_id: str | None = None,
) -> tuple[int, int]:
    schedule = _load_schedule_from_store(store, target_date)
    if not schedule:
        log.info("[%s] no schedule in store, fetching from MLB Stats API", target_date)
        schedule = _load_schedule_live(target_date)
    if only_game_id:
        schedule = [g for g in schedule if str(g["game_id"]) == str(only_game_id)]

    store_lineups = _load_lineups_from_store(store, target_date)
    legacy_lineups = _load_lineups_legacy(target_date)

    output_rows: list[dict[str, Any]] = []
    inserts = 0
    for game in schedule:
        game_id = str(game["game_id"])
        home_team_id = str(game["home_team_id"])
        away_team_id = str(game["away_team_id"])
        home_lineup, home_src = _resolve_lineup(game_id, home_team_id, store_lineups, legacy_lineups)
        away_lineup, away_src = _resolve_lineup(game_id, away_team_id, store_lineups, legacy_lineups)
        features = build_game_features(
            game_id=game_id,
            game_date=target_date,
            venue_id=game.get("venue_id"),
            home={
                "team_id": home_team_id,
                "team_name": game["home_team_name"],
                "lineup": home_lineup,
                "starter_id": game.get("home_pitcher_id"),
                "starter_name": game.get("home_pitcher_name"),
                "lineup_source": home_src,
            },
            away={
                "team_id": away_team_id,
                "team_name": game["away_team_name"],
                "lineup": away_lineup,
                "starter_id": game.get("away_pitcher_id"),
                "starter_name": game.get("away_pitcher_name"),
                "lineup_source": away_src,
            },
        )
        for side in ("home", "away"):
            row = features[side]
            output_rows.append(row)
            if write_db:
                inserts += int(store.upsert_team_features(row))
        log.info(
            "[%s] %s @ %s: home wRC+=%s pitch_fip=%s | away wRC+=%s pitch_fip=%s | park=%s",
            target_date, game["away_team_name"], game["home_team_name"],
            features["home"]["lineup_wrc_plus"],
            features["home"]["pitching_blended_fip"],
            features["away"]["lineup_wrc_plus"],
            features["away"]["pitching_blended_fip"],
            features["home"]["park_runs_factor"],
        )

    if write_db:
        store.conn.commit()

    if output_rows:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{target_date}.json")
        with open(out_path, "w") as fh:
            json.dump({"date": target_date, "rows": output_rows}, fh, indent=2)
        log.info("[%s] wrote %d feature rows -> %s", target_date, len(output_rows), out_path)

    return len(output_rows), inserts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLB feature engineering pipeline")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--out-dir", default=FEATURES_DIR, help="JSON output dir (default: data/features)")
    parser.add_argument("--write-db", action="store_true", help="Upsert rows into team_features")
    parser.add_argument("--log-level", default="INFO")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--date", help="Single date YYYY-MM-DD (default: today)")
    grp.add_argument("--backfill", help="Range START:END (idempotent, resumable)")
    parser.add_argument("--game", help="Restrict to a single game_id within --date")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    store = SQLiteStore(args.db)

    if args.backfill:
        start, end = _parse_range(args.backfill)
        total_rows = total_inserts = 0
        for d in _daterange(start, end):
            rows, ins = process_date(d.isoformat(), store, args.write_db, args.out_dir)
            total_rows += rows
            total_inserts += ins
        log.info("backfill complete: %d rows, %d inserts", total_rows, total_inserts)
        return 0

    target = args.date or date.today().isoformat()
    rows, inserts = process_date(target, store, args.write_db, args.out_dir, only_game_id=args.game)
    log.info("date %s: %d rows, %d inserts", target, rows, inserts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
