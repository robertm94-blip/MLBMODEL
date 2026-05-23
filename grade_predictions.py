#!/usr/bin/env python3
"""Settle logged predictions against final scores.

Walks the prediction_log, finds games that are now Final in the `games`
table (refreshing from the MLB Stats API first if --refresh is passed), and
fills in actual scores + grading metrics (side correctness, Brier, totals
error). Idempotent: re-running recomputes from current values.

Examples:
    python grade_predictions.py                 # grade all ungraded finals
    python grade_predictions.py --date 2026-05-14
    python grade_predictions.py --date 2026-05-14 --refresh   # re-pull finals first
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.data_ingestion.http import HttpClient
from src.data_ingestion.mlb_stats import MLBStatsSource
from src.data_ingestion.storage import DEFAULT_DB_PATH, SQLiteStore

log = logging.getLogger("grade_predictions")

FINAL_STATES = {"Final", "Game Over", "Completed Early"}


def _refresh_finals(store: SQLiteStore, game_date: str) -> None:
    """Re-pull box scores so final scores in `games` are current."""
    src = MLBStatsSource(http=HttpClient())
    rows = store.conn.execute(
        "SELECT game_id, home_team_id, away_team_id, status FROM games "
        "WHERE sport='mlb' AND game_date=?",
        (game_date,),
    ).fetchall()
    from src.data_ingestion.base import Game, Team
    for r in rows:
        g = Game(sport="mlb", game_id=r["game_id"], game_date=game_date, status=r["status"],
                 home_team=Team(sport="mlb", team_id=r["home_team_id"]),
                 away_team=Team(sport="mlb", team_id=r["away_team_id"]))
        result = src.fetch_boxscore(g)
        store.upsert_game(result.game)
    store.conn.commit()


def grade(store: SQLiteStore, game_date: str | None, refresh: bool) -> tuple[int, int]:
    if refresh and game_date:
        _refresh_finals(store, game_date)

    # Pull final scores for games that have logged predictions.
    q = """
        SELECT g.game_id, g.status, g.home_score, g.away_score
        FROM games g
        JOIN prediction_log p ON p.sport=g.sport AND p.game_id=g.game_id
        WHERE g.sport='mlb' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    """
    params: list = []
    if game_date:
        q += " AND g.game_date=?"
        params.append(game_date)
    rows = store.conn.execute(q, params).fetchall()

    graded = skipped = 0
    for r in rows:
        if r["status"] not in FINAL_STATES:
            skipped += 1
            continue
        if store.grade_prediction("mlb", r["game_id"], r["away_score"], r["home_score"]):
            graded += 1
    store.conn.commit()
    return graded, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Grade logged predictions vs final scores")
    p.add_argument("--db", default=DEFAULT_DB_PATH)
    p.add_argument("--date", help="Restrict to one date YYYY-MM-DD")
    p.add_argument("--refresh", action="store_true",
                   help="Re-pull box scores from MLB Stats before grading")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = SQLiteStore(args.db)
    graded, skipped = grade(store, args.date, args.refresh)
    log.info("graded %d predictions (%d games not yet final)", graded, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
