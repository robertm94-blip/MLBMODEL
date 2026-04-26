"""CLI for the data_ingestion package.

Subcommands:
    mlb       --date YYYY-MM-DD | --backfill START:END
    nba       --date YYYY-MM-DD | --backfill START:END
    all       (mlb + nba)
    migrate   (one-shot import of legacy JSON files into the SQLite store)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

from src.data_ingestion import migrate as migrate_mod
from src.data_ingestion.base import DataSource
from src.data_ingestion.http import HttpClient
from src.data_ingestion.mlb_stats import MLBStatsSource
from src.data_ingestion.nba_stats import NBAStatsSource
from src.data_ingestion.storage import DEFAULT_DB_PATH, SQLiteStore

log = logging.getLogger("data_ingestion")


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


def _ingest_one_date(source: DataSource, store: SQLiteStore, target_date: str) -> None:
    sport = source.sport
    if (
        store.date_already_ingested(sport, "lineups", target_date)
        and store.date_already_ingested(sport, "box_scores", target_date)
    ):
        log.info("[%s %s] already ingested; skipping", sport, target_date)
        return

    schedule_run = store.record_run_start(sport, "schedule", target_date)
    try:
        games = source.fetch_schedule(target_date)
        sched_inserts = sum(int(store.upsert_game(g)) for g in games)
        store.conn.commit()
        store.record_run_finish(schedule_run, sched_inserts, len(games) - sched_inserts)
        log.info("[%s %s] %d games on schedule", sport, target_date, len(games))
    except Exception as exc:
        store.record_run_finish(schedule_run, 0, 0, error=str(exc))
        raise

    lineup_run = store.record_run_start(sport, "lineups", target_date)
    box_run = store.record_run_start(sport, "box_scores", target_date)
    lineup_ins = lineup_upd = 0
    box_ins = box_upd = 0
    try:
        for game in games:
            result = source.fetch_boxscore(game)
            store.upsert_game(result.game)
            if result.lineups:
                ins, upd = store.upsert_lineups(result.lineups)
                lineup_ins += ins
                lineup_upd += upd
            else:
                # Live feed didn't carry lineups (pre-game). Try the lineup endpoint.
                fallback = source.fetch_lineups(game)
                if fallback:
                    ins, upd = store.upsert_lineups(fallback)
                    lineup_ins += ins
                    lineup_upd += upd
            if result.box_scores:
                ins, upd = store.upsert_boxscores(result.box_scores)
                box_ins += ins
                box_upd += upd
        store.conn.commit()
        store.record_run_finish(lineup_run, lineup_ins, lineup_upd)
        store.record_run_finish(box_run, box_ins, box_upd)
        log.info(
            "[%s %s] lineups +%d ~%d  box +%d ~%d",
            sport, target_date, lineup_ins, lineup_upd, box_ins, box_upd,
        )
    except Exception as exc:
        store.record_run_finish(lineup_run, lineup_ins, lineup_upd, error=str(exc))
        store.record_run_finish(box_run, box_ins, box_upd, error=str(exc))
        raise


def _build_source(name: str, http: HttpClient) -> DataSource:
    if name == "mlb":
        return MLBStatsSource(http=http)
    if name == "nba":
        return NBAStatsSource(http=http)
    raise ValueError(f"Unknown source: {name}")


def cmd_sport(args: argparse.Namespace, sports: list[str]) -> int:
    store = SQLiteStore(args.db)
    http = HttpClient()
    targets: list[str]
    if args.backfill:
        start, end = _parse_range(args.backfill)
        targets = [d.isoformat() for d in _daterange(start, end)]
    else:
        targets = [args.date or date.today().isoformat()]

    for target_date in targets:
        for sport in sports:
            source = _build_source(sport, http)
            _ingest_one_date(source, store, target_date)
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    store = SQLiteStore(args.db)
    counts = migrate_mod.migrate_all(store, repo_root=args.repo_root)
    for label, (ins, upd) in counts.items():
        log.info("migrated %s: +%d ~%d", label, ins, upd)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sport data ingestion (MLB + NBA -> SQLite)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("mlb", "nba", "all"):
        p = sub.add_parser(cmd, help=f"Ingest {cmd.upper()} schedule + lineups + box scores")
        grp = p.add_mutually_exclusive_group()
        grp.add_argument("--date", help="Single date YYYY-MM-DD (default: today)")
        grp.add_argument("--backfill", help="Range START:END (YYYY-MM-DD:YYYY-MM-DD), idempotent")

    pm = sub.add_parser("migrate", help="Import legacy data/results_*.json + lineups_*.json")
    pm.add_argument("--repo-root", default=".", help="Repo root containing data/ and lineups_*.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "migrate":
        return cmd_migrate(args)
    sports = ["mlb", "nba"] if args.command == "all" else [args.command]
    return cmd_sport(args, sports)


if __name__ == "__main__":
    sys.exit(main())
