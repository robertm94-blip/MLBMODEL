"""CLI entry point for the data ingestion module.

Examples:
    python ingest.py mlb --date 2026-04-26
    python ingest.py nba --date 2026-04-26
    python ingest.py all --date 2026-04-26
    python ingest.py mlb --backfill 2025-03-30:2025-10-01
    python ingest.py nba --backfill 2025-10-22:2026-04-13
    python ingest.py migrate
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

from src.ingest import db
from src.ingest.http import HttpClient
from src.ingest import mlb as mlb_ingest
from src.ingest import nba as nba_ingest
from src.ingest import migrate as migrate_mod

log = logging.getLogger("ingest")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_range(value: str) -> tuple[date, date]:
    start_str, _, end_str = value.partition(":")
    if not end_str:
        raise argparse.ArgumentTypeError("--backfill must be START:END (YYYY-MM-DD:YYYY-MM-DD)")
    start, end = _parse_date(start_str), _parse_date(end_str)
    if end < start:
        raise argparse.ArgumentTypeError("--backfill END must be >= START")
    return start, end


def _daterange(start: date, end: date):
    cur = start
    one = timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


SPORT_MODULES = {
    "mlb": mlb_ingest,
    "nba": nba_ingest,
}


def _run_sport(sport: str, target_date: str, http: HttpClient, conn) -> None:
    module = SPORT_MODULES[sport]
    log.info("[%s %s] lineups...", sport, target_date)
    ins_l, upd_l = module.ingest_lineups(conn, target_date, http)
    log.info("[%s %s] lineups +%d updated %d", sport, target_date, ins_l, upd_l)
    log.info("[%s %s] box scores...", sport, target_date)
    ins_b, upd_b = module.ingest_boxscores(conn, target_date, http)
    log.info("[%s %s] box_scores +%d updated %d", sport, target_date, ins_b, upd_b)


def cmd_sport(args: argparse.Namespace, sports: list[str]) -> int:
    conn = db.connect(args.db)
    http = HttpClient()
    if args.backfill:
        start, end = _parse_range(args.backfill)
        for d in _daterange(start, end):
            target_date = d.isoformat()
            for sport in sports:
                if (
                    db.date_already_ingested(conn, sport, "lineups", target_date)
                    and db.date_already_ingested(conn, sport, "box_scores", target_date)
                ):
                    log.info("[%s %s] already ingested, skipping", sport, target_date)
                    continue
                _run_sport(sport, target_date, http, conn)
    else:
        target_date = (args.date or date.today().isoformat())
        for sport in sports:
            _run_sport(sport, target_date, http, conn)
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    counts = migrate_mod.migrate_all(conn, repo_root=args.repo_root)
    for label, (ins, upd) in counts.items():
        log.info("migrated %s: +%d updated %d", label, ins, upd)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLB + NBA data ingestion")
    parser.add_argument(
        "--db", default=db.DEFAULT_DB_PATH, help="SQLite path (default: data/ingest.db)"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("mlb", "nba", "all"):
        p = sub.add_parser(cmd, help=f"Ingest {cmd.upper()} lineups + box scores")
        group = p.add_mutually_exclusive_group()
        group.add_argument("--date", help="Single date YYYY-MM-DD (default: today)")
        group.add_argument(
            "--backfill",
            help="Date range START:END (YYYY-MM-DD:YYYY-MM-DD), idempotent and resumable",
        )

    sub.add_parser("migrate", help="Import legacy data/results_*.json + lineups_*.json into the DB")
    for action in sub.choices["migrate"]._actions:
        pass
    sub.choices["migrate"].add_argument(
        "--repo-root", default=".", help="Repo root containing data/ and lineups_*.json"
    )

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
