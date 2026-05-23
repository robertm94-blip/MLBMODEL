#!/usr/bin/env python3
"""One-command daily runner: ingest -> features -> project (logs) -> grade.

Chains the full pipeline so a single invocation both produces today's
projections AND advances the forward-test record:

    1. ingest MLB schedule + lineups + box scores for the date
    2. build team features (writes team_features)
    3. run the projection engine (writes CSV + banks predictions to prediction_log)
    4. grade every ungraded final (catches today's completed games + any
       prior days that have since finished)

Run it once a day (ideally a few hours before first pitch for fresh lineups,
then again after the slate completes to grade). Idempotent throughout.

Examples:
    python daily.py                    # today
    python daily.py --date 2026-05-14
    python daily.py --date 2026-05-14 --no-weather
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import data_ingestion as ingest_cli
import feature_engineer as feat_cli
import grade_predictions as grade_cli
import projection_engine as proj_cli

log = logging.getLogger("daily")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Daily MLB pipeline: ingest -> features -> project -> grade")
    p.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (default today)")
    p.add_argument("--db", default=None, help="SQLite path (defaults to the module default)")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    d = args.date
    db_args = ["--db", args.db] if args.db else []

    log.info("=== [1/4] ingest MLB %s ===", d)
    ingest_cli.main(["--log-level", args.log_level, *db_args, "mlb", "--date", d])

    log.info("=== [2/4] feature engineering %s ===", d)
    feat_cli.main([*db_args, "--date", d, "--write-db", "--log-level", args.log_level])

    log.info("=== [3/4] projections %s (banking predictions) ===", d)
    proj_argv = [*db_args, "--date", d, "--log-level", args.log_level]
    if args.no_weather:
        proj_argv.append("--no-weather")
    proj_cli.main(proj_argv)

    log.info("=== [4/4] grading finals ===")
    grade_cli.main([*db_args, "--refresh", "--date", d, "--log-level", args.log_level])

    log.info("daily run complete for %s. Run track_record.py to see the cumulative record.", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
