"""CLI for the core projection engine.

Reads per-team feature rows from the SQLite `team_features` table
(produced by `feature_engineer.py`), produces per-game projections
(fair lines + projected totals), and writes a daily CSV.

Examples:
    python projection_engine.py --date 2026-03-30
    python projection_engine.py --date 2026-03-30 --out data/projections_csv/
    python projection_engine.py --backfill 2025-04-01:2025-04-30

Edge / Kelly columns are intentionally excluded for now (deferred until
a market-odds source is wired in). When that lands, run the rows from
this CSV through `src.edge.analyze_game` to populate edges without
changing this engine.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from src.data_ingestion.storage import DEFAULT_DB_PATH, SQLiteStore
from src.projection_engine import MODEL_VERSION, compute_game_projection
from src.projections import get_league_avg_rpg
from src.weather import fetch_game_weather

log = logging.getLogger("projection_engine")

DEFAULT_OUT_DIR = os.path.join("data", "projections_csv")

CSV_COLUMNS = [
    "date",
    "game_id",
    "venue_id",
    "park_runs_factor",
    "weather_factor",
    "weather_summary",
    "away_team_id", "away_team_name",
    "home_team_id", "home_team_name",
    "away_offensive_factor", "home_offensive_factor",
    "away_pitching_fip", "home_pitching_fip",
    "away_pitching_factor", "home_pitching_factor",
    "away_is_opener", "home_is_opener",
    "away_is_bullpen_heavy", "home_is_bullpen_heavy",
    "away_lambda", "home_lambda",
    "away_win_pct", "home_win_pct",
    "away_fair_decimal", "home_fair_decimal",
    "away_fair_line", "home_fair_line",
    "expected_total_nb",
    "projected_total",
    "ou_line",
    "totals_confidence",
    "predicted_score_away", "predicted_score_home", "score_probability_pct",
    # F5 (first 5 innings)
    "f5_away_lambda", "f5_home_lambda",
    "f5_away_pitching_factor", "f5_home_pitching_factor",
    "f5_away_win_pct", "f5_home_win_pct", "f5_tie_pct",
    "f5_predicted_score_away", "f5_predicted_score_home",
    "f5_score_probability_pct", "f5_total",
]


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


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _load_features_by_game(
    store: SQLiteStore, target_date: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Returns {game_id: {'home': {...}, 'away': {...}}} for the date."""
    rows = store.conn.execute(
        """
        SELECT * FROM team_features
        WHERE sport='mlb' AND game_date=?
        ORDER BY game_id, side
        """,
        (target_date,),
    ).fetchall()
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        grouped[r["game_id"]][r["side"]] = _row_to_dict(r)
    return grouped


def _summarize_weather(summary: dict[str, Any] | None, data: dict[str, Any] | None) -> str:
    """One-line description of the weather conditions for the CSV."""
    if not summary and not data:
        return ""
    bits = []
    if data:
        t = data.get("temperature_2m")
        w = data.get("wind_speed_10m")
        wd = data.get("wind_direction_10m")
        p = data.get("precipitation")
        if t is not None: bits.append(f"{t:.0f}F")
        if w is not None and wd is not None: bits.append(f"wind {w:.0f}mph @{wd:.0f}")
        if p is not None and p > 0: bits.append(f"precip {p:.1f}mm")
    if summary and summary.get("roof") in ("closed", "dome"):
        bits.append(summary["roof"])
    return ", ".join(bits)


def _lineup_state(home: dict, away: dict) -> str:
    h = (home.get("lineup_source") or "").startswith("official")
    a = (away.get("lineup_source") or "").startswith("official")
    if h and a:
        return "official_both"
    if h or a:
        return "official_partial"
    return "none"


def project_date(
    store: SQLiteStore,
    target_date: str,
    out_dir: str,
    league_avg_rpg: float,
    use_weather: bool = True,
    log_predictions: bool = True,
) -> tuple[int, str | None]:
    games = _load_features_by_game(store, target_date)
    if not games:
        log.warning("[%s] no team_features rows; skipping", target_date)
        return 0, None

    # Fetch weather once per venue (multiple games per day at the same park is rare for MLB
    # but the cache keeps us safe for doubleheaders and saves API calls).
    weather_cache: dict[Any, tuple[dict | None, float, dict]] = {}
    if use_weather:
        venue_ids = {sides[s].get("venue_id") for sides in games.values() for s in sides if sides[s]}
        for vid in venue_ids:
            if vid is None or vid in weather_cache:
                continue
            try:
                vid_int = int(vid)
            except (TypeError, ValueError):
                continue
            weather_cache[vid] = fetch_game_weather(vid_int)
            time.sleep(0.05)
        log.info("[%s] fetched weather for %d venues", target_date, sum(1 for v in weather_cache.values() if v[0]))

    rows_out: list[dict[str, Any]] = []
    skipped = 0
    for game_id, sides in games.items():
        home = sides.get("home")
        away = sides.get("away")
        if not home or not away:
            log.warning("[%s] game %s missing one side; skipping", target_date, game_id)
            skipped += 1
            continue
        venue_id = home.get("venue_id") or away.get("venue_id")
        wx_data, wx_factor, wx_summary = weather_cache.get(venue_id, (None, 1.0, {}))
        proj = compute_game_projection(
            home, away,
            league_avg_rpg=league_avg_rpg,
            weather_factor=wx_factor,
            weather_summary=wx_summary,
        )
        rows_out.append({
            "date": target_date,
            "game_id": proj["game_id"],
            "venue_id": proj.get("venue_id"),
            "park_runs_factor": proj.get("park_runs_factor"),
            "weather_factor": proj.get("weather_factor"),
            "weather_summary": _summarize_weather(wx_summary, wx_data),
            "away_team_id": proj.get("away_team_id"),
            "away_team_name": proj.get("away_team_name"),
            "home_team_id": proj.get("home_team_id"),
            "home_team_name": proj.get("home_team_name"),
            "away_offensive_factor": proj.get("away_offensive_factor"),
            "home_offensive_factor": proj.get("home_offensive_factor"),
            "away_pitching_fip": proj.get("away_pitching_fip"),
            "home_pitching_fip": proj.get("home_pitching_fip"),
            "away_pitching_factor": proj.get("away_pitching_factor"),
            "home_pitching_factor": proj.get("home_pitching_factor"),
            "away_is_opener": int(bool(proj.get("away_is_opener"))),
            "home_is_opener": int(bool(proj.get("home_is_opener"))),
            "away_is_bullpen_heavy": int(bool(proj.get("away_is_bullpen_heavy"))),
            "home_is_bullpen_heavy": int(bool(proj.get("home_is_bullpen_heavy"))),
            "away_lambda": proj.get("away_lambda"),
            "home_lambda": proj.get("home_lambda"),
            "away_win_pct": proj.get("away_win_pct"),
            "home_win_pct": proj.get("home_win_pct"),
            "away_fair_decimal": proj.get("away_fair_decimal"),
            "home_fair_decimal": proj.get("home_fair_decimal"),
            "away_fair_line": proj.get("away_fair_line"),
            "home_fair_line": proj.get("home_fair_line"),
            "expected_total_nb": proj.get("expected_total_nb"),
            "projected_total": proj.get("projected_total"),
            "ou_line": proj.get("ou_line"),
            "totals_confidence": proj.get("totals_confidence"),
            "predicted_score_away": proj.get("predicted_score_away"),
            "predicted_score_home": proj.get("predicted_score_home"),
            "score_probability_pct": proj.get("score_probability_pct"),
            "f5_away_lambda": proj.get("f5_away_lambda"),
            "f5_home_lambda": proj.get("f5_home_lambda"),
            "f5_away_pitching_factor": proj.get("f5_away_pitching_factor"),
            "f5_home_pitching_factor": proj.get("f5_home_pitching_factor"),
            "f5_away_win_pct": proj.get("f5_away_win_pct"),
            "f5_home_win_pct": proj.get("f5_home_win_pct"),
            "f5_tie_pct": proj.get("f5_tie_pct"),
            "f5_predicted_score_away": proj.get("f5_predicted_score_away"),
            "f5_predicted_score_home": proj.get("f5_predicted_score_home"),
            "f5_score_probability_pct": proj.get("f5_score_probability_pct"),
            "f5_total": proj.get("f5_total"),
        })

        # Bank this prediction for forward testing / track record.
        if log_predictions:
            store.log_prediction({
                "sport": "mlb",
                "game_id": proj["game_id"],
                "game_date": target_date,
                "model_version": MODEL_VERSION,
                "away_team": proj.get("away_team_name"),
                "home_team": proj.get("home_team_name"),
                "away_win_prob": (proj.get("away_win_pct") or 0) / 100.0,
                "home_win_prob": (proj.get("home_win_pct") or 0) / 100.0,
                "away_fair_line": proj.get("away_fair_line"),
                "home_fair_line": proj.get("home_fair_line"),
                "away_lambda": proj.get("away_lambda"),
                "home_lambda": proj.get("home_lambda"),
                "projected_total": proj.get("projected_total"),
                "ou_line": proj.get("ou_line"),
                "f5_away_win_prob": (proj.get("f5_away_win_pct") or 0) / 100.0,
                "f5_home_win_prob": (proj.get("f5_home_win_pct") or 0) / 100.0,
                "f5_total": proj.get("f5_total"),
                "lineup_state": _lineup_state(home, away),
                "weather_factor": proj.get("weather_factor"),
                "park_runs_factor": proj.get("park_runs_factor"),
            })
    if log_predictions:
        store.conn.commit()

    rows_out.sort(key=lambda r: str(r.get("game_id")))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"projections_{target_date}.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
    log.info(
        "[%s] wrote %d games (%d skipped) -> %s",
        target_date, len(rows_out), skipped, out_path,
    )

    for r in rows_out:
        log.info(
            "[%s] %s @ %s | %s/%s win%% | fair %+d/%+d | total %.1f (ou %.1f, %s)",
            target_date,
            r["away_team_name"], r["home_team_name"],
            r["away_win_pct"], r["home_win_pct"],
            r["away_fair_line"] or 0, r["home_fair_line"] or 0,
            r["projected_total"] or 0, r["ou_line"] or 0, r["totals_confidence"],
        )

    return len(rows_out), out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Core MLB projection engine")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="CSV output dir")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--no-weather", action="store_true",
        help="Skip the per-venue weather fetch (defaults to fetching live weather and "
             "applying it to lambdas + totals)",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Skip writing predictions to the prediction_log forward-test table",
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--date", help="Single date YYYY-MM-DD (default: today)")
    grp.add_argument("--backfill", help="Range START:END (YYYY-MM-DD:YYYY-MM-DD)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    store = SQLiteStore(args.db)
    league_avg_rpg = get_league_avg_rpg()
    log.info("league avg R/G = %.3f", league_avg_rpg)
    use_weather = not args.no_weather
    log_predictions = not args.no_log

    if args.backfill:
        start, end = _parse_range(args.backfill)
        total = 0
        for d in _daterange(start, end):
            n, _ = project_date(store, d.isoformat(), args.out_dir, league_avg_rpg,
                                use_weather=use_weather, log_predictions=log_predictions)
            total += n
        log.info("backfill complete: %d games projected", total)
        return 0

    target = args.date or date.today().isoformat()
    project_date(store, target, args.out_dir, league_avg_rpg,
                 use_weather=use_weather, log_predictions=log_predictions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
