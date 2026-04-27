#!/usr/bin/env python3
"""Totals model backtest + calibration.

Reads:
  data/results_{year}.json     - per-game finals for a season
  data/team_stats_{year}.json  - per-team season RS/RA + games

For every completed game:
  - team_offense / team_pitching factors derived from season RS/RA per game,
    normalized to that season's league RPG. Team-season RA is a coarse proxy
    for the per-game pitching factor (we don't have starter-level FIP in the
    historical files); good enough to characterize systemic bias.
  - feeds into `compute_totals_projection` and compares projected_total vs
    (away_score + home_score).

Reports signed bias, MAE, RMSE, over/under split, distribution buckets, and
the worst per-park biases. With --calibrate, sweeps a small grid over
LEAGUE_AVG_TOTAL / REGRESSION_TO_MEAN / FACTOR_SCALE_DAMPENER and prints
the top configs by |signed_bias| subject to MAE not regressing.

Examples:
    python backtest_totals.py --years 2024,2025
    python backtest_totals.py --years 2024,2025 --calibrate
    python backtest_totals.py --years 2025 --park-breakdown
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any

from src import totals_calibration as cal


def _load_year(year: int) -> tuple[list[dict[str, Any]], dict, dict, float]:
    res_path = os.path.join("data", f"results_{year}.json")
    stats_path = os.path.join("data", f"team_stats_{year}.json")
    with open(res_path) as fh:
        results = json.load(fh)
    with open(stats_path) as fh:
        stats_payload = json.load(fh)
    offense, pitching, league_rpg = cal.parse_team_stats(stats_payload)
    return results, offense, pitching, league_rpg


def _eligible_games(results, offense, pitching, league_rpg):
    """Yield (game, factor_tuple) for completed games with both teams in stats."""
    for game in results:
        if game.get("home_score") is None or game.get("away_score") is None:
            continue
        factors = cal.factors_for_game(game, offense, pitching, league_rpg)
        if factors is None:
            continue
        yield game, factors


def run_backtest(
    years: list[int],
    *,
    league_avg_total: float | None = None,
    regression: float | None = None,
    factor_scale: float | None = None,
) -> dict[str, Any]:
    """Iterate every game and aggregate metrics. Optional constant overrides
    are applied via cal.project_with_constants (which mutates totals_model
    in-process; restore_defaults afterwards)."""
    cal.snapshot_defaults()
    try:
        # Use defaults when caller didn't override.
        from src import totals_model
        lat = league_avg_total if league_avg_total is not None else totals_model.LEAGUE_AVG_TOTAL
        reg = regression if regression is not None else totals_model.REGRESSION_TO_MEAN
        scl = factor_scale if factor_scale is not None else getattr(totals_model, "FACTOR_SCALE_DAMPENER", 1.0)

        all_errors: list[float] = []
        per_park: dict[int, list[float]] = defaultdict(list)
        proj_buckets = defaultdict(int)
        actual_buckets = defaultdict(int)
        n_eligible = 0
        n_skipped = 0
        n_total = 0

        for year in years:
            results, offense, pitching, league_rpg = _load_year(year)
            n_total += len(results)
            for game, (a_off, h_off, a_pitch, h_pitch) in _eligible_games(
                results, offense, pitching, league_rpg
            ):
                projected = cal.project_with_constants(
                    a_off, h_off, a_pitch, h_pitch,
                    venue_id=game.get("venue_id"),
                    league_avg_total=lat,
                    regression=reg,
                    factor_scale=scl,
                )
                actual = float(game["home_score"] + game["away_score"])
                err = projected - actual
                all_errors.append(err)
                if game.get("venue_id") is not None:
                    per_park[game["venue_id"]].append(err)
                proj_buckets[_bucket(projected)] += 1
                actual_buckets[_bucket(actual)] += 1
                n_eligible += 1
            n_skipped += len(results) - sum(
                1 for g in results
                if g.get("home_score") is not None
                and cal.factors_for_game(g, offense, pitching, league_rpg) is not None
            )

        if not all_errors:
            return {"n": 0}

        mean_bias = statistics.fmean(all_errors)
        mae = statistics.fmean(abs(e) for e in all_errors)
        rmse = (statistics.fmean(e * e for e in all_errors)) ** 0.5
        over = sum(1 for e in all_errors if e > 0) / len(all_errors)

        worst_parks = sorted(
            (
                (vid, statistics.fmean(errs), len(errs))
                for vid, errs in per_park.items() if len(errs) >= 30
            ),
            key=lambda x: -abs(x[1]),
        )[:5]

        return {
            "n": n_eligible,
            "skipped": n_skipped,
            "total_in_files": n_total,
            "mean_bias": mean_bias,
            "mae": mae,
            "rmse": rmse,
            "over_pct": over * 100,
            "proj_buckets": dict(proj_buckets),
            "actual_buckets": dict(actual_buckets),
            "worst_parks": worst_parks,
            "constants": {
                "LEAGUE_AVG_TOTAL": lat,
                "REGRESSION_TO_MEAN": reg,
                "FACTOR_SCALE_DAMPENER": scl,
            },
        }
    finally:
        cal.restore_defaults()


def _bucket(total: float) -> str:
    if total < 8.0:
        return "<8.0"
    if total < 9.0:
        return "8.0-8.9"
    if total < 10.0:
        return "9.0-9.9"
    return ">=10.0"


def print_metrics(metrics: dict[str, Any], header: str) -> None:
    print(f"\n=== {header} ===")
    if metrics.get("n", 0) == 0:
        print("  no eligible games")
        return
    c = metrics["constants"]
    print(f"  constants: LEAGUE_AVG_TOTAL={c['LEAGUE_AVG_TOTAL']:.2f}  "
          f"REGRESSION_TO_MEAN={c['REGRESSION_TO_MEAN']:.2f}  "
          f"FACTOR_SCALE_DAMPENER={c['FACTOR_SCALE_DAMPENER']:.2f}")
    print(f"  games: {metrics['n']:,} eligible   "
          f"({metrics['total_in_files']:,} in files)")
    print(f"  signed bias: {metrics['mean_bias']:+.3f} runs   (>0 = over-projecting)")
    print(f"  MAE:         {metrics['mae']:.3f}")
    print(f"  RMSE:        {metrics['rmse']:.3f}")
    print(f"  projected over actual: {metrics['over_pct']:.1f}%")
    print(f"  distribution (projected | actual):")
    keys = ["<8.0", "8.0-8.9", "9.0-9.9", ">=10.0"]
    for k in keys:
        p = metrics["proj_buckets"].get(k, 0)
        a = metrics["actual_buckets"].get(k, 0)
        print(f"    {k:>8}  proj {p:>5}   actual {a:>5}")
    if metrics["worst_parks"]:
        print(f"  worst per-park bias (signed; >=30 games):")
        for vid, b, n in metrics["worst_parks"]:
            print(f"    venue {vid:>5}  bias {b:+.2f}  n={n}")


def calibration_grid(years: list[int]) -> None:
    print("\n=== calibration grid (lower |signed_bias| while not blowing up MAE) ===")
    baseline = run_backtest(years)
    print_metrics(baseline, "baseline (current constants)")
    base_mae = baseline["mae"]
    base_bias = baseline["mean_bias"]

    candidates = []
    for lat in [8.30, 8.40, 8.50, 8.60, 8.70, 8.80, 8.90]:
        for reg in [0.55, 0.65, 0.75, 0.85]:
            for scl in [1.00, 0.85, 0.70, 0.55]:
                m = run_backtest(years, league_avg_total=lat, regression=reg, factor_scale=scl)
                if m["n"] == 0:
                    continue
                if m["mae"] > base_mae + 0.10:
                    continue  # don't accept worse precision
                candidates.append((abs(m["mean_bias"]), m))
    candidates.sort(key=lambda x: x[0])

    print(f"\n  baseline bias = {base_bias:+.3f}, MAE = {base_mae:.3f}")
    print(f"  showing top 8 candidates (by |bias|, MAE within +0.10 of baseline):\n")
    print(f"  {'LAT':>6} {'REG':>5} {'SCALE':>6}  {'bias':>9} {'MAE':>6} {'RMSE':>6} {'over%':>6}")
    print("  " + "-" * 50)
    for absb, m in candidates[:8]:
        c = m["constants"]
        print(f"  {c['LEAGUE_AVG_TOTAL']:>6.2f} {c['REGRESSION_TO_MEAN']:>5.2f} "
              f"{c['FACTOR_SCALE_DAMPENER']:>6.2f}  {m['mean_bias']:>+9.3f} "
              f"{m['mae']:>6.3f} {m['rmse']:>6.3f} {m['over_pct']:>5.1f}%")
    if candidates:
        best = candidates[0][1]["constants"]
        print(f"\n  recommended: LEAGUE_AVG_TOTAL={best['LEAGUE_AVG_TOTAL']}, "
              f"REGRESSION_TO_MEAN={best['REGRESSION_TO_MEAN']}, "
              f"FACTOR_SCALE_DAMPENER={best['FACTOR_SCALE_DAMPENER']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Totals model backtest + calibration")
    p.add_argument("--years", default="2024,2025",
                   help="Comma-separated season years to evaluate (default 2024,2025)")
    p.add_argument("--calibrate", action="store_true",
                   help="Run a constants grid search after baseline")
    p.add_argument("--park-breakdown", action="store_true",
                   help="Show top per-park biases")
    args = p.parse_args(argv)

    years = [int(y) for y in args.years.split(",") if y.strip()]
    if args.calibrate:
        calibration_grid(years)
    else:
        metrics = run_backtest(years)
        print_metrics(metrics, f"baseline backtest on {years}")
        if not args.park_breakdown:
            metrics["worst_parks"] = []
            print_metrics(metrics, "summary (no park breakdown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
