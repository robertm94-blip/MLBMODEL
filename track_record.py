#!/usr/bin/env python3
"""Out-of-sample track record from the live prediction log.

This is the leak-free counterpart to backtest_winprob.py / backtest_totals.py:
instead of reconstructing history with a proxy, it reports how the *actual live
model* has performed on predictions it banked before games were played.

Reports (over graded prediction_log rows):
  - Win prob: Brier, log loss, accuracy, ECE, reliability curve
  - Totals: signed bias, MAE
  - Sample size + date span + breakdown by lineup_state

Until enough games accumulate this will be a small sample - that's expected.
The point is that every daily run grows a verifiable record.

Examples:
    python track_record.py
    python track_record.py --start 2026-05-01 --end 2026-09-30 --buckets 10
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from typing import Any

from src.data_ingestion.storage import DEFAULT_DB_PATH, SQLiteStore


def _winprob_metrics(rows: list[dict[str, Any]], n_buckets: int) -> dict[str, Any]:
    preds = [(r["home_win_prob"], 1 if r["actual_winner"] == "home" else 0)
             for r in rows if r["home_win_prob"] is not None and r["actual_winner"]]
    if not preds:
        return {"n": 0}
    n = len(preds)
    brier = sum((p - y) ** 2 for p, y in preds) / n
    eps = 1e-12
    logloss = -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                   for p, y in preds) / n
    acc = sum(1 for p, y in preds if (p >= 0.5) == (y == 1)) / n
    ece = 0.0
    buckets = []
    for b in range(n_buckets):
        lo, hi = b / n_buckets, (b + 1) / n_buckets
        mem = [(p, y) for p, y in preds
               if (lo <= p < hi) or (b == n_buckets - 1 and p == 1.0)]
        if not mem:
            continue
        bn = len(mem)
        ap = sum(p for p, _ in mem) / bn
        aa = sum(y for _, y in mem) / bn
        ece += (bn / n) * abs(ap - aa)
        buckets.append((lo, hi, bn, ap, aa))
    return {"n": n, "brier": brier, "logloss": logloss, "accuracy": acc,
            "ece": ece, "buckets": buckets}


def _totals_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errs = [r["total_error"] for r in rows if r["total_error"] is not None]
    if not errs:
        return {"n": 0}
    n = len(errs)
    return {"n": n, "bias": sum(errs) / n, "mae": sum(abs(e) for e in errs) / n}


def report(store: SQLiteStore, start: str | None, end: str | None, n_buckets: int) -> None:
    rows = store.query_prediction_log("mlb", graded_only=True, start_date=start, end_date=end)
    print(f"\n=== Live model track record ({len(rows)} graded predictions) ===")
    if not rows:
        print("  No graded predictions yet. Run projection_engine.py daily, then")
        print("  grade_predictions.py after games finish to start building the record.")
        return
    dates = sorted({r["game_date"] for r in rows})
    print(f"  span: {dates[0]} -> {dates[-1]}   ({len(dates)} dates)")
    versions = sorted({r["model_version"] for r in rows})
    print(f"  model versions: {', '.join(versions)}")

    wp = _winprob_metrics(rows, n_buckets)
    if wp["n"]:
        print(f"\n  -- Win probability ({wp['n']} games) --")
        print(f"    accuracy:  {wp['accuracy']*100:.1f}%")
        print(f"    Brier:     {wp['brier']:.4f}   (0.25 = coin flip)")
        print(f"    log loss:  {wp['logloss']:.4f}")
        print(f"    ECE:       {wp['ece']*100:.2f}%   (<2% well-calibrated)")
        if wp["buckets"]:
            print(f"    reliability:")
            print(f"      {'bucket':>10}  {'n':>4}  {'pred':>6}  {'actual':>7}")
            for lo, hi, bn, ap, aa in wp["buckets"]:
                print(f"      {lo*100:>3.0f}-{hi*100:<3.0f}%  {bn:>4}  {ap*100:>5.1f}% {aa*100:>6.1f}%")

    tot = _totals_metrics(rows)
    if tot["n"]:
        print(f"\n  -- Totals ({tot['n']} games) --")
        print(f"    signed bias: {tot['bias']:+.3f} runs  (>0 = over-projecting)")
        print(f"    MAE:         {tot['mae']:.3f} runs")

    by_state: dict[str, list] = defaultdict(list)
    for r in rows:
        by_state[r["lineup_state"] or "unknown"].append(r)
    print(f"\n  -- By lineup state at prediction time --")
    for state, grp in sorted(by_state.items()):
        sc = [r for r in grp if r["side_correct"] is not None]
        acc = sum(r["side_correct"] for r in sc) / len(sc) * 100 if sc else 0
        print(f"    {state:<18} {len(grp):>4} games   side acc {acc:.1f}%")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live model track record from prediction_log")
    p.add_argument("--db", default=DEFAULT_DB_PATH)
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD")
    p.add_argument("--buckets", type=int, default=10)
    args = p.parse_args(argv)
    store = SQLiteStore(args.db)
    report(store, args.start, args.end, args.buckets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
