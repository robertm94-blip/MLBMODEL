#!/usr/bin/env python3
"""Win-probability calibration backtest.

Answers the question: "when the engine says the home team has a P% chance to
win, does the home team actually win ~P% of the time?" That is the measure of
whether our probability estimates are *true*, independent of pick accuracy.

Method (mirrors backtest_totals.py):
  - For each completed game in data/results_{year}.json, derive team-level
    offensive / pitching factors from that season's RS/RA (per-season, valid;
    not the live lineup model, which can't be reconstructed historically because
    the projection files are point-in-time-2026 — see CLAUDE.md). This validates
    the NB win-probability *machinery* on real outcomes.
  - Build lambdas via the same compute_expected_runs the live engine uses,
    run the Negative Binomial, take the home win probability.
  - Compare to the actual result.

Reports the metrics that quantify "confidence in truth":
  - Brier score (lower = better; 0.25 = coin flip, ~0.23 = good MLB model)
  - Log loss
  - Reliability curve (predicted bucket vs realized win rate) + ECE
  - Accuracy + favorite-wins sanity check (~58% leaguewide)

Examples:
    python backtest_winprob.py --years 2024,2025
    python backtest_winprob.py --years 2023,2024,2025 --buckets 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

from src import totals_calibration as cal
from src.features import compute_expected_runs
from src.model import predict_score_distribution, win_probability
from src.park_factors import get_park_factor

LEAGUE_AVG_FIP = 4.05  # unused here; team-RA proxy already normalized to league RPG


def _home_win_prob(away_off, home_off, away_pitch, home_pitch, venue_id, league_rpg) -> float:
    """Same lambda construction as src/projection_engine.compute_game_projection,
    using team-level factors as the historical proxy."""
    park = get_park_factor(venue_id)
    away_lambda = compute_expected_runs(
        batting_team_off_factor=away_off,
        pitching_team_def_factor=home_pitch,
        starter_factor=home_pitch,
        park_factor=park,
        league_avg_rpg=league_rpg,
        is_home=False,
    )
    home_lambda = compute_expected_runs(
        batting_team_off_factor=home_off,
        pitching_team_def_factor=away_pitch,
        starter_factor=away_pitch,
        park_factor=park,
        league_avg_rpg=league_rpg,
        is_home=True,
    )
    matrix = predict_score_distribution(away_lambda, home_lambda)
    wp = win_probability(matrix)
    return wp["home_win"]


def run(years: list[int], n_buckets: int) -> dict[str, Any]:
    preds: list[tuple[float, int]] = []  # (home_win_prob, home_won)

    for year in years:
        with open(os.path.join("data", f"results_{year}.json")) as fh:
            results = json.load(fh)
        with open(os.path.join("data", f"team_stats_{year}.json")) as fh:
            offense, pitching, league_rpg = cal.parse_team_stats(json.load(fh))

        for g in results:
            if g.get("home_score") is None or g.get("away_score") is None:
                continue
            if g["home_score"] == g["away_score"]:
                continue  # ties (suspended games) — drop
            factors = cal.factors_for_game(g, offense, pitching, league_rpg)
            if factors is None:
                continue
            a_off, h_off, a_pitch, h_pitch = factors
            p = _home_win_prob(a_off, h_off, a_pitch, h_pitch, g.get("venue_id"), league_rpg)
            home_won = 1 if g["home_score"] > g["away_score"] else 0
            preds.append((p, home_won))

    if not preds:
        return {"n": 0}

    n = len(preds)
    brier = sum((p - y) ** 2 for p, y in preds) / n
    eps = 1e-12
    logloss = -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in preds
    ) / n

    # Accuracy: predict home win when p >= 0.5
    correct = sum(1 for p, y in preds if (p >= 0.5) == (y == 1))
    accuracy = correct / n

    # Base rates
    home_win_rate = sum(y for _, y in preds) / n
    mean_pred = sum(p for p, _ in preds) / n

    # Favorite-wins sanity: how often does the side we make the favorite win?
    fav_correct = sum(1 for p, y in preds if (y == 1) == (p >= 0.5))
    fav_rate = fav_correct / n  # same as accuracy here, kept for clarity

    # Reliability curve + ECE
    buckets: list[dict[str, Any]] = []
    ece = 0.0
    for b in range(n_buckets):
        lo = b / n_buckets
        hi = (b + 1) / n_buckets
        members = [(p, y) for p, y in preds if (lo <= p < hi) or (b == n_buckets - 1 and p == 1.0)]
        if not members:
            buckets.append({"lo": lo, "hi": hi, "n": 0, "pred": None, "actual": None})
            continue
        bn = len(members)
        avg_pred = sum(p for p, _ in members) / bn
        avg_act = sum(y for _, y in members) / bn
        ece += (bn / n) * abs(avg_pred - avg_act)
        buckets.append({"lo": lo, "hi": hi, "n": bn, "pred": avg_pred, "actual": avg_act})

    return {
        "n": n, "brier": brier, "logloss": logloss, "accuracy": accuracy,
        "home_win_rate": home_win_rate, "mean_pred": mean_pred,
        "fav_rate": fav_rate, "ece": ece, "buckets": buckets,
    }


def print_report(m: dict[str, Any], years: list[int]) -> None:
    if m.get("n", 0) == 0:
        print("no eligible games")
        return
    print(f"\n=== Win-probability calibration — seasons {years} ===\n")
    print(f"  games:                {m['n']:,}")
    print(f"  actual home win rate: {m['home_win_rate']*100:.1f}%   (model mean pred: {m['mean_pred']*100:.1f}%)")
    print(f"  accuracy (p>=.5):     {m['accuracy']*100:.1f}%   (favorite-wins baseline ~58%)")
    print(f"  Brier score:          {m['brier']:.4f}   (0.25 = coin flip; ~0.225 = sharp)")
    print(f"  Log loss:             {m['logloss']:.4f}   (0.693 = coin flip)")
    print(f"  ECE (calibration err):{m['ece']*100:.2f}%   (lower = better; <2% is well-calibrated)")
    print(f"\n  Reliability curve (predicted home-win% vs actual):")
    print(f"    {'bucket':>12}  {'n':>6}  {'pred':>7}  {'actual':>7}  {'gap':>7}")
    print(f"    {'-'*12}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")
    for b in m["buckets"]:
        if b["n"] == 0:
            continue
        gap = b["actual"] - b["pred"]
        flag = "  <--" if abs(gap) > 0.05 else ""
        print(f"    {b['lo']*100:>4.0f}-{b['hi']*100:<3.0f}%   {b['n']:>6}  "
              f"{b['pred']*100:>6.1f}% {b['actual']*100:>6.1f}% {gap*100:>+6.1f}%{flag}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Win-probability calibration backtest")
    p.add_argument("--years", default="2024,2025")
    p.add_argument("--buckets", type=int, default=10, help="Reliability curve buckets (default 10)")
    args = p.parse_args(argv)
    years = [int(y) for y in args.years.split(",") if y.strip()]
    m = run(years, args.buckets)
    print_report(m, years)
    return 0


if __name__ == "__main__":
    sys.exit(main())
