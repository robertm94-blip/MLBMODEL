#!/usr/bin/env python3
"""Backtest the MLB model against the 2025 season.

Uses 2024 full-season projections (or 2025 preseason projections)
to predict 2025 game outcomes, then measures accuracy, calibration,
and simulated profitability.

For this backtest we use:
- Team strength: 2025 projected standings (Depth Charts)
- Pitchers: Blended Steamer/ZiPS/THE BAT projections
- Bullpen: Aggregated reliever projections
- Park factors: Same as production model
- Home advantage: +0.25 runs
- Weather/umpire/rest: Omitted (not retroactively available)
  This gives us a conservative baseline — real performance would be
  slightly better with these factors.
"""

import json
import sys
import os
import argparse

from src.projections import (
    get_team_projected_strength,
    load_all_pitcher_projections,
    get_league_avg_rpg,
)
from src.features import compute_expected_runs, get_park_factor
from src.bullpen import (
    get_bullpen_summary,
    get_starter_projected_length,
    compute_game_pitching_factor,
)
from src.model import generate_prediction
from src.backtest import (
    compute_accuracy,
    run_edge_sweep,
    print_backtest_report,
)


def predict_historical_game(
    game: dict,
    pitcher_proj: dict,
    league_avg: float,
) -> dict | None:
    """Generate prediction for a historical game."""
    away_name = game.get("away_team_name", "")
    home_name = game.get("home_team_name", "")

    if not away_name or not home_name:
        return None

    # Team strength
    away_proj = get_team_projected_strength(away_name)
    home_proj = get_team_projected_strength(home_name)

    away_off = away_proj["proj_off_factor"]
    home_off = home_proj["proj_off_factor"]
    away_def = away_proj["proj_def_factor"]
    home_def = home_proj["proj_def_factor"]

    # Starter quality
    away_pitcher = game.get("away_pitcher_name", "")
    home_pitcher = game.get("home_pitcher_name", "")

    away_sp = _sp_factor(away_pitcher, pitcher_proj, league_avg)
    home_sp = _sp_factor(home_pitcher, pitcher_proj, league_avg)

    # Bullpen
    away_bp = get_bullpen_summary(away_name).get("factor", 1.0)
    home_bp = get_bullpen_summary(home_name).get("factor", 1.0)
    away_sp_ip = get_starter_projected_length(away_pitcher, pitcher_proj)
    home_sp_ip = get_starter_projected_length(home_pitcher, pitcher_proj)

    away_game_pitch = compute_game_pitching_factor(away_sp, away_bp, away_sp_ip)
    home_game_pitch = compute_game_pitching_factor(home_sp, home_bp, home_sp_ip)

    # Park factor
    park_factor = get_park_factor(game.get("venue_id"))

    # Expected runs
    away_lambda = compute_expected_runs(
        batting_team_off_factor=away_off,
        pitching_team_def_factor=home_def,
        starter_factor=home_game_pitch,
        park_factor=park_factor,
        league_avg_rpg=league_avg,
        is_home=False,
    )

    home_lambda = compute_expected_runs(
        batting_team_off_factor=home_off,
        pitching_team_def_factor=away_def,
        starter_factor=away_game_pitch,
        park_factor=park_factor,
        league_avg_rpg=league_avg,
        is_home=True,
    )

    pred = generate_prediction(
        away_lambda, home_lambda, away_name, home_name,
    )

    # Determine actual outcome
    away_score = game.get("away_score", 0)
    home_score = game.get("home_score", 0)

    if away_score == home_score:
        return None  # Tie/suspended — skip

    home_won = 1 if home_score > away_score else 0
    home_win_prob = pred["win_probability"]["home"] / 100

    return {
        "game_id": game.get("game_id"),
        "date": game.get("date", ""),
        "away_team": away_name,
        "home_team": home_name,
        "away_pitcher": away_pitcher,
        "home_pitcher": home_pitcher,
        "home_win_prob": home_win_prob,
        "away_win_prob": 1 - home_win_prob,
        "home_won": home_won,
        "away_score": away_score,
        "home_score": home_score,
        "pred_away_runs": pred["away_expected_runs"],
        "pred_home_runs": pred["home_expected_runs"],
        "pred_total": pred["expected_total"],
        "actual_total": away_score + home_score,
    }


def _sp_factor(name: str, pitcher_proj: dict, league_avg: float) -> float:
    """Compute starter factor from projections."""
    p = pitcher_proj.get(name)
    if not p or p.get("era", 0) == 0:
        return 1.0

    era = p["era"]
    fip = p.get("fip", era)
    whip = p.get("whip", 1.30)
    ip = p.get("ip", 0)
    k9 = p.get("k9", 0)
    bb9 = p.get("bb9", 0)

    blended = era * 0.40 + fip * 0.60
    era_factor = blended / league_avg if league_avg > 0 else 1.0
    whip_factor = whip / 1.28 if whip > 0 else 1.0
    k_bb_factor = 1.0
    if k9 > 0 and bb9 > 0:
        k_bb_factor = 1.0 - ((k9 - bb9) - 5.0) * 0.015

    raw = 0.70 * era_factor + 0.20 * whip_factor + 0.10 * k_bb_factor
    reliability = min(ip / 150.0, 1.0)
    return raw * reliability + 1.0 * (1 - reliability)


def main():
    parser = argparse.ArgumentParser(description="Backtest MLB model against 2025 season")
    parser.add_argument("--results", type=str, default="data/results_2025.json",
                        help="Path to historical results JSON")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Limit number of games to process")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"\n❌ Results file not found: {args.results}")
        print(f"   Run the data fetcher first to download 2025 game results.")
        sys.exit(1)

    # Load results
    print(f"\n📊 Loading 2025 season results...")
    with open(args.results) as f:
        games = json.load(f)
    print(f"   {len(games)} games loaded")

    if args.max_games:
        games = games[:args.max_games]
        print(f"   Limited to {len(games)} games")

    # Load projections
    print(f"\n📊 Loading projections...")
    pitcher_proj = load_all_pitcher_projections()
    league_avg = get_league_avg_rpg()
    print(f"   {len(pitcher_proj)} pitcher projections")
    print(f"   League avg: {league_avg:.3f} R/G")

    # Run predictions
    print(f"\n🔮 Running predictions on {len(games)} games...")
    predictions = []
    skipped = 0
    no_pitcher = 0

    for i, game in enumerate(games):
        if (i + 1) % 500 == 0:
            print(f"   {i + 1}/{len(games)} processed...")

        pred = predict_historical_game(game, pitcher_proj, league_avg)
        if pred is None:
            skipped += 1
            continue
        predictions.append(pred)

    print(f"   ✓ {len(predictions)} predictions generated")
    print(f"   ✗ {skipped} games skipped (ties/suspended/missing data)")

    # Compute metrics
    print(f"\n📈 Computing metrics...")
    metrics = compute_accuracy(predictions)

    # Run edge sweep
    edge_sweep = run_edge_sweep(predictions)

    # Print report
    print_backtest_report(metrics, edge_sweep, season="2025")

    # Also compute total accuracy
    print(f"\n  TOTAL PREDICTION ANALYSIS:")
    print(f"  {'─' * 50}")

    # How well do we predict totals?
    total_diffs = [abs(p["pred_total"] - p["actual_total"]) for p in predictions]
    avg_total_error = sum(total_diffs) / len(total_diffs)
    within_1 = sum(1 for d in total_diffs if d <= 1.5) / len(total_diffs) * 100
    within_2 = sum(1 for d in total_diffs if d <= 2.5) / len(total_diffs) * 100

    over_count = sum(1 for p in predictions if p["actual_total"] > p["pred_total"])
    under_count = sum(1 for p in predictions if p["actual_total"] < p["pred_total"])

    print(f"  Avg total error: {avg_total_error:.2f} runs")
    print(f"  Within 1.5 runs: {within_1:.1f}%")
    print(f"  Within 2.5 runs: {within_2:.1f}%")
    print(f"  Over tendency: {over_count} ({over_count/len(predictions)*100:.1f}%)")
    print(f"  Under tendency: {under_count} ({under_count/len(predictions)*100:.1f}%)")

    # Save predictions for further analysis
    output_path = "output/backtest_predictions_2025.json"
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\n  Predictions saved to {output_path}")

    return metrics, edge_sweep


if __name__ == "__main__":
    main()
