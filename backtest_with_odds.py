#!/usr/bin/env python3
"""Backtest MLB model against actual closing moneyline odds.

Unlike the standard backtest (flat -110 assumption), this uses real
closing lines to compute true edge, ROI, and profitability.

Usage:
    # Backtest with cached odds data
    python backtest_with_odds.py --season 2025

    # Multi-season
    python backtest_with_odds.py --seasons 2023,2024,2025

    # Import odds from CSV and backtest in one step
    python backtest_with_odds.py --season 2025 --odds-csv odds_2025.csv

    # Use The Odds API
    python backtest_with_odds.py --season 2025 --odds-api --api-key YOUR_KEY

    # Show detailed bet log
    python backtest_with_odds.py --season 2025 --show-bets --min-edge 0.05

Data requirements:
    - Game results: data/results_{year}.json (from existing backtest infra)
    - Closing odds: data/odds_{year}.json (from fetch_odds.py or CSV import)
"""

import argparse
import json
import os
import sys

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
    run_edge_sweep_with_odds,
    simulate_flat_betting_with_odds,
    print_odds_backtest_report,
)
from src.odds import (
    load_cached_odds,
    import_odds_csv,
    save_odds_cache,
    merge_odds,
    match_odds_to_games,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def predict_historical_game(
    game: dict,
    pitcher_proj: dict,
    league_avg: float,
) -> dict | None:
    """Generate prediction for a historical game (same as backtest_2025.py)."""
    away_name = game.get("away_team_name", "")
    home_name = game.get("home_team_name", "")

    if not away_name or not home_name:
        return None

    away_proj = get_team_projected_strength(away_name)
    home_proj = get_team_projected_strength(home_name)

    away_off = away_proj["proj_off_factor"]
    home_off = home_proj["proj_off_factor"]
    away_def = away_proj["proj_def_factor"]
    home_def = home_proj["proj_def_factor"]

    away_pitcher = game.get("away_pitcher_name", "")
    home_pitcher = game.get("home_pitcher_name", "")

    away_sp = _sp_factor(away_pitcher, pitcher_proj, league_avg)
    home_sp = _sp_factor(home_pitcher, pitcher_proj, league_avg)

    away_bp = get_bullpen_summary(away_name).get("factor", 1.0)
    home_bp = get_bullpen_summary(home_name).get("factor", 1.0)
    away_sp_ip = get_starter_projected_length(away_pitcher, pitcher_proj)
    home_sp_ip = get_starter_projected_length(home_pitcher, pitcher_proj)

    away_game_pitch = compute_game_pitching_factor(away_sp, away_bp, away_sp_ip)
    home_game_pitch = compute_game_pitching_factor(home_sp, home_bp, home_sp_ip)

    park_factor = get_park_factor(game.get("venue_id"))

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

    pred = generate_prediction(away_lambda, home_lambda, away_name, home_name)

    away_score = game.get("away_score", 0)
    home_score = game.get("home_score", 0)

    if away_score == home_score:
        return None

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


def run_backtest_with_odds(
    season: int,
    odds_data: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run backtest for a season, returning (predictions_with_odds, raw_games).

    If odds_data is None, loads from cache.
    """
    results_path = os.path.join(DATA_DIR, f"results_{season}.json")
    if not os.path.exists(results_path):
        print(f"\n❌ Results file not found: {results_path}")
        return [], []

    with open(results_path) as f:
        games = json.load(f)
    print(f"\n  {season} season: {len(games)} games")

    # Load odds
    if odds_data is None:
        odds_data = load_cached_odds(season)

    if odds_data:
        games = match_odds_to_games(games, odds_data)
        with_odds = sum(1 for g in games if g.get("home_odds") is not None)
        print(f"  Odds matched: {with_odds}/{len(games)} ({with_odds/len(games)*100:.0f}%)")
    else:
        print(f"  ⚠ No odds data for {season}")

    # Load projections
    pitcher_proj = load_all_pitcher_projections()
    league_avg = get_league_avg_rpg()

    # Run predictions
    predictions = []
    for game in games:
        pred = predict_historical_game(game, pitcher_proj, league_avg)
        if pred is None:
            continue

        # Carry over odds data
        pred["home_odds"] = game.get("home_odds")
        pred["away_odds"] = game.get("away_odds")
        pred["odds_source"] = game.get("odds_source")
        pred["season"] = season
        predictions.append(pred)

    print(f"  Predictions: {len(predictions)}")
    with_odds = sum(1 for p in predictions if p.get("home_odds") is not None)
    print(f"  With odds: {with_odds}")

    return predictions, games


def show_bet_log(predictions: list[dict], min_edge: float = 0.03) -> None:
    """Show individual bets that met the edge threshold."""
    from src.backtest import simulate_betting_with_odds

    result = simulate_betting_with_odds(predictions, min_edge=min_edge)
    bets = result.get("bets", [])

    if not bets:
        print(f"\n  No bets at ≥{min_edge*100:.0f}% edge threshold")
        return

    print(f"\n  BET LOG (≥{min_edge*100:.0f}% edge, {len(bets)} bets)")
    print(f"  {'─' * 90}")
    print(f"  {'Date':<12} {'Side':<6} {'Odds':>6} {'Edge':>6} {'Bet $':>8} {'Result':>7} {'P/L':>9} {'Bank':>9}")
    print(f"  {'─' * 90}")

    for b in bets:
        result_str = "WIN" if b["won"] else "LOSS"
        marker = "✓" if b["won"] else "✗"
        print(f"  {'':<12} {b['bet_side']:<6} {b['bet_odds']:>+6d} "
              f"{b['edge']*100:>+5.1f}% {b['bet_amount']:>7.2f} "
              f"{result_str:>6} {marker} {b['profit']:>+8.2f} {b['bankroll']:>8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Backtest MLB model vs closing odds")
    parser.add_argument("--season", type=int, default=None, help="Single season")
    parser.add_argument("--seasons", type=str, default=None,
                        help="Comma-separated seasons (e.g., 2023,2024,2025)")
    parser.add_argument("--odds-csv", type=str, help="Import odds from CSV before backtesting")
    parser.add_argument("--show-bets", action="store_true", help="Show individual bet log")
    parser.add_argument("--min-edge", type=float, default=0.03,
                        help="Min edge for bet log display (default: 3%%)")

    args = parser.parse_args()

    # Determine seasons
    if args.seasons:
        seasons = [int(s) for s in args.seasons.split(",")]
    elif args.season:
        seasons = [args.season]
    else:
        seasons = [2023, 2024, 2025]

    print(f"\n{'=' * 80}")
    print(f"  ⚾ BACKTEST vs CLOSING ODDS: {', '.join(str(s) for s in seasons)}")
    print(f"{'=' * 80}")

    all_predictions = []

    for season in seasons:
        # Import CSV if provided
        odds_data = None
        if args.odds_csv:
            print(f"\n  Importing odds from {args.odds_csv}...")
            odds_data = import_odds_csv(args.odds_csv)
            existing = load_cached_odds(season)
            merged = merge_odds(existing, odds_data)
            save_odds_cache(season, merged)
            odds_data = merged
            print(f"  {len(odds_data)} odds records loaded")

        preds, _ = run_backtest_with_odds(season, odds_data)
        all_predictions.extend(preds)

    if not all_predictions:
        print("\n❌ No predictions generated. Check data files.")
        sys.exit(1)

    # Per-season summary
    if len(seasons) > 1:
        print(f"\n{'=' * 80}")
        print(f"  PER-SEASON RESULTS")
        print(f"{'=' * 80}")
        print(f"\n  {'Season':>7} {'Games':>7} {'Acc%':>7} {'w/Odds':>8} {'Coverage':>10}")
        print(f"  {'─' * 45}")

        for season in seasons:
            sp = [p for p in all_predictions if p.get("season") == season]
            if not sp:
                continue
            metrics = compute_accuracy(sp)
            with_odds = sum(1 for p in sp if p.get("home_odds") is not None)
            coverage = with_odds / len(sp) * 100 if sp else 0
            print(f"  {season:>7} {metrics['n_games']:>7} {metrics['accuracy']:>6.1f}% "
                  f"{with_odds:>8} {coverage:>9.0f}%")

    # Combined analysis
    metrics = compute_accuracy(all_predictions)

    # Odds coverage
    with_odds = sum(1 for p in all_predictions if p.get("home_odds") is not None)
    odds_coverage = with_odds / len(all_predictions) * 100 if all_predictions else 0

    # Edge sweeps with actual odds
    edge_sweep = run_edge_sweep_with_odds(all_predictions)

    # Flat bet sweep
    flat_thresholds = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    flat_sweep = []
    for t in flat_thresholds:
        r = simulate_flat_betting_with_odds(all_predictions, min_edge=t)
        flat_sweep.append(r)

    # Print report
    label = f"{seasons[0]}" if len(seasons) == 1 else f"{seasons[0]}-{seasons[-1]}"
    print_odds_backtest_report(metrics, edge_sweep, flat_sweep,
                               season=label, odds_coverage=odds_coverage)

    # Bet log
    if args.show_bets:
        show_bet_log(all_predictions, min_edge=args.min_edge)

    # Save
    os.makedirs("output", exist_ok=True)
    out_path = f"output/backtest_odds_{label}.json"
    with open(out_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"  Saved predictions to {out_path}\n")


if __name__ == "__main__":
    main()
