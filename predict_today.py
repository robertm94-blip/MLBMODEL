#!/usr/bin/env python3
"""MLB Score Prediction Model — Predict final scores for today's slate.

Uses a Poisson-based model powered by:
- Team offensive/defensive strength factors (normalized to league average)
- Starting pitcher quality adjustments (ERA, WHIP, IP-weighted)
- Park factors (multi-year historical data)
- Home field advantage

Data sourced live from the MLB Stats API.
"""

import sys
import time
import argparse
from datetime import date

from src.mlb_api import get_schedule, get_team_stats, get_pitcher_stats, get_league_averages
from src.features import (
    compute_team_offensive_factor,
    compute_team_defensive_factor,
    compute_pitcher_factor,
    compute_expected_runs,
    get_park_factor,
)
from src.model import generate_prediction


def fetch_all_data(games: list[dict], season: int) -> dict:
    """Fetch all required stats for the day's games."""
    print(f"\n📊 Fetching {season} season stats from MLB Stats API...\n")

    # Get league averages
    print("  → League averages...", end=" ", flush=True)
    league_avg = get_league_averages(season)
    print(f"Done (avg {league_avg['avg_runs_per_game']:.2f} R/G)")

    # Collect unique team and pitcher IDs
    team_ids = set()
    pitcher_ids = set()
    for g in games:
        team_ids.add(g["away_team_id"])
        team_ids.add(g["home_team_id"])
        if g["away_pitcher_id"]:
            pitcher_ids.add(g["away_pitcher_id"])
        if g["home_pitcher_id"]:
            pitcher_ids.add(g["home_pitcher_id"])

    # Fetch team stats
    team_stats = {}
    print(f"  → Team stats ({len(team_ids)} teams)...", end=" ", flush=True)
    for tid in team_ids:
        try:
            team_stats[tid] = get_team_stats(tid, season)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"\n    ⚠ Failed to fetch team {tid}: {e}")
            team_stats[tid] = {"hitting": {}, "pitching": {}}
    print("Done")

    # Fetch pitcher stats
    pitcher_stats = {}
    print(f"  → Pitcher stats ({len(pitcher_ids)} pitchers)...", end=" ", flush=True)
    for pid in pitcher_ids:
        try:
            pitcher_stats[pid] = get_pitcher_stats(pid, season)
            time.sleep(0.1)
        except Exception as e:
            print(f"\n    ⚠ Failed to fetch pitcher {pid}: {e}")
            pitcher_stats[pid] = {}
    print("Done")

    return {
        "league_avg": league_avg,
        "team_stats": team_stats,
        "pitcher_stats": pitcher_stats,
    }


def predict_game(game: dict, data: dict) -> dict:
    """Generate prediction for a single game."""
    league_rpg = data["league_avg"]["avg_runs_per_game"]

    # Team factors
    away_ts = data["team_stats"].get(game["away_team_id"], {"hitting": {}, "pitching": {}})
    home_ts = data["team_stats"].get(game["home_team_id"], {"hitting": {}, "pitching": {}})

    away_off = compute_team_offensive_factor(away_ts["hitting"], league_rpg)
    home_off = compute_team_offensive_factor(home_ts["hitting"], league_rpg)
    away_def = compute_team_defensive_factor(away_ts["pitching"], league_rpg)
    home_def = compute_team_defensive_factor(home_ts["pitching"], league_rpg)

    # Pitcher factors
    away_sp = data["pitcher_stats"].get(game["away_pitcher_id"], {})
    home_sp = data["pitcher_stats"].get(game["home_pitcher_id"], {})

    away_sp_factor = compute_pitcher_factor(away_sp, away_ts["pitching"], league_rpg)
    home_sp_factor = compute_pitcher_factor(home_sp, home_ts["pitching"], league_rpg)

    # Park factor
    park_factor = get_park_factor(game.get("venue_id"))

    # Expected runs: away team faces home pitcher + home defense
    away_lambda = compute_expected_runs(
        batting_team_off_factor=away_off,
        pitching_team_def_factor=home_def,
        starter_factor=home_sp_factor,
        park_factor=park_factor,
        league_avg_rpg=league_rpg,
        is_home=False,
    )

    # Home team faces away pitcher + away defense
    home_lambda = compute_expected_runs(
        batting_team_off_factor=home_off,
        pitching_team_def_factor=away_def,
        starter_factor=away_sp_factor,
        park_factor=park_factor,
        league_avg_rpg=league_rpg,
        is_home=True,
    )

    prediction = generate_prediction(
        away_lambda, home_lambda,
        game["away_team_name"], game["home_team_name"],
    )
    prediction["away_pitcher"] = game["away_pitcher_name"]
    prediction["home_pitcher"] = game["home_pitcher_name"]
    prediction["venue"] = game.get("venue_name", "Unknown")
    prediction["game_id"] = game["game_id"]
    prediction["park_factor"] = park_factor

    return prediction


def print_predictions(predictions: list[dict], game_date: str) -> None:
    """Print formatted predictions for all games."""
    print("\n" + "=" * 90)
    print(f"  ⚾ MLB SCORE PREDICTIONS — {game_date}")
    print(f"  Model: Poisson Regression | Data: MLB Stats API | Games: {len(predictions)}")
    print("=" * 90)

    for i, p in enumerate(predictions, 1):
        away = p["away_team"]
        home = p["home_team"]
        a_score = p["predicted_score"]["away"]
        h_score = p["predicted_score"]["home"]
        a_exp = p["away_expected_runs"]
        h_exp = p["home_expected_runs"]
        a_wp = p["win_probability"]["away"]
        h_wp = p["win_probability"]["home"]

        print(f"\n{'─' * 90}")
        print(f"  Game {i}: {away} @ {home}")
        print(f"  📍 {p['venue']} | Park Factor: {p['park_factor']:.2f}")
        print(f"  🎯 Pitching: {p['away_pitcher']} vs {p['home_pitcher']}")
        print(f"{'─' * 90}")

        # Predicted score
        winner_marker_a = " ◀" if a_score > h_score else ""
        winner_marker_h = " ◀" if h_score > a_score else ""
        print(f"  PREDICTED FINAL SCORE:")
        print(f"    {away:<25} {a_score:>2}  (E[R] = {a_exp:.2f}){winner_marker_a}")
        print(f"    {home:<25} {h_score:>2}  (E[R] = {h_exp:.2f}){winner_marker_h}")
        if a_score == h_score:
            print(f"    → Model projects extras (coin flip)")

        # Win probability
        fav = away if a_wp > h_wp else home
        fav_pct = max(a_wp, h_wp)
        print(f"\n  WIN PROBABILITY:")
        print(f"    {away:<25} {a_wp:>5.1f}%")
        print(f"    {home:<25} {h_wp:>5.1f}%")
        print(f"    → Favorite: {fav} ({fav_pct:.1f}%)")

        # Over/Under
        ou = p["over_under"]
        print(f"\n  TOTAL RUNS:")
        print(f"    Expected Total: {p['expected_total']:.1f}")
        print(f"    O/U Line: {ou['line']:.1f}  |  Over {ou['over_pct']:.1f}% / Under {ou['under_pct']:.1f}%")

        # Top 5 most likely scores
        print(f"\n  TOP 5 MOST LIKELY SCORES:")
        for j, s in enumerate(p["top_5_scores"], 1):
            print(f"    {j}. {away} {s['away']} - {home} {s['home']}  ({s['probability']:.1f}%)")

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY — ALL GAMES")
    print(f"{'=' * 90}")
    print(f"  {'Matchup':<45} {'Score':>8} {'Win%':>8} {'Total':>7} {'O/U':>6}")
    print(f"  {'─' * 78}")

    for p in predictions:
        away = p["away_team"]
        home = p["home_team"]
        a_s = p["predicted_score"]["away"]
        h_s = p["predicted_score"]["home"]
        fav_pct = max(p["win_probability"]["away"], p["win_probability"]["home"])
        fav = "A" if p["win_probability"]["away"] > p["win_probability"]["home"] else "H"
        total = p["expected_total"]
        ou_line = p["over_under"]["line"]

        # Abbreviate team names for summary
        away_short = away.split()[-1][:4]
        home_short = home.split()[-1][:4]
        matchup = f"{away_short} @ {home_short}"

        print(f"  {matchup:<45} {a_s:>2}-{h_s:<2}   {fav}{fav_pct:>5.1f}%  {total:>6.1f}  {ou_line:>5.1f}")

    print(f"\n{'=' * 90}")
    print(f"  ⚠  Predictions are probabilistic estimates based on season stats.")
    print(f"     The most likely score has ~5-10% probability in any given game.")
    print(f"     Use expected runs (E[R]) and win% for more reliable insights.")
    print(f"{'=' * 90}\n")


def main():
    parser = argparse.ArgumentParser(description="MLB Score Prediction Model")
    parser.add_argument(
        "--date",
        type=str,
        default=str(date.today()),
        help="Game date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year to use for stats (default: previous season or current if data available)",
    )
    args = parser.parse_args()

    game_date = args.date
    # For early-season games, use prior year stats since current season has minimal data
    # For mid-season, use current year
    stats_season = args.season
    if stats_season is None:
        game_year = int(game_date.split("-")[0])
        game_month = int(game_date.split("-")[1])
        # If it's March/April (early season), use last year's full season stats
        if game_month <= 4:
            stats_season = game_year - 1
        else:
            stats_season = game_year

    print(f"\n⚾ MLB Score Prediction Model")
    print(f"   Game Date: {game_date}")
    print(f"   Stats Season: {stats_season}")

    # Fetch schedule
    print(f"\n📅 Fetching schedule for {game_date}...")
    games = get_schedule(game_date)

    if not games:
        print(f"\n❌ No games found for {game_date}. Check the date and try again.")
        sys.exit(1)

    print(f"   Found {len(games)} games")

    # Fetch all required data
    data = fetch_all_data(games, stats_season)

    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    predictions = []
    for game in games:
        pred = predict_game(game, data)
        predictions.append(pred)

    # Display results
    print_predictions(predictions, game_date)

    return predictions


if __name__ == "__main__":
    main()
