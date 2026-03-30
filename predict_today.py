#!/usr/bin/env python3
"""MLB Score Prediction Model — Predict final scores for today's slate.

Uses a Poisson-based model powered by an aggregate of projection systems:
- FanGraphs Depth Charts (50/50 Steamer + ZiPS blend)
- ZiPS standalone projected standings
- Optional CSV imports: Steamer, ZiPS, THE BAT pitcher-level projections

Team strength is derived from projected WAR splits (offense vs defense),
then adjusted per-game by starting pitcher quality and park factors.

Data sources:
- Pre-season projections: FanGraphs Depth Charts + ZiPS (compiled)
- Live pitcher stats: MLB Stats API (fallback when CSV projections unavailable)
- Park factors: Multi-year historical averages
"""

import sys
import time
import argparse
from datetime import date

from src.mlb_api import get_schedule, get_team_stats, get_pitcher_stats, get_league_averages
from src.projections import (
    get_team_projected_strength,
    load_all_pitcher_projections,
    LEAGUE_AVG_RPG_2026,
)
from src.features import (
    compute_pitcher_factor,
    compute_expected_runs,
    get_park_factor,
    safe_float,
)
from src.model import generate_prediction


def fetch_all_data(games: list[dict], season: int) -> dict:
    """Fetch all required stats for the day's games."""
    print(f"\n📊 Loading projection data...\n")

    # Load blended pitcher projections from CSVs (if available)
    pitcher_projections = load_all_pitcher_projections()
    if pitcher_projections:
        print(f"  ✓ Loaded blended pitcher projections for {len(pitcher_projections)} pitchers")
    else:
        print(f"  ⚠ No pitcher projection CSVs found in data/")
        print(f"    To use Steamer/ZiPS/THE BAT projections, export CSVs from FanGraphs:")
        print(f"    → data/steamer_pitchers.csv")
        print(f"    → data/zips_pitchers.csv")
        print(f"    → data/thebat_pitchers.csv")

    # Load team projection strengths (always available — compiled from FanGraphs)
    print(f"\n  ✓ Team projections loaded (Depth Charts + ZiPS blend)")

    # Fetch live pitcher stats from API as fallback
    print(f"\n📊 Fetching {season} pitcher stats from MLB Stats API...\n")

    # Get league averages from API for pitcher factor computation
    print("  → League averages...", end=" ", flush=True)
    league_avg = get_league_averages(season)
    print(f"Done (avg {league_avg['avg_runs_per_game']:.2f} R/G)")

    # Collect unique pitcher IDs
    pitcher_ids = set()
    team_ids = set()
    for g in games:
        if g["away_pitcher_id"]:
            pitcher_ids.add(g["away_pitcher_id"])
        if g["home_pitcher_id"]:
            pitcher_ids.add(g["home_pitcher_id"])
        team_ids.add(g["away_team_id"])
        team_ids.add(g["home_team_id"])

    # Fetch individual pitcher stats from API
    api_pitcher_stats = {}
    print(f"  → Pitcher stats ({len(pitcher_ids)} pitchers)...", end=" ", flush=True)
    for pid in pitcher_ids:
        try:
            api_pitcher_stats[pid] = get_pitcher_stats(pid, season)
            time.sleep(0.1)
        except Exception as e:
            print(f"\n    ⚠ Failed to fetch pitcher {pid}: {e}")
            api_pitcher_stats[pid] = {}
    print("Done")

    # Fetch team pitching stats (for pitcher factor baseline)
    team_stats = {}
    print(f"  → Team pitching baselines ({len(team_ids)} teams)...", end=" ", flush=True)
    for tid in team_ids:
        try:
            team_stats[tid] = get_team_stats(tid, season)
            time.sleep(0.1)
        except Exception as e:
            team_stats[tid] = {"hitting": {}, "pitching": {}}
    print("Done")

    return {
        "league_avg": league_avg,
        "team_stats": team_stats,
        "api_pitcher_stats": api_pitcher_stats,
        "pitcher_projections": pitcher_projections,
    }


def compute_pitcher_adjustment(
    pitcher_name: str,
    pitcher_id: int | None,
    team_name: str,
    data: dict,
) -> float:
    """Compute starting pitcher quality factor from best available source.

    Priority:
    1. Blended projection CSVs (Steamer/ZiPS/THE BAT) if available
    2. MLB Stats API season stats (prior year)
    3. Default to league average (1.0)
    """
    league_rpg = data["league_avg"]["avg_runs_per_game"]

    # Try projection CSVs first
    proj = data["pitcher_projections"].get(pitcher_name)
    if proj and proj.get("era", 0) > 0 and proj.get("ip", 0) >= 10:
        # Use projected ERA + FIP blend for stability
        proj_era = proj["era"]
        proj_fip = proj.get("fip", proj_era)
        proj_whip = proj.get("whip", 1.30)
        proj_ip = proj.get("ip", 100)

        # ERA/FIP blend (FIP is more predictive than ERA)
        blended_era = proj_era * 0.4 + proj_fip * 0.6

        league_era = league_rpg  # ~4.5 scale
        era_factor = blended_era / league_era if league_era > 0 else 1.0
        whip_factor = proj_whip / 1.30

        # Weight ERA/FIP blend 75%, WHIP 25%
        raw_factor = 0.75 * era_factor + 0.25 * whip_factor

        # Projected IP reliability (full weight at 150+ IP projection)
        reliability = min(proj_ip / 150.0, 1.0)
        factor = raw_factor * reliability + 1.0 * (1 - reliability)

        return factor

    # Fallback: MLB Stats API
    api_stats = data["api_pitcher_stats"].get(pitcher_id, {})
    if api_stats:
        # Find team pitching stats for baseline
        team_pitching = {}
        for tid, ts in data["team_stats"].items():
            team_pitching = ts.get("pitching", {})
            break  # just need a reasonable baseline
        return compute_pitcher_factor(api_stats, team_pitching, league_rpg)

    # No data available
    return 1.0


def predict_game(game: dict, data: dict) -> dict:
    """Generate prediction for a single game using projection-based strength."""
    # Get projection-based team strength
    away_proj = get_team_projected_strength(game["away_team_name"])
    home_proj = get_team_projected_strength(game["home_team_name"])

    # Team offensive and defensive factors from projections
    away_off = away_proj["proj_off_factor"]
    home_off = home_proj["proj_off_factor"]
    away_def = away_proj["proj_def_factor"]  # >1.0 = worse defense (allows more runs)
    home_def = home_proj["proj_def_factor"]

    # Pitcher adjustments (from projections or API)
    away_sp_factor = compute_pitcher_adjustment(
        game["away_pitcher_name"], game["away_pitcher_id"],
        game["away_team_name"], data,
    )
    home_sp_factor = compute_pitcher_adjustment(
        game["home_pitcher_name"], game["home_pitcher_id"],
        game["home_team_name"], data,
    )

    # Park factor
    park_factor = get_park_factor(game.get("venue_id"))
    league_rpg = LEAGUE_AVG_RPG_2026

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
    prediction["away_proj_wins"] = away_proj["blended_wins"]
    prediction["home_proj_wins"] = home_proj["blended_wins"]
    prediction["away_sp_factor"] = round(away_sp_factor, 3)
    prediction["home_sp_factor"] = round(home_sp_factor, 3)

    return prediction


def print_predictions(predictions: list[dict], game_date: str, has_csv_projections: bool) -> None:
    """Print formatted predictions for all games."""
    sources = "Depth Charts (Steamer+ZiPS)"
    if has_csv_projections:
        sources += " + Pitcher CSVs"
    sources += " + MLB API"

    print("\n" + "=" * 92)
    print(f"  ⚾ MLB SCORE PREDICTIONS — {game_date}")
    print(f"  Model: Poisson Regression | Sources: {sources}")
    print(f"  Games: {len(predictions)}")
    print("=" * 92)

    for i, p in enumerate(predictions, 1):
        away = p["away_team"]
        home = p["home_team"]
        a_score = p["predicted_score"]["away"]
        h_score = p["predicted_score"]["home"]
        a_exp = p["away_expected_runs"]
        h_exp = p["home_expected_runs"]
        a_wp = p["win_probability"]["away"]
        h_wp = p["win_probability"]["home"]

        print(f"\n{'─' * 92}")
        print(f"  Game {i}: {away} @ {home}")
        print(f"  📍 {p['venue']} | Park Factor: {p['park_factor']:.2f}")
        print(f"  🎯 Pitching: {p['away_pitcher']} (adj: {p['away_sp_factor']:.3f}) vs "
              f"{p['home_pitcher']} (adj: {p['home_sp_factor']:.3f})")
        print(f"  📈 Proj Wins: {away.split()[-1]} {p['away_proj_wins']:.0f}W | "
              f"{home.split()[-1]} {p['home_proj_wins']:.0f}W")
        print(f"{'─' * 92}")

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
    print(f"\n{'=' * 92}")
    print(f"  SUMMARY — ALL GAMES")
    print(f"{'=' * 92}")
    print(f"  {'Matchup':<40} {'Score':>7} {'Win%':>8} {'Total':>7} {'O/U':>6}  {'SP Adj':>12}")
    print(f"  {'─' * 84}")

    for p in predictions:
        away = p["away_team"]
        home = p["home_team"]
        a_s = p["predicted_score"]["away"]
        h_s = p["predicted_score"]["home"]
        fav_pct = max(p["win_probability"]["away"], p["win_probability"]["home"])
        fav = "A" if p["win_probability"]["away"] > p["win_probability"]["home"] else "H"
        total = p["expected_total"]
        ou_line = p["over_under"]["line"]

        away_short = away.split()[-1][:5]
        home_short = home.split()[-1][:5]
        matchup = f"{away_short} @ {home_short}"
        sp_adj = f"{p['away_sp_factor']:.2f}/{p['home_sp_factor']:.2f}"

        print(f"  {matchup:<40} {a_s:>2}-{h_s:<2}  {fav}{fav_pct:>5.1f}%  {total:>6.1f}  {ou_line:>5.1f}  {sp_adj:>12}")

    print(f"\n{'=' * 92}")
    print(f"  MODEL METHODOLOGY:")
    print(f"  • Team strength: Blended Depth Charts (Steamer+ZiPS) + ZiPS projected W/L & WAR")
    print(f"  • Pitcher adj: {'Blended CSV projections (Steamer/ZiPS/THE BAT)' if has_csv_projections else 'MLB Stats API (prior season ERA/WHIP/IP)'}")
    print(f"  • Park factors: Multi-year historical averages (30 venues)")
    print(f"  • Home advantage: +0.25 runs | Starter weight: 55% of game")
    print(f"  • Score model: Independent Poisson distributions per team")
    print(f"{'─' * 92}")
    print(f"  ⚠  The most likely exact score has ~5-10% probability.")
    print(f"     Expected runs (E[R]) and win% are more reliable signals.")
    print(f"{'=' * 92}\n")


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
        help="Season year for API pitcher stats (default: auto-detect)",
    )
    args = parser.parse_args()

    game_date = args.date
    stats_season = args.season
    if stats_season is None:
        game_year = int(game_date.split("-")[0])
        game_month = int(game_date.split("-")[1])
        if game_month <= 4:
            stats_season = game_year - 1
        else:
            stats_season = game_year

    print(f"\n⚾ MLB Score Prediction Model (Projection-Based)")
    print(f"   Game Date: {game_date}")
    print(f"   Team Projections: 2026 Depth Charts + ZiPS (blended)")
    print(f"   Pitcher Stats Season: {stats_season}")

    # Fetch schedule
    print(f"\n📅 Fetching schedule for {game_date}...")
    games = get_schedule(game_date)

    if not games:
        print(f"\n❌ No games found for {game_date}.")
        sys.exit(1)

    print(f"   Found {len(games)} games")

    # Fetch all required data
    data = fetch_all_data(games, stats_season)
    has_csv = bool(data["pitcher_projections"])

    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    predictions = []
    for game in games:
        pred = predict_game(game, data)
        predictions.append(pred)

    # Display results
    print_predictions(predictions, game_date, has_csv)

    return predictions


if __name__ == "__main__":
    main()
