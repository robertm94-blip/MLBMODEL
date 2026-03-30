#!/usr/bin/env python3
"""MLB Score Prediction Model — Predict final scores for today's slate.

Uses a Poisson-based model powered by an aggregate of projection systems:
- FanGraphs Depth Charts projected standings (Steamer + ZiPS composite)
  → Team-level projected runs scored / allowed per game
- Steamer, ZiPS, THE BAT pitcher-level projections (blended equally)
  → Per-game starting pitcher quality adjustments
- Park factors (multi-year historical averages)
- Home field advantage

All projection data loaded from FanGraphs API JSON exports.
"""

import sys
import argparse
from datetime import date

from src.mlb_api import get_schedule
from src.projections import (
    get_team_projected_strength,
    load_all_pitcher_projections,
    load_team_batting_projections,
    get_league_avg_rpg,
)
from src.features import (
    compute_expected_runs,
    get_park_factor,
)
from src.model import generate_prediction


def load_projection_data() -> dict:
    """Load all projection data from JSON files."""
    print(f"\n📊 Loading projection data...\n")

    # Pitcher projections (Steamer + ZiPS + THE BAT blend)
    pitcher_proj = load_all_pitcher_projections()
    systems_summary = {}
    for name, p in pitcher_proj.items():
        for sys in p.get("systems", []):
            systems_summary[sys] = systems_summary.get(sys, 0) + 1

    if pitcher_proj:
        print(f"  ✓ Pitcher projections: {len(pitcher_proj)} pitchers")
        for sys, count in sorted(systems_summary.items()):
            print(f"    • {sys.upper()}: {count} pitchers")
    else:
        print(f"  ⚠ No pitcher projections loaded")

    # Team batting projections (aggregated from individual batters)
    batting_proj = load_team_batting_projections()
    if batting_proj:
        print(f"  ✓ Team batting projections: {len(batting_proj)} teams")
    else:
        print(f"  ⚠ No batter projections loaded")

    # League average
    league_avg = get_league_avg_rpg()
    print(f"  ✓ League avg: {league_avg:.3f} R/G (from projected standings)")

    return {
        "pitcher_projections": pitcher_proj,
        "batting_projections": batting_proj,
        "league_avg_rpg": league_avg,
    }


def compute_pitcher_adjustment(
    pitcher_name: str,
    data: dict,
) -> tuple[float, dict | None]:
    """Compute starting pitcher quality factor from blended projections.

    Uses ERA/FIP blend (FIP weighted more heavily as it's more predictive)
    plus WHIP as a secondary signal, with reliability weighting by
    projected IP.

    Returns (factor, projection_data_or_None).
    Factor < 1.0 = pitcher suppresses scoring, > 1.0 = inflates scoring.
    """
    proj = data["pitcher_projections"].get(pitcher_name)
    if not proj or proj.get("era", 0) == 0:
        return 1.0, None

    league_avg = data["league_avg_rpg"]

    era = proj["era"]
    fip = proj.get("fip", era)
    whip = proj.get("whip", 1.30)
    ip = proj.get("ip", 0)
    k9 = proj.get("k9", 0)
    bb9 = proj.get("bb9", 0)

    # ERA/FIP blend: FIP is more predictive, so weight it 60%
    blended_run_rate = era * 0.40 + fip * 0.60

    # Normalize to league average
    era_factor = blended_run_rate / league_avg if league_avg > 0 else 1.0

    # WHIP factor (secondary signal)
    league_whip = 1.28
    whip_factor = whip / league_whip if whip > 0 else 1.0

    # K-BB differential bonus/penalty (tertiary signal)
    # Elite K-BB% further suppresses; bad K-BB% inflates
    k_bb_factor = 1.0
    if k9 > 0 and bb9 > 0:
        k_bb_diff = k9 - bb9
        league_k_bb = 5.0  # ~average K/9 - BB/9 difference
        k_bb_factor = 1.0 - (k_bb_diff - league_k_bb) * 0.015

    # Weighted combination: ERA/FIP 70%, WHIP 20%, K-BB 10%
    raw_factor = (0.70 * era_factor) + (0.20 * whip_factor) + (0.10 * k_bb_factor)

    # Reliability weighting by projected IP (full confidence at 150+ IP)
    reliability = min(ip / 150.0, 1.0)
    factor = raw_factor * reliability + 1.0 * (1 - reliability)

    return factor, proj


def predict_game(game: dict, data: dict) -> dict:
    """Generate prediction for a single game."""
    league_rpg = data["league_avg_rpg"]

    # Team strength from projected standings (RS/RA per game)
    away_proj = get_team_projected_strength(game["away_team_name"])
    home_proj = get_team_projected_strength(game["home_team_name"])

    away_off = away_proj["proj_off_factor"]
    home_off = home_proj["proj_off_factor"]
    away_def = away_proj["proj_def_factor"]
    home_def = home_proj["proj_def_factor"]

    # Starting pitcher adjustments from blended projections
    away_sp_factor, away_sp_data = compute_pitcher_adjustment(
        game["away_pitcher_name"], data,
    )
    home_sp_factor, home_sp_data = compute_pitcher_adjustment(
        game["home_pitcher_name"], data,
    )

    # Park factor
    park_factor = get_park_factor(game.get("venue_id"))

    # Expected runs: away team batting vs home pitching + home starter
    away_lambda = compute_expected_runs(
        batting_team_off_factor=away_off,
        pitching_team_def_factor=home_def,
        starter_factor=home_sp_factor,
        park_factor=park_factor,
        league_avg_rpg=league_rpg,
        is_home=False,
    )

    # Home team batting vs away pitching + away starter
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

    # Add metadata
    prediction["away_pitcher"] = game["away_pitcher_name"]
    prediction["home_pitcher"] = game["home_pitcher_name"]
    prediction["venue"] = game.get("venue_name", "Unknown")
    prediction["game_id"] = game["game_id"]
    prediction["park_factor"] = park_factor
    prediction["away_proj_wins"] = away_proj["proj_wins"]
    prediction["home_proj_wins"] = home_proj["proj_wins"]
    prediction["away_sp_factor"] = round(away_sp_factor, 3)
    prediction["home_sp_factor"] = round(home_sp_factor, 3)

    # Per-system pitcher breakdown
    prediction["away_sp_breakdown"] = (
        away_sp_data.get("by_system", {}) if away_sp_data else {}
    )
    prediction["home_sp_breakdown"] = (
        home_sp_data.get("by_system", {}) if home_sp_data else {}
    )
    prediction["away_sp_systems"] = (
        away_sp_data.get("systems_count", 0) if away_sp_data else 0
    )
    prediction["home_sp_systems"] = (
        home_sp_data.get("systems_count", 0) if home_sp_data else 0
    )

    return prediction


def _format_sp_breakdown(breakdown: dict) -> str:
    """Format per-system pitcher ERA/FIP for display."""
    if not breakdown:
        return "No projections"
    parts = []
    for sys, data in sorted(breakdown.items()):
        parts.append(f"{sys.upper()}: {data['era']:.2f}/{data['fip']:.2f} ERA/FIP")
    return " | ".join(parts)


def print_predictions(predictions: list[dict], game_date: str) -> None:
    """Print formatted predictions for all games."""
    print("\n" + "=" * 94)
    print(f"  ⚾ MLB SCORE PREDICTIONS — {game_date}")
    print(f"  Model: Poisson Regression")
    print(f"  Sources: Steamer + ZiPS + THE BAT (blended) | FanGraphs Depth Charts Standings")
    print(f"  Games: {len(predictions)}")
    print("=" * 94)

    for i, p in enumerate(predictions, 1):
        away = p["away_team"]
        home = p["home_team"]
        a_score = p["predicted_score"]["away"]
        h_score = p["predicted_score"]["home"]
        a_exp = p["away_expected_runs"]
        h_exp = p["home_expected_runs"]
        a_wp = p["win_probability"]["away"]
        h_wp = p["win_probability"]["home"]

        print(f"\n{'─' * 94}")
        print(f"  Game {i}: {away} @ {home}")
        print(f"  📍 {p['venue']} | Park Factor: {p['park_factor']:.2f}")
        print(f"  📈 Projected Season: {away.split()[-1]} {p['away_proj_wins']:.0f}W | "
              f"{home.split()[-1]} {p['home_proj_wins']:.0f}W")

        # Pitcher info with per-system breakdown
        a_sys = p["away_sp_systems"]
        h_sys = p["home_sp_systems"]
        print(f"  🎯 {p['away_pitcher']} (adj: {p['away_sp_factor']:.3f}, {a_sys} systems)")
        if p["away_sp_breakdown"]:
            print(f"     {_format_sp_breakdown(p['away_sp_breakdown'])}")
        print(f"  🎯 {p['home_pitcher']} (adj: {p['home_sp_factor']:.3f}, {h_sys} systems)")
        if p["home_sp_breakdown"]:
            print(f"     {_format_sp_breakdown(p['home_sp_breakdown'])}")

        print(f"{'─' * 94}")

        # Predicted score
        winner_marker_a = " ◀" if a_score > h_score else ""
        winner_marker_h = " ◀" if h_score > a_score else ""
        print(f"  PREDICTED FINAL SCORE:")
        print(f"    {away:<25} {a_score:>2}  (E[R] = {a_exp:.2f}){winner_marker_a}")
        print(f"    {home:<25} {h_score:>2}  (E[R] = {h_exp:.2f}){winner_marker_h}")
        if a_score == h_score:
            print(f"    → Model projects extras")

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
    print(f"\n{'=' * 94}")
    print(f"  SUMMARY — ALL GAMES")
    print(f"{'=' * 94}")
    print(f"  {'Matchup':<34} {'Score':>7} {'Win%':>8} {'E[Total]':>8} {'O/U':>6}  {'SP (A/H)':>14}")
    print(f"  {'─' * 82}")

    for p in predictions:
        away = p["away_team"]
        home = p["home_team"]
        a_s = p["predicted_score"]["away"]
        h_s = p["predicted_score"]["home"]
        fav_pct = max(p["win_probability"]["away"], p["win_probability"]["home"])
        fav = "A" if p["win_probability"]["away"] > p["win_probability"]["home"] else "H"
        total = p["expected_total"]
        ou_line = p["over_under"]["line"]

        away_short = away.split()[-1][:6]
        home_short = home.split()[-1][:6]
        matchup = f"{away_short:<6} @ {home_short:<6}"
        sp_adj = f"{p['away_sp_factor']:.2f} / {p['home_sp_factor']:.2f}"

        print(f"  {matchup:<34} {a_s:>2}-{h_s:<2}  {fav}{fav_pct:>5.1f}%  {total:>7.1f}  {ou_line:>5.1f}  {sp_adj:>14}")

    print(f"\n{'=' * 94}")
    print(f"  MODEL METHODOLOGY:")
    print(f"  • Team offense/defense: FanGraphs Depth Charts projected RS/RA per game")
    print(f"  • Pitcher adjustment: Blended ERA/FIP (Steamer + ZiPS + THE BAT)")
    print(f"  • Park factors: Multi-year historical averages (30 venues)")
    print(f"  • Home advantage: +0.25 runs | Starter game share: 55%")
    print(f"  • Score distribution: Independent Poisson per team")
    print(f"{'─' * 94}")
    print(f"  ⚠  Exact scores have ~5-10% individual probability.")
    print(f"     E[R] and win% are the most reliable prediction outputs.")
    print(f"{'=' * 94}\n")


def main():
    parser = argparse.ArgumentParser(description="MLB Score Prediction Model")
    parser.add_argument(
        "--date", type=str, default=str(date.today()),
        help="Game date (YYYY-MM-DD, default: today)",
    )
    args = parser.parse_args()
    game_date = args.date

    print(f"\n⚾ MLB Score Prediction Model")
    print(f"   Game Date: {game_date}")
    print(f"   Projections: Steamer + ZiPS + THE BAT (blended)")
    print(f"   Standings: FanGraphs Depth Charts (Steamer + ZiPS composite)")

    # Fetch schedule
    print(f"\n📅 Fetching schedule for {game_date}...")
    games = get_schedule(game_date)

    if not games:
        print(f"\n❌ No games found for {game_date}.")
        sys.exit(1)

    print(f"   Found {len(games)} games")

    # Load projection data
    data = load_projection_data()

    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    predictions = []
    for game in games:
        pred = predict_game(game, data)
        predictions.append(pred)

    print_predictions(predictions, game_date)
    return predictions


if __name__ == "__main__":
    main()
