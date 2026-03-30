#!/usr/bin/env python3
"""MLB Full Model Prediction Engine — Market-Grade Projections.

Unified engine combining all model components:
1. Team-level projected RS/RA (FanGraphs Depth Charts)
2. Lineup-specific offensive factors (individual batter projections)
3. Starting pitcher quality (Steamer/ZiPS/THE BAT blend)
4. Bullpen quality with dynamic starter/pen IP split
5. L/R platoon adjustments (batter hand vs pitcher hand)
6. Home plate umpire strike zone tendencies
7. Rest and travel fatigue adjustments
8. Live weather (temp, wind, humidity, precipitation)
9. Park factors (multi-year historical)
10. Home field advantage

Output: Win probabilities, score predictions, and market edge analysis.
"""

import sys
import json
import time
import argparse
from datetime import date

from src.mlb_api import get_schedule
from src.projections import (
    get_team_projected_strength,
    load_all_pitcher_projections,
    get_league_avg_rpg,
)
from src.features import compute_expected_runs, get_park_factor, HOME_ADVANTAGE
from src.model import generate_prediction
from src.weather import fetch_game_weather, VENUE_DATA
from src.bullpen import (
    load_team_bullpen_projections,
    get_starter_projected_length,
    compute_game_pitching_factor,
    get_bullpen_summary,
)
from src.lineup_offense import compute_lineup_offensive_factor
from src.platoon import (
    load_pitcher_hands,
    load_batter_hands,
    compute_team_platoon_factor,
)
from src.umpire import get_umpire_factor
from src.rest_travel import compute_rest_travel_factor
from src.edge import analyze_game, analyze_total


def load_lineups(date_str: str) -> dict:
    """Load lineup data."""
    import os
    filepath = os.path.join(os.path.dirname(__file__),
                            f"lineups_{date_str.replace('-', '_')}.json")
    if not os.path.exists(filepath):
        return {}
    with open(filepath) as f:
        return json.load(f)


def sp_factor_from_proj(name: str, pitcher_proj: dict, league_avg: float) -> float:
    """Compute starter quality factor from blended projections."""
    p = pitcher_proj.get(name)
    if not p or p.get("era", 0) == 0:
        return 1.0

    era = p["era"]
    fip = p.get("fip", era)
    whip = p.get("whip", 1.30)
    ip = p.get("ip", 0)
    k9 = p.get("k9", 0)
    bb9 = p.get("bb9", 0)

    blended_run_rate = era * 0.40 + fip * 0.60
    era_factor = blended_run_rate / league_avg if league_avg > 0 else 1.0
    whip_factor = whip / 1.28 if whip > 0 else 1.0
    k_bb_factor = 1.0
    if k9 > 0 and bb9 > 0:
        k_bb_factor = 1.0 - ((k9 - bb9) - 5.0) * 0.015

    raw_factor = 0.70 * era_factor + 0.20 * whip_factor + 0.10 * k_bb_factor
    reliability = min(ip / 150.0, 1.0)
    return raw_factor * reliability + 1.0 * (1 - reliability)


def predict_game_full(game: dict, data: dict) -> dict:
    """Generate prediction using all model components."""
    league_rpg = data["league_avg_rpg"]
    pitcher_proj = data["pitcher_projections"]
    game_date = data["game_date"]

    # ── 1. TEAM BASELINE (Depth Charts RS/RA) ──
    away_team_proj = get_team_projected_strength(game["away_team_name"])
    home_team_proj = get_team_projected_strength(game["home_team_name"])

    away_team_off = away_team_proj["proj_off_factor"]
    home_team_off = home_team_proj["proj_off_factor"]
    away_team_def = away_team_proj["proj_def_factor"]
    home_team_def = home_team_proj["proj_def_factor"]

    # ── 2. LINEUP-SPECIFIC OFFENSE ──
    lineup_data = data.get("lineups", {})
    gid = str(game["game_id"])
    game_lineups = lineup_data.get("games", {}).get(gid)

    pitcher_hands = data.get("pitcher_hands", {})
    away_sp_hand = pitcher_hands.get(game.get("away_pitcher_id"), "R")
    home_sp_hand = pitcher_hands.get(game.get("home_pitcher_id"), "R")

    away_lineup_factor = away_team_off  # Fallback to team-level
    home_lineup_factor = home_team_off
    lineup_used = False

    if game_lineups:
        away_lineup = [
            {"id": p["id"], "name": p["name"], "batting_order": p.get("batting_order", i + 1)}
            for i, p in enumerate(game_lineups["away_lineup"])
        ]
        home_lineup = [
            {"id": p["id"], "name": p["name"], "batting_order": p.get("batting_order", i + 1)}
            for i, p in enumerate(game_lineups["home_lineup"])
        ]

        away_lf, _ = compute_lineup_offensive_factor(away_lineup)
        home_lf, _ = compute_lineup_offensive_factor(home_lineup)

        # Blend lineup-specific (60%) with team-level (40%) for stability
        away_lineup_factor = away_lf * 0.60 + away_team_off * 0.40
        home_lineup_factor = home_lf * 0.60 + home_team_off * 0.40
        lineup_used = True

    # ── 3. PLATOON ADJUSTMENTS ──
    away_platoon = 1.0
    home_platoon = 1.0

    if game_lineups and data.get("batter_hands"):
        away_platoon = compute_team_platoon_factor(away_lineup, home_sp_hand)
        home_platoon = compute_team_platoon_factor(home_lineup, away_sp_hand)

    # Apply platoon to offense
    away_off_final = away_lineup_factor * away_platoon
    home_off_final = home_lineup_factor * home_platoon

    # ── 4. STARTING PITCHER ──
    away_sp = sp_factor_from_proj(game["away_pitcher_name"], pitcher_proj, league_rpg)
    home_sp = sp_factor_from_proj(game["home_pitcher_name"], pitcher_proj, league_rpg)

    # ── 5. BULLPEN + STARTER LENGTH ──
    away_bp = get_bullpen_summary(game["away_team_name"]).get("factor", 1.0)
    home_bp = get_bullpen_summary(game["home_team_name"]).get("factor", 1.0)
    away_sp_ip = get_starter_projected_length(game["away_pitcher_name"], pitcher_proj)
    home_sp_ip = get_starter_projected_length(game["home_pitcher_name"], pitcher_proj)

    away_game_pitch = compute_game_pitching_factor(away_sp, away_bp, away_sp_ip)
    home_game_pitch = compute_game_pitching_factor(home_sp, home_bp, home_sp_ip)

    # ── 6. UMPIRE ──
    ump_factor, ump_name = get_umpire_factor(game["game_id"], game_date)

    # ── 7. REST / TRAVEL ──
    away_rest, away_rest_info = compute_rest_travel_factor(
        game["away_team_id"], game["away_team_name"],
        game.get("venue_id"), game_date, is_home=False,
    )
    home_rest, home_rest_info = compute_rest_travel_factor(
        game["home_team_id"], game["home_team_name"],
        game.get("venue_id"), game_date, is_home=True,
    )

    # ── 8. WEATHER ──
    weather_cache = data.get("weather", {})
    vid = game.get("venue_id")
    weather_data, weather_factor, weather_breakdown = weather_cache.get(
        vid, (None, 1.0, {"total": 1.0, "roof": "unknown"})
    )

    # ── 9. PARK FACTOR ──
    park_factor = get_park_factor(vid)

    # ── COMPUTE EXPECTED RUNS ──
    # Away team: batting (lineup + platoon) vs home pitching (SP + bullpen)
    # Adjusted by: park, weather, umpire, rest/travel
    away_lambda = compute_expected_runs(
        batting_team_off_factor=away_off_final,
        pitching_team_def_factor=home_team_def,
        starter_factor=home_game_pitch,
        park_factor=park_factor,
        league_avg_rpg=league_rpg,
        is_home=False,
    ) * weather_factor * ump_factor * away_rest

    home_lambda = compute_expected_runs(
        batting_team_off_factor=home_off_final,
        pitching_team_def_factor=away_team_def,
        starter_factor=away_game_pitch,
        park_factor=park_factor,
        league_avg_rpg=league_rpg,
        is_home=True,
    ) * weather_factor * ump_factor * home_rest

    # ── GENERATE PREDICTION ──
    pred = generate_prediction(
        away_lambda, home_lambda,
        game["away_team_name"], game["home_team_name"],
    )

    # ── METADATA ──
    pred["away_pitcher"] = game["away_pitcher_name"]
    pred["home_pitcher"] = game["home_pitcher_name"]
    pred["venue"] = game.get("venue_name", "Unknown")
    pred["game_id"] = game["game_id"]
    pred["park_factor"] = park_factor
    pred["weather_factor"] = round(weather_factor, 4)
    pred["ump_name"] = ump_name
    pred["ump_factor"] = round(ump_factor, 4)
    pred["away_rest_factor"] = round(away_rest, 4)
    pred["home_rest_factor"] = round(home_rest, 4)
    pred["away_platoon"] = round(away_platoon, 4)
    pred["home_platoon"] = round(home_platoon, 4)
    pred["away_sp_factor"] = round(away_sp, 3)
    pred["home_sp_factor"] = round(home_sp, 3)
    pred["away_bp_factor"] = round(away_bp, 3)
    pred["home_bp_factor"] = round(home_bp, 3)
    pred["away_sp_ip"] = round(away_sp_ip, 1)
    pred["home_sp_ip"] = round(home_sp_ip, 1)
    pred["away_sp_hand"] = away_sp_hand
    pred["home_sp_hand"] = home_sp_hand
    pred["lineup_used"] = lineup_used
    pred["away_off_final"] = round(away_off_final, 4)
    pred["home_off_final"] = round(home_off_final, 4)
    pred["away_proj_wins"] = away_team_proj["proj_wins"]
    pred["home_proj_wins"] = home_team_proj["proj_wins"]

    # Factor breakdown for transparency
    pred["factor_breakdown"] = {
        "away_offense": round(away_off_final, 3),
        "home_offense": round(home_off_final, 3),
        "away_pitching": round(away_game_pitch, 3),
        "home_pitching": round(home_game_pitch, 3),
        "park": round(park_factor, 3),
        "weather": round(weather_factor, 3),
        "umpire": round(ump_factor, 3),
        "away_rest": round(away_rest, 3),
        "home_rest": round(home_rest, 3),
        "away_platoon": round(away_platoon, 3),
        "home_platoon": round(home_platoon, 3),
    }

    return pred


def print_full_output(predictions: list[dict], game_date: str) -> None:
    """Print compact market-ready output."""
    print(f"\n{'=' * 84}")
    print(f"  ⚾ MLB FULL MODEL — {game_date}")
    print(f"  10 factors: Lineup | Platoon | SP+Pen | Umpire | Rest | Weather | Park | Home")
    print(f"{'=' * 84}\n")

    all_games = []

    for i, p in enumerate(predictions, 1):
        away = p["away_team"]
        home = p["home_team"]
        a_wp = p["win_probability"]["away"]
        h_wp = p["win_probability"]["home"]
        a_er = p["away_expected_runs"]
        h_er = p["home_expected_runs"]
        total = p["expected_total"]

        away_short = away.split()[-1]
        home_short = home.split()[-1]

        fb = p["factor_breakdown"]

        print(f"  {'─' * 82}")
        print(f"  {i:>2}. {away_short} @ {home_short}")
        print(f"      Win%: {away_short} {a_wp:.1f}% | {home_short} {h_wp:.1f}%  "
              f"│  E[R]: {a_er:.2f} - {h_er:.2f}  │  Total: {total:.1f}")
        print(f"      SP: {p['away_pitcher']} ({p['away_sp_hand']}HP, {p['away_sp_factor']:.3f}, "
              f"~{p['away_sp_ip']:.0f}IP) vs "
              f"{p['home_pitcher']} ({p['home_sp_hand']}HP, {p['home_sp_factor']:.3f}, "
              f"~{p['home_sp_ip']:.0f}IP)")
        print(f"      Pen: {p['away_bp_factor']:.3f} / {p['home_bp_factor']:.3f}  │  "
              f"Platoon: {p['away_platoon']:.3f} / {p['home_platoon']:.3f}  │  "
              f"Ump: {p['ump_name']} ({p['ump_factor']:.3f})")
        print(f"      Park: {p['park_factor']:.2f}  │  Wx: {p['weather_factor']:.3f}  │  "
              f"Rest: {p['away_rest_factor']:.3f} / {p['home_rest_factor']:.3f}  │  "
              f"Lineup: {'✓' if p['lineup_used'] else '✗'}")

        all_games.append(p)

    # Summary table
    print(f"\n{'=' * 84}")
    print(f"  SUMMARY")
    print(f"{'=' * 84}")
    print(f"  {'MATCHUP':<24} {'AWAY%':>6} {'HOME%':>6} {'E[R]':>10} {'TOTAL':>6} {'PLAT':>10} {'UMP':>6}")
    print(f"  {'─' * 74}")

    for p in all_games:
        away_short = p["away_team"].split()[-1][:5]
        home_short = p["home_team"].split()[-1][:5]
        matchup = f"{away_short} @ {home_short}"
        a_wp = p["win_probability"]["away"]
        h_wp = p["win_probability"]["home"]
        a_er = p["away_expected_runs"]
        h_er = p["home_expected_runs"]
        total = p["expected_total"]
        plat = f"{p['away_platoon']:.2f}/{p['home_platoon']:.2f}"
        ump = f"{p['ump_factor']:.3f}"

        marker_a = "◀" if a_wp > h_wp else " "
        marker_h = "◀" if h_wp > a_wp else " "

        print(f"  {matchup:<24} {a_wp:>5.1f}%{marker_a} {h_wp:>5.1f}%{marker_h} "
              f"{a_er:>4.1f}-{h_er:<4.1f} {total:>5.1f}  {plat:>10} {ump:>6}")

    print(f"  {'─' * 74}")

    print(f"\n{'=' * 84}")
    print(f"  MODEL COMPONENTS:")
    print(f"   1. Team RS/RA (FanGraphs Depth Charts)")
    print(f"   2. Lineup-specific offense (individual batter wOBA/wRC+)")
    print(f"   3. L/R platoon adjustments (batter hand vs pitcher hand)")
    print(f"   4. Starting pitcher quality (3-system ERA/FIP/WHIP/K-BB blend)")
    print(f"   5. Bullpen quality (IP-weighted reliever ERA/FIP per team)")
    print(f"   6. Dynamic SP/pen split (projected IP/GS per starter)")
    print(f"   7. Home plate umpire zone tendency")
    print(f"   8. Rest & travel fatigue (schedule, timezone, road streak)")
    print(f"   9. Live weather (temp, wind direction, humidity, precip)")
    print(f"  10. Park factors + home field advantage")
    print(f"{'=' * 84}\n")


def main():
    parser = argparse.ArgumentParser(description="MLB Full Model Predictions")
    parser.add_argument("--date", type=str, default=str(date.today()))
    parser.add_argument("--odds", type=str, default=None,
                        help="JSON file with market odds for edge analysis")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    args = parser.parse_args()

    game_date = args.date

    print(f"\n⚾ MLB Full Model Engine")
    print(f"   Date: {game_date}")

    # Load schedule
    games = get_schedule(game_date)
    if not games:
        print(f"❌ No games found for {game_date}")
        sys.exit(1)
    print(f"   Games: {len(games)}")

    # Load all projection data
    print(f"\n📊 Loading data...")
    pitcher_proj = load_all_pitcher_projections()
    print(f"   ✓ {len(pitcher_proj)} pitcher projections")

    bullpen_proj = load_team_bullpen_projections()
    print(f"   ✓ {len(bullpen_proj)} team bullpen projections")

    pitcher_hands = load_pitcher_hands()
    batter_hands = load_batter_hands()
    print(f"   ✓ {len(pitcher_hands)} pitcher hands, {len(batter_hands)} batter hands")

    lineups = load_lineups(game_date)
    n_lineups = len(lineups.get("games", {}))
    print(f"   ✓ {n_lineups} game lineups")

    league_avg = get_league_avg_rpg()
    print(f"   ✓ League avg: {league_avg:.3f} R/G")

    # Fetch weather
    print(f"\n🌤️  Fetching weather...")
    weather_cache = {}
    for g in games:
        vid = g.get("venue_id")
        if vid and vid not in weather_cache:
            w, wf, wb = fetch_game_weather(vid)
            weather_cache[vid] = (w, wf, wb)
            time.sleep(0.1)
    print(f"   ✓ {len(weather_cache)} venues")

    # Fetch rest/travel (this hits the API for each team)
    print(f"   Fetching rest/travel data...")

    # Bundle all data
    data = {
        "pitcher_projections": pitcher_proj,
        "bullpen_projections": bullpen_proj,
        "pitcher_hands": pitcher_hands,
        "batter_hands": batter_hands,
        "lineups": lineups,
        "league_avg_rpg": league_avg,
        "weather": weather_cache,
        "game_date": game_date,
    }

    # Generate predictions
    print(f"\n🔮 Generating predictions...\n")
    predictions = []
    for game in games:
        pred = predict_game_full(game, data)
        predictions.append(pred)

    print_full_output(predictions, game_date)

    return predictions


if __name__ == "__main__":
    main()
