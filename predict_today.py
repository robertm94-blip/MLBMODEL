#!/usr/bin/env python3
"""MLB Score Prediction Model — Predict final scores for today's slate.

Uses a Poisson-based model powered by an aggregate of projection systems:
- FanGraphs Depth Charts projected standings (Steamer + ZiPS composite)
  → Team-level projected runs scored / allowed per game
- Steamer, ZiPS, THE BAT pitcher-level projections (blended equally)
  → Per-game starting pitcher quality adjustments
- Team bullpen projections (reliever ERA/FIP aggregated per team)
  → Dynamic starter/bullpen split based on projected IP/GS
- Park factors (multi-year historical averages)
- Live weather adjustments (temperature, wind, humidity, precipitation)
- Home field advantage

All projection data loaded from FanGraphs API JSON exports.
Weather data from Open-Meteo API (free, no key required).
"""

import sys
import time
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
from src.weather import fetch_game_weather, VENUE_DATA
from src.bullpen import (
    load_team_bullpen_projections,
    get_starter_projected_length,
    compute_game_pitching_factor,
    get_bullpen_summary,
)


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

    # Team bullpen projections
    bullpen_proj = load_team_bullpen_projections()
    if bullpen_proj:
        bp_eras = [bp["era"] for bp in bullpen_proj.values()]
        avg_bp = sum(bp_eras) / len(bp_eras)
        best = min(bullpen_proj.items(), key=lambda x: x[1]["era"])
        worst = max(bullpen_proj.items(), key=lambda x: x[1]["era"])
        print(f"  ✓ Bullpen projections: {len(bullpen_proj)} teams (avg {avg_bp:.2f} ERA)")
        print(f"    Best: {best[0].split()[-1]} ({best[1]['era']:.2f}) | "
              f"Worst: {worst[0].split()[-1]} ({worst[1]['era']:.2f})")
    else:
        print(f"  ⚠ No bullpen projections loaded")

    # League average
    league_avg = get_league_avg_rpg()
    print(f"  ✓ League avg: {league_avg:.3f} R/G (from projected standings)")

    return {
        "pitcher_projections": pitcher_proj,
        "batting_projections": batting_proj,
        "bullpen_projections": bullpen_proj,
        "league_avg_rpg": league_avg,
    }


def fetch_weather_for_games(games: list[dict]) -> dict[int, tuple]:
    """Fetch weather for all unique venues."""
    print(f"\n🌤️  Fetching live weather for game venues...\n")
    venue_weather = {}
    seen_venues = set()

    for game in games:
        vid = game.get("venue_id")
        if vid is None or vid in seen_venues:
            continue
        seen_venues.add(vid)

        weather, factor, breakdown = fetch_game_weather(vid)
        venue_weather[vid] = (weather, factor, breakdown)

        venue_name = VENUE_DATA.get(vid, {}).get("name", game.get("venue_name", "Unknown"))
        if weather:
            emoji = "☀️" if weather["weather_code"] <= 2 else "⛅" if weather["weather_code"] <= 3 else "🌧️"
            wind_dir_arrow = _wind_arrow(weather["wind_direction_deg"])
            print(f"  {emoji} {venue_name}: {weather['temp_f']:.0f}°F, "
                  f"{weather['wind_mph']:.0f}mph {wind_dir_arrow}, "
                  f"{weather['humidity']:.0f}% humidity"
                  f"{', Rain' if weather['precip_mm'] > 0 else ''}"
                  f" → adj: {factor:.3f}"
                  f"{' (roof closed)' if breakdown.get('roof', '').endswith('closed') else ''}"
                  f"{' (dome)' if breakdown.get('roof') == 'dome' else ''}")
        else:
            print(f"  ❓ {venue_name}: Weather unavailable → adj: 1.000")

        time.sleep(0.15)  # Rate limit

    return venue_weather


def _wind_arrow(degrees: float) -> str:
    """Convert wind direction (where FROM) to arrow character."""
    arrows = ["↓", "↙", "←", "↖", "↑", "↗", "→", "↘"]
    idx = int((degrees + 22.5) / 45) % 8
    return arrows[idx]


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


def predict_game(game: dict, data: dict, venue_weather: dict) -> dict:
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

    # Bullpen factors
    away_bp_factor = get_bullpen_summary(game["away_team_name"]).get("factor", 1.0)
    home_bp_factor = get_bullpen_summary(game["home_team_name"]).get("factor", 1.0)

    # Starter projected length (IP/GS)
    pitcher_proj = data["pitcher_projections"]
    away_sp_ip = get_starter_projected_length(game["away_pitcher_name"], pitcher_proj)
    home_sp_ip = get_starter_projected_length(game["home_pitcher_name"], pitcher_proj)

    # Blended game pitching factors (starter innings + bullpen for remainder)
    away_game_pitch = compute_game_pitching_factor(away_sp_factor, away_bp_factor, away_sp_ip)
    home_game_pitch = compute_game_pitching_factor(home_sp_factor, home_bp_factor, home_sp_ip)

    # Park factor
    park_factor = get_park_factor(game.get("venue_id"))

    # Weather adjustment
    vid = game.get("venue_id")
    weather_data, weather_factor, weather_breakdown = venue_weather.get(
        vid, (None, 1.0, {"total": 1.0, "roof": "unknown"})
    )

    # Expected runs: away team batting vs home game pitching (starter + bullpen)
    away_lambda = compute_expected_runs(
        batting_team_off_factor=away_off,
        pitching_team_def_factor=home_def,
        starter_factor=home_game_pitch,
        park_factor=park_factor,
        league_avg_rpg=league_rpg,
        is_home=False,
    ) * weather_factor

    # Home team batting vs away game pitching (starter + bullpen)
    home_lambda = compute_expected_runs(
        batting_team_off_factor=home_off,
        pitching_team_def_factor=away_def,
        starter_factor=away_game_pitch,
        park_factor=park_factor,
        league_avg_rpg=league_rpg,
        is_home=True,
    ) * weather_factor

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

    # Bullpen data
    prediction["away_bp_factor"] = round(away_bp_factor, 3)
    prediction["home_bp_factor"] = round(home_bp_factor, 3)
    prediction["away_sp_ip"] = round(away_sp_ip, 1)
    prediction["home_sp_ip"] = round(home_sp_ip, 1)
    prediction["away_game_pitch"] = round(away_game_pitch, 3)
    prediction["home_game_pitch"] = round(home_game_pitch, 3)

    # Weather data
    prediction["weather"] = weather_data
    prediction["weather_factor"] = round(weather_factor, 4)
    prediction["weather_breakdown"] = weather_breakdown

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

        # Weather line
        w = p.get("weather")
        wf = p.get("weather_factor", 1.0)
        wb = p.get("weather_breakdown", {})
        if w:
            emoji = "☀️" if w["weather_code"] <= 2 else "⛅" if w["weather_code"] <= 3 else "🌧️"
            wind_arrow = _wind_arrow(w["wind_direction_deg"])
            roof_note = ""
            if wb.get("roof") == "dome":
                roof_note = " (dome — no weather effect)"
            elif wb.get("roof", "").endswith("closed"):
                roof_note = " (roof closed — no weather effect)"
            weather_impact = ""
            if abs(wf - 1.0) >= 0.005:
                direction = "↑ runs" if wf > 1.0 else "↓ runs"
                weather_impact = f" | Impact: {direction} ({wf:.3f}x)"
                # Show component breakdown if significant
                components = []
                if abs(wb.get("temp", 0)) >= 0.005:
                    components.append(f"temp {'+'if wb['temp']>0 else ''}{wb['temp']:.1%}")
                if abs(wb.get("wind", 0)) >= 0.005:
                    components.append(f"wind {'+'if wb['wind']>0 else ''}{wb['wind']:.1%}")
                if abs(wb.get("humidity", 0)) >= 0.003:
                    components.append(f"humid {'+'if wb['humidity']>0 else ''}{wb['humidity']:.1%}")
                if wb.get("precip", 0) != 0:
                    components.append(f"rain {wb['precip']:.1%}")
                if components:
                    weather_impact += f" [{', '.join(components)}]"
            print(f"  {emoji} Weather: {w['temp_f']:.0f}°F, {w['wind_mph']:.0f}mph {wind_arrow} "
                  f"({w['weather_desc']}), {w['humidity']:.0f}% humidity"
                  f"{roof_note}{weather_impact}")
        else:
            print(f"  ❓ Weather: unavailable")

        print(f"  📈 Projected Season: {away.split()[-1]} {p['away_proj_wins']:.0f}W | "
              f"{home.split()[-1]} {p['home_proj_wins']:.0f}W")

        # Pitcher info with bullpen context
        a_sys = p["away_sp_systems"]
        h_sys = p["home_sp_systems"]
        print(f"  🎯 {p['away_pitcher']} (SP: {p['away_sp_factor']:.3f}, ~{p.get('away_sp_ip',5.0):.1f} IP) "
              f"→ pen ({p.get('away_bp_factor',1.0):.3f}) → game: {p.get('away_game_pitch',1.0):.3f}")
        if p["away_sp_breakdown"]:
            print(f"     {_format_sp_breakdown(p['away_sp_breakdown'])}")
        print(f"  🎯 {p['home_pitcher']} (SP: {p['home_sp_factor']:.3f}, ~{p.get('home_sp_ip',5.0):.1f} IP) "
              f"→ pen ({p.get('home_bp_factor',1.0):.3f}) → game: {p.get('home_game_pitch',1.0):.3f}")
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
    print(f"  {'Matchup':<28} {'Score':>6} {'Win%':>7} {'E[Tot]':>6} {'O/U':>5}  {'SP(A/H)':>12} {'Wx':>6}")
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
        wf = p.get("weather_factor", 1.0)

        away_short = away.split()[-1][:5]
        home_short = home.split()[-1][:5]
        matchup = f"{away_short:<5} @ {home_short:<5}"
        sp_adj = f"{p['away_sp_factor']:.2f}/{p['home_sp_factor']:.2f}"

        print(f"  {matchup:<28} {a_s:>2}-{h_s:<2}  {fav}{fav_pct:>5.1f}% {total:>5.1f}  {ou_line:>4.1f}  {sp_adj:>12} {wf:>5.3f}")

    print(f"\n{'=' * 94}")
    print(f"  MODEL METHODOLOGY:")
    print(f"  • Team offense/defense: FanGraphs Depth Charts projected RS/RA per game")
    print(f"  • Pitcher adjustment: Blended ERA/FIP (Steamer + ZiPS + THE BAT)")
    print(f"  • Bullpen: IP-weighted reliever ERA/FIP per team, split by starter IP/GS")
    print(f"  • Park factors: Multi-year historical averages (30 venues)")
    print(f"  • Weather: Live conditions (Open-Meteo) — temp, wind, humidity, precip")
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

    # Fetch weather
    venue_weather = fetch_weather_for_games(games)

    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    predictions = []
    for game in games:
        pred = predict_game(game, data, venue_weather)
        predictions.append(pred)

    print_predictions(predictions, game_date)
    return predictions


if __name__ == "__main__":
    main()
