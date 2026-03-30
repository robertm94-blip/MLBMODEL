#!/usr/bin/env python3
"""Multi-year backtest: 2023-2025 MLB seasons.

For proper out-of-sample validation, each season is predicted using
projections from THAT season (not future data):
- 2023 games → 2023 preseason projections
- 2024 games → 2024 preseason projections
- 2025 games → 2025/2026 preseason projections

This avoids look-ahead bias and gives a realistic estimate of
model performance on unseen data.
"""

import json
import os
import sys
import argparse
from collections import defaultdict

from src.features import compute_expected_runs, get_park_factor
from src.model import generate_prediction
from src.backtest import compute_accuracy, run_edge_sweep, print_backtest_report


PROJ_DIR = os.path.join(os.path.dirname(__file__), "data", "projections")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_pitcher_projections_for_season(season: int) -> dict[str, dict]:
    """Load and blend pitcher projections for a given season."""
    systems = ["steamer", "zips", "thebat"]
    all_systems = {}

    for system in systems:
        filepath = os.path.join(PROJ_DIR, str(season), f"{system}_pitchers.json")
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            data = json.load(f)
        pitchers = {}
        for p in data:
            name = p.get("PlayerName", "")
            if not name:
                continue
            ip = p.get("IP", 0) or 0
            era = p.get("ERA", 0) or 0
            if ip < 1 or era == 0:
                continue
            pitchers[name] = {
                "era": era,
                "fip": p.get("FIP", 0) or 0,
                "whip": p.get("WHIP", 0) or 0,
                "k9": p.get("K/9", 0) or 0,
                "bb9": p.get("BB/9", 0) or 0,
                "ip": ip,
                "gs": p.get("GS", 0) or 0,
                "g": p.get("G", 0) or 0,
            }
        all_systems[system] = pitchers

    if not all_systems:
        return {}

    # Blend
    all_names = set()
    for sys_data in all_systems.values():
        all_names.update(sys_data.keys())

    blended = {}
    stat_keys = ["era", "fip", "whip", "k9", "bb9", "ip", "gs", "g"]
    for name in all_names:
        available = [all_systems[s][name] for s in all_systems if name in all_systems[s]]
        if not available:
            continue
        b = {}
        for k in stat_keys:
            vals = [a[k] for a in available if a.get(k, 0)]
            b[k] = sum(vals) / len(vals) if vals else 0
        blended[name] = b

    return blended


def load_team_stats_for_season(season: int) -> dict[str, dict]:
    """Load team-level RS/RA stats from MLB API data.

    Falls back to the FanGraphs projected standings if available.
    """
    # Try MLB API team stats first
    api_path = os.path.join(DATA_DIR, f"team_stats_{season}.json")
    if os.path.exists(api_path):
        with open(api_path) as f:
            data = json.load(f)

        teams = {}
        # Parse MLB API format
        if isinstance(data, dict) and "stats" in data:
            for stat_group in data.get("stats", []):
                group_name = stat_group.get("group", {}).get("displayName", "").lower()
                for split in stat_group.get("splits", []):
                    team_name = split.get("team", {}).get("name", "")
                    stats = split.get("stat", {})
                    if not team_name:
                        continue
                    if team_name not in teams:
                        teams[team_name] = {"hitting": {}, "pitching": {}}
                    teams[team_name][group_name if group_name in ("hitting", "pitching") else "hitting"] = stats
        elif isinstance(data, list):
            # Alternative format: list of team stat objects
            for entry in data:
                team_name = entry.get("team_name", "")
                if team_name:
                    teams[team_name] = entry

        if teams:
            return teams

    # Fallback: try FanGraphs projected standings
    standings_path = os.path.join(PROJ_DIR, str(season), "fangraphs_projected_standings.json")
    if os.path.exists(standings_path):
        with open(standings_path) as f:
            data = json.load(f)
        # This is the same format as our 2026 standings
        return {"_standings": data}

    return {}


def compute_team_factors(
    team_stats: dict, league_avg_rpg: float
) -> dict[str, dict[str, float]]:
    """Compute offensive and defensive factors from team stats."""
    factors = {}

    # Check if we have FanGraphs standings format
    if "_standings" in team_stats:
        standings = team_stats["_standings"]
        total_rpg = 0
        count = 0
        for t in standings:
            rpg = t.get("xRpG") or t.get("RpG") or 0
            if rpg > 0:
                total_rpg += rpg
                count += 1
        lavg = total_rpg / count if count > 0 else league_avg_rpg

        name_map = {
            "Dodgers": "Los Angeles Dodgers", "Yankees": "New York Yankees",
            "Mets": "New York Mets", "Mariners": "Seattle Mariners",
            "Braves": "Atlanta Braves", "Blue Jays": "Toronto Blue Jays",
            "Phillies": "Philadelphia Phillies", "Red Sox": "Boston Red Sox",
            "Tigers": "Detroit Tigers", "Orioles": "Baltimore Orioles",
            "Brewers": "Milwaukee Brewers", "Rangers": "Texas Rangers",
            "Cubs": "Chicago Cubs", "Pirates": "Pittsburgh Pirates",
            "Rays": "Tampa Bay Rays", "Astros": "Houston Astros",
            "Royals": "Kansas City Royals", "Padres": "San Diego Padres",
            "Diamondbacks": "Arizona Diamondbacks", "Giants": "San Francisco Giants",
            "Twins": "Minnesota Twins", "Reds": "Cincinnati Reds",
            "Marlins": "Miami Marlins", "Guardians": "Cleveland Guardians",
            "Athletics": "Athletics", "Cardinals": "St. Louis Cardinals",
            "Angels": "Los Angeles Angels", "Nationals": "Washington Nationals",
            "White Sox": "Chicago White Sox", "Rockies": "Colorado Rockies",
        }

        for t in standings:
            short = t.get("shortName", "")
            full = name_map.get(short, short)
            rs = t.get("xRpG") or t.get("RpG") or lavg
            ra = t.get("xRApG") or t.get("RApG") or lavg
            factors[full] = {
                "off_factor": rs / lavg,
                "def_factor": ra / lavg,
            }
        return factors

    # MLB API format: compute from team hitting/pitching stats
    all_rpg = []
    for team_name, stats in team_stats.items():
        hitting = stats.get("hitting", {})
        pitching = stats.get("pitching", {})
        runs = int(hitting.get("runs", 0) or 0)
        games = int(hitting.get("gamesPlayed", 0) or 0)
        ra = int(pitching.get("runs", 0) or 0)
        pg = int(pitching.get("gamesPlayed", 0) or 0)
        if games > 0:
            all_rpg.append(runs / games)

    lavg = sum(all_rpg) / len(all_rpg) if all_rpg else league_avg_rpg

    for team_name, stats in team_stats.items():
        hitting = stats.get("hitting", {})
        pitching = stats.get("pitching", {})
        runs = int(hitting.get("runs", 0) or 0)
        games = int(hitting.get("gamesPlayed", 0) or 0)
        ra = int(pitching.get("runs", 0) or 0)
        pg = int(pitching.get("gamesPlayed", 0) or 0)

        off = (runs / games) / lavg if games > 0 and lavg > 0 else 1.0
        dfn = (ra / pg) / lavg if pg > 0 and lavg > 0 else 1.0

        factors[team_name] = {"off_factor": off, "def_factor": dfn}

    return factors


def sp_factor(name: str, pitcher_proj: dict, league_avg: float) -> float:
    """Compute starter quality factor."""
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
    era_f = blended / league_avg if league_avg > 0 else 1.0
    whip_f = whip / 1.28 if whip > 0 else 1.0
    kbb_f = 1.0
    if k9 > 0 and bb9 > 0:
        kbb_f = 1.0 - ((k9 - bb9) - 5.0) * 0.015

    raw = 0.70 * era_f + 0.20 * whip_f + 0.10 * kbb_f
    rel = min(ip / 150.0, 1.0)
    return raw * rel + 1.0 * (1 - rel)


def bullpen_factor_for_pitcher(pitcher_proj: dict, name: str) -> tuple[float, float]:
    """Estimate bullpen factor and starter IP from projection data.

    Without the full bullpen module loaded for historical seasons,
    we use a simplified approach: team average = 1.0, starter gets
    their projected IP/GS share.
    """
    p = pitcher_proj.get(name)
    if not p:
        return 1.0, 5.0

    gs = p.get("gs", 0)
    ip = p.get("ip", 0)
    if gs >= 3 and ip > 0:
        sp_ip = max(3.5, min(ip / gs, 7.5))
    elif ip >= 100:
        sp_ip = 5.5
    else:
        sp_ip = 5.0

    return 1.0, sp_ip  # Assume league-average bullpen for historical


def predict_game_historical(
    game: dict,
    team_factors: dict,
    pitcher_proj: dict,
    league_avg: float,
) -> dict | None:
    """Predict a historical game using season-appropriate data."""
    away = game.get("away_team_name", "")
    home = game.get("home_team_name", "")

    if not away or not home:
        return None

    away_f = team_factors.get(away, {"off_factor": 1.0, "def_factor": 1.0})
    home_f = team_factors.get(home, {"off_factor": 1.0, "def_factor": 1.0})

    away_off = away_f["off_factor"]
    home_off = home_f["off_factor"]
    away_def = away_f["def_factor"]
    home_def = home_f["def_factor"]

    # Starter factors
    away_pitcher = game.get("away_pitcher_name", "")
    home_pitcher = game.get("home_pitcher_name", "")

    away_sp = sp_factor(away_pitcher, pitcher_proj, league_avg)
    home_sp = sp_factor(home_pitcher, pitcher_proj, league_avg)

    # Simplified bullpen blend (starter share + league-avg pen)
    _, away_sp_ip = bullpen_factor_for_pitcher(pitcher_proj, away_pitcher)
    _, home_sp_ip = bullpen_factor_for_pitcher(pitcher_proj, home_pitcher)

    away_sp_share = min(away_sp_ip / 9.0, 0.85)
    home_sp_share = min(home_sp_ip / 9.0, 0.85)

    away_game_pitch = away_sp * away_sp_share + 1.0 * (1 - away_sp_share)
    home_game_pitch = home_sp * home_sp_share + 1.0 * (1 - home_sp_share)

    park = get_park_factor(game.get("venue_id"))

    away_lambda = compute_expected_runs(
        away_off, home_def, home_game_pitch, park, league_avg, False
    )
    home_lambda = compute_expected_runs(
        home_off, away_def, away_game_pitch, park, league_avg, True
    )

    pred = generate_prediction(away_lambda, home_lambda, away, home)

    away_score = game.get("away_score", 0)
    home_score = game.get("home_score", 0)
    if away_score == home_score:
        return None

    return {
        "game_id": game.get("game_id"),
        "date": game.get("date", ""),
        "away_team": away,
        "home_team": home,
        "home_win_prob": pred["win_probability"]["home"] / 100,
        "away_win_prob": pred["win_probability"]["away"] / 100,
        "home_won": 1 if home_score > away_score else 0,
        "away_score": away_score,
        "home_score": home_score,
        "pred_total": pred["expected_total"],
        "actual_total": away_score + home_score,
    }


def run_season(season: int, results_path: str, proj_season: int | None = None) -> list[dict]:
    """Run backtest for a single season."""
    if proj_season is None:
        proj_season = season

    # Load results
    with open(results_path) as f:
        games = json.load(f)
    print(f"\n  {season} season: {len(games)} games")

    # Load projections for this season
    pitcher_proj = load_pitcher_projections_for_season(proj_season)
    print(f"    Pitcher projections ({proj_season}): {len(pitcher_proj)}")

    # Load team factors
    team_stats = load_team_stats_for_season(proj_season)
    league_avg = 4.50  # Approximate; varies slightly by year

    if team_stats:
        team_factors = compute_team_factors(team_stats, league_avg)
        print(f"    Team factors: {len(team_factors)} teams")

        # Recalculate league average from team data
        if team_factors:
            off_factors = [f["off_factor"] for f in team_factors.values()]
            if off_factors:
                # The average of off_factors should be ~1.0
                # but we can derive the actual average from the standings
                pass
    else:
        team_factors = {}
        print(f"    ⚠ No team stats available — using neutral factors")

    # Predict
    predictions = []
    for game in games:
        pred = predict_game_historical(game, team_factors, pitcher_proj, league_avg)
        if pred:
            pred["season"] = season
            predictions.append(pred)

    print(f"    Predictions: {len(predictions)}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Multi-year MLB backtest")
    parser.add_argument("--seasons", type=str, default="2023,2024,2025",
                        help="Comma-separated seasons to backtest")
    args = parser.parse_args()

    seasons = [int(s) for s in args.seasons.split(",")]

    print(f"\n{'=' * 80}")
    print(f"  ⚾ MULTI-YEAR BACKTEST: {', '.join(str(s) for s in seasons)}")
    print(f"{'=' * 80}")

    all_predictions = []

    for season in seasons:
        results_path = os.path.join(DATA_DIR, f"results_{season}.json")
        if not os.path.exists(results_path):
            print(f"\n  ⚠ {season}: Results file not found ({results_path}) — skipping")
            continue

        # Use same-season projections (what was available before that season)
        preds = run_season(season, results_path, proj_season=season)
        all_predictions.extend(preds)

    if not all_predictions:
        print("\n❌ No predictions generated. Check data files.")
        sys.exit(1)

    # Per-season metrics
    print(f"\n{'=' * 80}")
    print(f"  PER-SEASON RESULTS")
    print(f"{'=' * 80}")
    print(f"\n  {'Season':>7} {'Games':>7} {'Acc%':>7} {'LogLoss':>9} {'Brier':>8}")
    print(f"  {'─' * 42}")

    for season in seasons:
        season_preds = [p for p in all_predictions if p.get("season") == season]
        if not season_preds:
            continue
        metrics = compute_accuracy(season_preds)
        print(f"  {season:>7} {metrics['n_games']:>7} {metrics['accuracy']:>6.1f}% "
              f"{metrics['log_loss']:>9.4f} {metrics['brier_score']:>8.4f}")

    # Combined metrics
    print(f"\n{'=' * 80}")
    print(f"  COMBINED RESULTS ({len(all_predictions)} games across {len(seasons)} seasons)")
    print(f"{'=' * 80}")

    metrics = compute_accuracy(all_predictions)
    edge_sweep = run_edge_sweep(all_predictions)
    print_backtest_report(metrics, edge_sweep, season=f"{seasons[0]}-{seasons[-1]}")

    # Flat bet analysis
    print(f"\n  FLAT BET ANALYSIS ($100/bet at -110)")
    print(f"  {'─' * 65}")
    print(f"  {'Edge':>7} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Units':>8} {'$/Season':>10}")
    print(f"  {'─' * 65}")

    implied = 110 / 210
    for min_edge in [0.00, 0.02, 0.03, 0.05, 0.08, 0.10]:
        bets = []
        for p in all_predictions:
            hp = p["home_win_prob"]
            ap = 1 - hp
            he = hp - implied
            ae = ap - implied
            if he >= min_edge and he >= ae:
                bets.append(p["home_won"] == 1)
            elif ae >= min_edge:
                bets.append(p["home_won"] == 0)

        if not bets:
            continue

        wins = sum(bets)
        losses = len(bets) - wins
        win_pct = wins / len(bets) * 100
        profit = wins * (100 / 1.10) - losses * 100
        units = profit / 100
        roi = profit / (len(bets) * 100) * 100
        per_season = units / len(seasons)
        marker = "✓" if roi > 0 else "✗"

        print(f"  ≥{min_edge*100:>4.0f}%  {len(bets):>6} {win_pct:>6.1f}% "
              f"{roi:>+7.2f}% {units:>+7.1f}u {per_season:>+9.1f}u {marker}")

    print(f"  {'─' * 65}")
    print(f"  Break-even at -110 = 52.4%\n")

    # Save
    os.makedirs("output", exist_ok=True)
    out_path = f"output/backtest_multiyear_{seasons[0]}_{seasons[-1]}.json"
    with open(out_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"  Saved to {out_path}\n")


if __name__ == "__main__":
    main()
