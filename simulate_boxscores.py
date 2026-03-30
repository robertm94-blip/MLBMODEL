#!/usr/bin/env python3
"""MLB Full Box Score Simulator — Monte Carlo plate-by-plate simulation.

Simulates each game 1,000 times using batter-specific projected rates
(Steamer + ZiPS + THE BAT blend) adjusted for the opposing pitcher.
Produces expected stat lines for every player in every game.
"""

import json
import argparse
import os
from datetime import date

from src.mlb_api import get_schedule
from src.projections import (
    load_all_pitcher_projections,
    get_team_projected_strength,
    get_league_avg_rpg,
)
from src.features import get_park_factor
from src.simulator import run_simulations, get_batter_projections


def load_lineups(date_str: str) -> dict:
    """Load lineup data from JSON file."""
    filepath = os.path.join(
        os.path.dirname(__file__), f"lineups_{date_str.replace('-', '_')}.json"
    )
    if not os.path.exists(filepath):
        print(f"\n❌ Lineup file not found: {filepath}")
        print(f"   Run the lineup fetcher first.")
        return {}
    with open(filepath) as f:
        return json.load(f)


def format_boxscore(
    game_info: dict,
    sim_result: dict,
    away_lineup: list[dict],
    home_lineup: list[dict],
) -> str:
    """Format a single game's simulation results as a box score."""
    lines = []
    away_name = game_info["away_team_name"]
    home_name = game_info["home_team_name"]
    away_short = away_name.split()[-1]
    home_short = home_name.split()[-1]

    lines.append(f"\n{'━' * 96}")
    lines.append(f"  {away_name} @ {home_name}")
    lines.append(f"  📍 {game_info.get('venue_name', 'Unknown')} | "
                 f"🎯 {game_info['away_pitcher_name']} vs {game_info['home_pitcher_name']}")
    lines.append(f"  Simulations: {sim_result['n_sims']:,} | "
                 f"Win%: {away_short} {sim_result['away_win_pct']}% / "
                 f"{home_short} {sim_result['home_win_pct']}%")
    lines.append(f"{'━' * 96}")

    # Linescore
    ls_a = sim_result["avg_linescore_away"]
    ls_h = sim_result["avg_linescore_home"]
    header = "         " + "".join(f" {i+1:>4}" for i in range(9)) + "  │   R     H"
    lines.append(f"\n  EXPECTED LINESCORE:")
    lines.append(f"  {header}")
    lines.append(f"  {'─' * 70}")

    # Compute total expected hits per team
    avgs = sim_result["player_avgs"]
    away_hits = sum(avgs[p["id"]]["h"] for p in away_lineup if p["id"] in avgs)
    home_hits = sum(avgs[p["id"]]["h"] for p in home_lineup if p["id"] in avgs)

    away_ls = f"  {away_short:>8}" + "".join(f" {x:>4.1f}" for x in ls_a[:9])
    away_ls += f"  │ {sim_result['avg_away_runs']:>5.2f} {away_hits:>5.1f}"
    lines.append(away_ls)

    home_ls = f"  {home_short:>8}" + "".join(f" {x:>4.1f}" for x in ls_h[:9])
    home_ls += f"  │ {sim_result['avg_home_runs']:>5.2f} {home_hits:>5.1f}"
    lines.append(home_ls)

    # Most likely scores
    lines.append(f"\n  MOST LIKELY FINAL SCORES:")
    for a, h, pct in sim_result["top_scores"]:
        marker = ""
        if a > h:
            marker = f" ({away_short} wins)"
        elif h > a:
            marker = f" ({home_short} wins)"
        else:
            marker = " (extras)"
        lines.append(f"    {away_short} {a} - {home_short} {h}  ({pct:.1f}%){marker}")

    # Away box score
    lines.append(f"\n  {'─' * 96}")
    lines.append(f"  {away_name} — PROJECTED BOX SCORE (per game averages)")
    lines.append(f"  {'─' * 96}")
    lines.append(f"  {'Player':<24} {'Pos':>4}  {'PA':>5} {'AB':>5} {'H':>5} "
                 f"{'R':>5} {'RBI':>5} {'HR':>4} {'2B':>4} {'3B':>4} "
                 f"{'BB':>5} {'K':>5} {'AVG':>6}")
    lines.append(f"  {'─' * 96}")

    away_totals = {"pa": 0, "ab": 0, "h": 0, "r": 0, "rbi": 0, "hr": 0,
                   "double": 0, "triple": 0, "bb": 0, "k": 0}

    for p in away_lineup:
        pid = p["id"]
        if pid not in avgs:
            continue
        s = avgs[pid]
        avg_str = f"{s['avg']:.3f}" if s['ab'] > 0 else "  ---"
        lines.append(
            f"  {s['name']:<24} {s['pos']:>4}  {s['pa']:>5.1f} {s['ab']:>5.1f} {s['h']:>5.2f} "
            f"{s['r']:>5.2f} {s['rbi']:>5.2f} {s['hr']:>4.2f} {s['double']:>4.2f} {s['triple']:>4.2f} "
            f"{s['bb']:>5.2f} {s['k']:>5.2f} {avg_str:>6}"
        )
        for k in away_totals:
            away_totals[k] += s[k]

    total_avg = f"{away_totals['h'] / away_totals['ab']:.3f}" if away_totals['ab'] > 0 else "---"
    lines.append(f"  {'─' * 96}")
    lines.append(
        f"  {'TOTALS':<24} {'':>4}  {away_totals['pa']:>5.1f} {away_totals['ab']:>5.1f} "
        f"{away_totals['h']:>5.2f} {away_totals['r']:>5.2f} {away_totals['rbi']:>5.2f} "
        f"{away_totals['hr']:>4.2f} {away_totals['double']:>4.2f} {away_totals['triple']:>4.2f} "
        f"{away_totals['bb']:>5.2f} {away_totals['k']:>5.2f} {total_avg:>6}"
    )

    # Home box score
    lines.append(f"\n  {'─' * 96}")
    lines.append(f"  {home_name} — PROJECTED BOX SCORE (per game averages)")
    lines.append(f"  {'─' * 96}")
    lines.append(f"  {'Player':<24} {'Pos':>4}  {'PA':>5} {'AB':>5} {'H':>5} "
                 f"{'R':>5} {'RBI':>5} {'HR':>4} {'2B':>4} {'3B':>4} "
                 f"{'BB':>5} {'K':>5} {'AVG':>6}")
    lines.append(f"  {'─' * 96}")

    home_totals = {"pa": 0, "ab": 0, "h": 0, "r": 0, "rbi": 0, "hr": 0,
                   "double": 0, "triple": 0, "bb": 0, "k": 0}

    for p in home_lineup:
        pid = p["id"]
        if pid not in avgs:
            continue
        s = avgs[pid]
        avg_str = f"{s['avg']:.3f}" if s['ab'] > 0 else "  ---"
        lines.append(
            f"  {s['name']:<24} {s['pos']:>4}  {s['pa']:>5.1f} {s['ab']:>5.1f} {s['h']:>5.2f} "
            f"{s['r']:>5.2f} {s['rbi']:>5.2f} {s['hr']:>4.2f} {s['double']:>4.2f} {s['triple']:>4.2f} "
            f"{s['bb']:>5.2f} {s['k']:>5.2f} {avg_str:>6}"
        )
        for k in home_totals:
            home_totals[k] += s[k]

    total_avg = f"{home_totals['h'] / home_totals['ab']:.3f}" if home_totals['ab'] > 0 else "---"
    lines.append(f"  {'─' * 96}")
    lines.append(
        f"  {'TOTALS':<24} {'':>4}  {home_totals['pa']:>5.1f} {home_totals['ab']:>5.1f} "
        f"{home_totals['h']:>5.2f} {home_totals['r']:>5.2f} {home_totals['rbi']:>5.2f} "
        f"{home_totals['hr']:>4.2f} {home_totals['double']:>4.2f} {home_totals['triple']:>4.2f} "
        f"{home_totals['bb']:>5.2f} {home_totals['k']:>5.2f} {total_avg:>6}"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MLB Full Box Score Simulator")
    parser.add_argument("--date", type=str, default=str(date.today()),
                        help="Game date (YYYY-MM-DD)")
    parser.add_argument("--sims", type=int, default=1000,
                        help="Number of simulations per game (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--game", type=int, default=None,
                        help="Simulate only this game ID")
    args = parser.parse_args()

    game_date = args.date

    print(f"\n⚾ MLB Full Box Score Simulator")
    print(f"   Game Date: {game_date}")
    print(f"   Simulations per game: {args.sims:,}")
    print(f"   Projections: Steamer + ZiPS + THE BAT (blended)")

    # Load schedule
    print(f"\n📅 Fetching schedule...")
    games = get_schedule(game_date)
    if not games:
        print(f"❌ No games found for {game_date}")
        return

    if args.game:
        games = [g for g in games if g["game_id"] == args.game]
        if not games:
            print(f"❌ Game {args.game} not found")
            return

    print(f"   {len(games)} games to simulate")

    # Load lineups
    print(f"\n📋 Loading lineups...")
    lineup_data = load_lineups(game_date)
    if not lineup_data:
        return
    print(f"   Loaded lineups for {len(lineup_data.get('games', {}))} games")

    # Load projections
    print(f"\n📊 Loading projections...")
    pitcher_projs = load_all_pitcher_projections()
    batter_projs = get_batter_projections()
    print(f"   {len(pitcher_projs)} pitcher projections")
    print(f"   {len(batter_projs)} batter projections")

    # Simulate each game
    for i, game in enumerate(games, 1):
        gid = str(game["game_id"])
        lineup_info = lineup_data.get("games", {}).get(gid)

        if not lineup_info:
            print(f"\n⚠ No lineup data for game {gid} — skipping")
            continue

        # Build lineup structures
        away_lineup = [
            {"id": p["id"], "name": p["name"], "pos": p["position"]}
            for p in lineup_info["away_lineup"]
        ]
        home_lineup = [
            {"id": p["id"], "name": p["name"], "pos": p["position"]}
            for p in lineup_info["home_lineup"]
        ]

        # Get pitcher projections
        away_pitcher = pitcher_projs.get(game["away_pitcher_name"])
        home_pitcher = pitcher_projs.get(game["home_pitcher_name"])

        park_factor = get_park_factor(game.get("venue_id"))

        print(f"\n🔄 Simulating Game {i}/{len(games)}: "
              f"{game['away_team_name']} @ {game['home_team_name']} "
              f"({args.sims:,} sims)...", end=" ", flush=True)

        result = run_simulations(
            away_lineup, home_lineup,
            away_pitcher, home_pitcher,
            park_factor,
            n_sims=args.sims,
            seed=args.seed + i,
        )

        print(f"Done — {result['avg_away_runs']:.1f} to {result['avg_home_runs']:.1f}")

        # Format and print box score
        boxscore = format_boxscore(game, result, away_lineup, home_lineup)
        print(boxscore)

    # Footer
    print(f"\n{'━' * 96}")
    print(f"  SIMULATION METHODOLOGY:")
    print(f"  • {args.sims:,} Monte Carlo simulations per game")
    print(f"  • PA outcomes drawn from batter-specific projected rates")
    print(f"  • Rates adjusted for opposing pitcher quality (K%, BB%, ERA/FIP)")
    print(f"  • Park factors applied to HR and extra-base hit rates")
    print(f"  • Full baserunning simulation with probabilistic advancement")
    print(f"  • Projections blended equally from Steamer, ZiPS, and THE BAT")
    print(f"{'━' * 96}\n")


if __name__ == "__main__":
    main()
