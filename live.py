#!/usr/bin/env python3
"""MLB Live Win Probability — Real-time pitch-by-pitch updates.

Polls the MLB Stats API every second for live game state, then runs
Monte Carlo simulations from the current state to compute win probability.

Usage:
    python live.py                    # All live games
    python live.py --game 824135      # Single game
    python live.py --sims 2000        # More sims (slower but sharper)

The display updates in-place in your terminal showing:
- Current score, inning, outs, runners
- Win probability for each team
- WP change since last update
- Current batter vs pitcher matchup
"""

import sys
import os
import time
import json
import argparse
import requests
from datetime import date

from src.projections import load_all_pitcher_projections
from src.features import get_park_factor
from src.bullpen import load_team_bullpen_projections, get_starter_projected_length
from src.platoon import load_pitcher_hands, load_batter_hands, get_platoon_factor
from src.live_sim import simulate_from_state, get_batter_projections


def fetch_live_data(game_id: int) -> dict | None:
    """Fetch current game state from MLB Stats API."""
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def parse_game_state(data: dict) -> dict | None:
    """Parse MLB API live feed into our game state format."""
    game_data = data.get("gameData", {})
    live_data = data.get("liveData", {})
    linescore = live_data.get("linescore", {})
    plays = live_data.get("plays", {})

    status = game_data.get("status", {}).get("abstractGameState", "")
    if status not in ("Live", "InProgress", "In Progress"):
        return None

    # Score
    away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0) or 0
    home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0) or 0

    # Inning
    inning = linescore.get("currentInning", 1)
    is_top = linescore.get("isTopInning", True)

    # Outs
    outs = linescore.get("outs", 0) or 0

    # Runners
    offense = linescore.get("offense", {})
    bases = [0, 0, 0]
    if offense.get("first"):
        bases[0] = 1
    if offense.get("second"):
        bases[1] = 1
    if offense.get("third"):
        bases[2] = 1

    # Current matchup
    current_play = plays.get("currentPlay", {})
    matchup = current_play.get("matchup", {})
    batter_name = matchup.get("batter", {}).get("fullName", "Unknown")
    batter_id = matchup.get("batter", {}).get("id", 0)
    pitcher_name = matchup.get("pitcher", {}).get("fullName", "Unknown")
    pitcher_id = matchup.get("pitcher", {}).get("id", 0)

    # Count
    count = current_play.get("count", {})
    balls = count.get("balls", 0)
    strikes = count.get("strikes", 0)

    # Teams
    teams = game_data.get("teams", {})
    away_team = teams.get("away", {}).get("name", "Away")
    home_team = teams.get("home", {}).get("name", "Home")
    away_team_id = teams.get("away", {}).get("id", 0)
    home_team_id = teams.get("home", {}).get("id", 0)

    # Venue
    venue_id = game_data.get("venue", {}).get("id")

    # Pitchers (current)
    away_pitcher_name = ""
    home_pitcher_name = ""
    boxscore = live_data.get("boxscore", {})
    away_pitchers = boxscore.get("teams", {}).get("away", {}).get("pitchers", [])
    home_pitchers = boxscore.get("teams", {}).get("home", {}).get("pitchers", [])

    # Get starting pitchers from boxscore
    players = boxscore.get("teams", {}).get("away", {}).get("players", {})
    for pid in away_pitchers[:1]:  # First pitcher = starter
        pdata = players.get(f"ID{pid}", {})
        away_pitcher_name = pdata.get("person", {}).get("fullName", "")

    players = boxscore.get("teams", {}).get("home", {}).get("players", {})
    for pid in home_pitchers[:1]:
        pdata = players.get(f"ID{pid}", {})
        home_pitcher_name = pdata.get("person", {}).get("fullName", "")

    # Batting order from boxscore
    away_batters = boxscore.get("teams", {}).get("away", {}).get("battingOrder", [])
    home_batters = boxscore.get("teams", {}).get("home", {}).get("battingOrder", [])

    away_lineup = []
    away_players = boxscore.get("teams", {}).get("away", {}).get("players", {})
    for pid in away_batters:
        pdata = away_players.get(f"ID{pid}", {})
        away_lineup.append({
            "id": pid,
            "name": pdata.get("person", {}).get("fullName", "Unknown"),
            "pos": pdata.get("position", {}).get("abbreviation", ""),
        })

    home_lineup = []
    home_players = boxscore.get("teams", {}).get("home", {}).get("players", {})
    for pid in home_batters:
        pdata = home_players.get(f"ID{pid}", {})
        home_lineup.append({
            "id": pid,
            "name": pdata.get("person", {}).get("fullName", "Unknown"),
            "pos": pdata.get("position", {}).get("abbreviation", ""),
        })

    # Figure out batting order index
    # Which batter in the order is currently up?
    if is_top:
        # Away is batting
        batting_lineup = away_lineup
    else:
        batting_lineup = home_lineup

    batter_order_idx = 0
    for idx, p in enumerate(batting_lineup):
        if p["id"] == batter_id:
            batter_order_idx = idx
            break

    # Last play description
    last_play = ""
    all_plays = plays.get("allPlays", [])
    if all_plays:
        last = all_plays[-1]
        result = last.get("result", {})
        last_play = result.get("description", "")

    return {
        "away_team": away_team,
        "home_team": home_team,
        "away_team_id": away_team_id,
        "home_team_id": home_team_id,
        "away_score": away_score,
        "home_score": home_score,
        "inning": inning,
        "is_top": is_top,
        "outs": outs,
        "bases": bases,
        "batter_name": batter_name,
        "batter_id": batter_id,
        "pitcher_name": pitcher_name,
        "pitcher_id": pitcher_id,
        "balls": balls,
        "strikes": strikes,
        "away_lineup": away_lineup,
        "home_lineup": home_lineup,
        "away_pitcher_name": away_pitcher_name,
        "home_pitcher_name": home_pitcher_name,
        "venue_id": venue_id,
        "last_play": last_play,
        "away_batter_idx": batter_order_idx if is_top else 0,
        "home_batter_idx": batter_order_idx if not is_top else 0,
    }


def get_live_games(date_str: str = None) -> list[dict]:
    """Get all live game IDs for today."""
    if date_str is None:
        date_str = str(date.today())
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?date={date_str}&sportId=1"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        games = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                status = g.get("status", {}).get("abstractGameState", "")
                games.append({
                    "game_id": g["gamePk"],
                    "away": g["teams"]["away"]["team"]["name"],
                    "home": g["teams"]["home"]["team"]["name"],
                    "status": status,
                })
        return games
    except Exception:
        return []


def format_bases(bases: list[int]) -> str:
    """Visual base diagram."""
    b1 = "\033[93m●\033[0m" if bases[0] else "○"
    b2 = "\033[93m●\033[0m" if bases[1] else "○"
    b3 = "\033[93m●\033[0m" if bases[2] else "○"
    return f"{b2}  {b1}  {b3}"


def format_outs(outs: int) -> str:
    """Visual outs indicator."""
    return "●" * outs + "○" * (3 - outs)


def clear_screen():
    """Clear terminal."""
    os.system("cls" if os.name == "nt" else "clear")


def run_live_wp(
    game_id: int,
    n_sims: int = 3000,
    refresh_rate: float = 1.0,
):
    """Run live win probability tracker for a single game."""
    # Pre-load projection data
    pitcher_proj = load_all_pitcher_projections()
    batter_projs = get_batter_projections()
    pitcher_hands = load_pitcher_hands()
    batter_hands = load_batter_hands()
    bp_proj = load_team_bullpen_projections()

    prev_wp = None
    prev_state_key = None

    print(f"\n  Loading game {game_id}...")

    while True:
        try:
            data = fetch_live_data(game_id)
            if data is None:
                print(f"\r  Waiting for game data...", end="", flush=True)
                time.sleep(refresh_rate)
                continue

            state = parse_game_state(data)
            if state is None:
                # Game not live — check status
                game_status = data.get("gameData", {}).get("status", {}).get("detailedState", "")
                if game_status in ("Final", "Game Over", "Completed Early"):
                    # Show final
                    ls = data.get("liveData", {}).get("linescore", {})
                    a_s = ls.get("teams", {}).get("away", {}).get("runs", 0)
                    h_s = ls.get("teams", {}).get("home", {}).get("runs", 0)
                    teams = data.get("gameData", {}).get("teams", {})
                    away = teams.get("away", {}).get("name", "Away")
                    home = teams.get("home", {}).get("name", "Home")
                    clear_screen()
                    print(f"\n  {'='*60}")
                    print(f"  FINAL")
                    print(f"  {away:<25} {a_s}")
                    print(f"  {home:<25} {h_s}")
                    print(f"  {'='*60}\n")
                    break
                else:
                    print(f"\r  Game status: {game_status} — waiting...   ", end="", flush=True)
                    time.sleep(5)
                    continue

            # Create state key to detect changes
            state_key = (
                state["away_score"], state["home_score"],
                state["inning"], state["is_top"],
                state["outs"], tuple(state["bases"]),
                state["batter_id"],
            )

            # Only re-simulate if state changed
            if state_key != prev_state_key:
                # Get pitcher projections
                away_sp = pitcher_proj.get(state["away_pitcher_name"])
                home_sp = pitcher_proj.get(state["home_pitcher_name"])

                # Current pitcher on mound
                cur_pitcher = pitcher_proj.get(state["pitcher_name"])

                # Bullpen pitchers
                def make_bp(team_name):
                    bp = bp_proj.get(team_name)
                    if not bp:
                        return None
                    return {
                        "era": bp["era"], "fip": bp["fip"], "whip": bp["whip"],
                        "k9": bp["k9"], "bb9": bp["bb9"],
                        "k_pct": bp.get("k9", 8.5) / 38.0,
                        "bb_pct": bp.get("bb9", 3.2) / 38.0,
                    }

                away_bp = make_bp(state["away_team"])
                home_bp = make_bp(state["home_team"])

                # Estimate remaining SP PA
                # Rough: starters typically face 24-28 batters
                # In a live game, assume pen is in after inning 6ish
                est_innings_pitched = state["inning"] - 1
                if state["is_top"]:
                    # Home pitcher has been pitching (top = away batting)
                    home_sp_pa_left = max(0, 26 - est_innings_pitched * 4)
                    away_sp_pa_left = max(0, 26 - (est_innings_pitched - 1) * 4) if est_innings_pitched > 0 else 26
                else:
                    away_sp_pa_left = max(0, 26 - est_innings_pitched * 4)
                    home_sp_pa_left = max(0, 26 - est_innings_pitched * 4)

                # If we're past inning 6, assume bullpen is in
                if state["inning"] >= 7:
                    away_sp_pa_left = 0
                    home_sp_pa_left = 0

                # Platoon factors
                a_plat = {}
                h_plat = {}
                for p in state["away_lineup"]:
                    hand = batter_hands.get(p["id"], "R")
                    opp_hand = pitcher_hands.get(state.get("pitcher_id"), "R") if not state["is_top"] else "R"
                    a_plat[p["id"]] = get_platoon_factor(hand, opp_hand)
                for p in state["home_lineup"]:
                    hand = batter_hands.get(p["id"], "R")
                    opp_hand = pitcher_hands.get(state.get("pitcher_id"), "R") if state["is_top"] else "R"
                    h_plat[p["id"]] = get_platoon_factor(hand, opp_hand)

                park = get_park_factor(state["venue_id"])

                # Use actual lineups if available, fall back
                away_lu = state["away_lineup"] if state["away_lineup"] else [
                    {"id": 0, "name": f"Batter {i+1}", "pos": ""} for i in range(9)
                ]
                home_lu = state["home_lineup"] if state["home_lineup"] else [
                    {"id": 0, "name": f"Batter {i+1}", "pos": ""} for i in range(9)
                ]

                # Run simulation from current state
                result = simulate_from_state(
                    away_lu, home_lu,
                    away_sp if away_sp_pa_left > 0 else away_bp,
                    home_sp if home_sp_pa_left > 0 else home_bp,
                    away_bp, home_bp,
                    state["away_score"], state["home_score"],
                    state["inning"], state["is_top"],
                    state["outs"], state["bases"],
                    state["away_batter_idx"], state["home_batter_idx"],
                    away_sp_pa_left, home_sp_pa_left,
                    park_factor=park,
                    away_platoon_factors=a_plat,
                    home_platoon_factors=h_plat,
                    n_sims=n_sims,
                    seed=None,  # Different seed each time for stability
                )

                wp_change = ""
                if prev_wp is not None:
                    delta = result["home_win_pct"] - prev_wp
                    if abs(delta) >= 0.5:
                        arrow = "\033[92m▲\033[0m" if delta > 0 else "\033[91m▼\033[0m"
                        wp_change = f"  {arrow} {abs(delta):+.1f}%"

                prev_wp = result["home_win_pct"]
                prev_state_key = state_key

                # Display
                clear_screen()
                away_short = state["away_team"].split()[-1]
                home_short = state["home_team"].split()[-1]
                half = "▲" if state["is_top"] else "▼"

                print(f"\n  {'='*62}")
                print(f"  ⚾ LIVE WIN PROBABILITY — Game {game_id}")
                print(f"  {'='*62}")
                print()
                print(f"  {half} {state['inning']}{'st' if state['inning']==1 else 'nd' if state['inning']==2 else 'rd' if state['inning']==3 else 'th'}   "
                      f"Outs: {format_outs(state['outs'])}   "
                      f"Count: {state['balls']}-{state['strikes']}")
                print(f"  Bases: {format_bases(state['bases'])}")
                print()

                # Score + WP bars
                a_wp = result["away_win_pct"]
                h_wp = result["home_win_pct"]
                a_bar_len = int(a_wp / 2)
                h_bar_len = int(h_wp / 2)

                a_bar = "\033[94m" + "█" * a_bar_len + "\033[0m"
                h_bar = "\033[91m" + "█" * h_bar_len + "\033[0m"

                print(f"  {away_short:<15} {state['away_score']:>2}   {a_wp:>5.1f}%  {a_bar}")
                print(f"  {home_short:<15} {state['home_score']:>2}   {h_wp:>5.1f}%  {h_bar}{wp_change}")
                print()

                # Current matchup
                batting_side = "away" if state["is_top"] else "home"
                print(f"  AB: {state['batter_name']}")
                print(f"  P:  {state['pitcher_name']}")

                if state["last_play"]:
                    # Truncate long descriptions
                    desc = state["last_play"][:70]
                    print(f"\n  Last: {desc}")

                print(f"\n  E[Final]: {away_short} {result['avg_away_final']:.1f} - {home_short} {result['avg_home_final']:.1f}")
                print(f"  Sims: {n_sims:,}  |  Updated: {time.strftime('%H:%M:%S')}")
                print(f"  {'='*62}")
                print(f"  Press Ctrl+C to exit")

            time.sleep(refresh_rate)

        except KeyboardInterrupt:
            print("\n\n  Stopped.\n")
            break
        except Exception as e:
            print(f"\r  Error: {e}  ", end="", flush=True)
            time.sleep(2)


def run_all_live(n_sims: int = 2000, refresh_rate: float = 2.0):
    """Track all live games simultaneously."""
    pitcher_proj = load_all_pitcher_projections()
    bp_proj = load_team_bullpen_projections()
    batter_hands_data = load_batter_hands()

    print(f"\n  Scanning for live games...")

    while True:
        try:
            games = get_live_games()
            live_games = [g for g in games if g["status"] == "Live"]

            if not live_games:
                scheduled = [g for g in games if g["status"] in ("Preview", "Pre-Game", "Scheduled", "Warmup")]
                clear_screen()
                print(f"\n  {'='*60}")
                print(f"  ⚾ MLB LIVE TRACKER — {date.today()}")
                print(f"  {'='*60}")
                print(f"\n  No games currently live.")
                if scheduled:
                    print(f"  {len(scheduled)} games scheduled:")
                    for g in scheduled[:5]:
                        print(f"    {g['away'].split()[-1]} @ {g['home'].split()[-1]} — {g['status']}")
                print(f"\n  Checking every 10 seconds...")
                print(f"  Press Ctrl+C to exit")
                time.sleep(10)
                continue

            clear_screen()
            print(f"\n  {'='*68}")
            print(f"  ⚾ MLB LIVE WIN PROBABILITY — {len(live_games)} games in progress")
            print(f"  {'='*68}\n")

            for g in live_games:
                data = fetch_live_data(g["game_id"])
                if data is None:
                    continue

                state = parse_game_state(data)
                if state is None:
                    continue

                away_short = state["away_team"].split()[-1]
                home_short = state["home_team"].split()[-1]
                half = "▲" if state["is_top"] else "▼"
                outs_str = format_outs(state["outs"])
                bases_short = ""
                if state["bases"][2]: bases_short += "3"
                if state["bases"][1]: bases_short += "2"
                if state["bases"][0]: bases_short += "1"
                if not bases_short: bases_short = "-"

                # Quick sim (fewer sims for multi-game mode)
                away_lu = state["away_lineup"] if state["away_lineup"] else [{"id": 0, "name": f"B{i}", "pos": ""} for i in range(9)]
                home_lu = state["home_lineup"] if state["home_lineup"] else [{"id": 0, "name": f"B{i}", "pos": ""} for i in range(9)]

                away_sp = pitcher_proj.get(state["away_pitcher_name"])
                home_sp = pitcher_proj.get(state["home_pitcher_name"])

                def make_bp(team_name):
                    bp = bp_proj.get(team_name)
                    if not bp: return None
                    return {"era": bp["era"], "fip": bp["fip"], "whip": bp["whip"],
                            "k9": bp["k9"], "bb9": bp["bb9"],
                            "k_pct": bp.get("k9", 8.5) / 38.0, "bb_pct": bp.get("bb9", 3.2) / 38.0}

                sp_pa = 0 if state["inning"] >= 7 else max(0, 26 - state["inning"] * 4)

                result = simulate_from_state(
                    away_lu, home_lu,
                    away_sp, home_sp, make_bp(state["away_team"]), make_bp(state["home_team"]),
                    state["away_score"], state["home_score"],
                    state["inning"], state["is_top"], state["outs"], state["bases"],
                    0, 0, sp_pa, sp_pa,
                    park_factor=get_park_factor(state["venue_id"]),
                    n_sims=n_sims,
                )

                a_wp = result["away_win_pct"]
                h_wp = result["home_win_pct"]
                fav = away_short if a_wp > h_wp else home_short
                fav_pct = max(a_wp, h_wp)

                a_bar = "█" * int(a_wp / 5)
                h_bar = "█" * int(h_wp / 5)

                print(f"  {half}{state['inning']:>2}  {away_short:<10} {state['away_score']:>2} - {state['home_score']:<2} {home_short:<10}  "
                      f"O:{state['outs']} R:{bases_short:<4} "
                      f"{away_short} {a_wp:>5.1f}% | {home_short} {h_wp:>5.1f}%")

            print(f"\n  Updated: {time.strftime('%H:%M:%S')}  |  Sims: {n_sims:,}/game")
            print(f"  {'='*68}")
            print(f"  Tip: python live.py --game {live_games[0]['game_id']} for detailed single-game view")
            print(f"  Press Ctrl+C to exit")

            time.sleep(refresh_rate)

        except KeyboardInterrupt:
            print("\n\n  Stopped.\n")
            break
        except Exception as e:
            print(f"\r  Error: {e}  ", end="", flush=True)
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="MLB Live Win Probability")
    parser.add_argument("--game", type=int, default=None,
                        help="Track a specific game ID")
    parser.add_argument("--sims", type=int, default=3000,
                        help="Simulations per update (default: 3000)")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Refresh rate in seconds (default: 1.0)")
    args = parser.parse_args()

    if args.game:
        run_live_wp(args.game, n_sims=args.sims, refresh_rate=args.rate)
    else:
        run_all_live(n_sims=min(args.sims, 2000), refresh_rate=max(args.rate, 2.0))


if __name__ == "__main__":
    main()
