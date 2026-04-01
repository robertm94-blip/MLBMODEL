"""Live mid-game Monte Carlo simulator.

Simulates from any game state forward to compute real-time win probability.
Takes current score, inning, outs, baserunners, batting order position,
and remaining pitcher quality, then runs N simulations to completion.
"""

import random
from typing import Any

from src.simulator import (
    compute_pa_probabilities,
    simulate_pa,
    advance_runners,
    get_batter_projections,
    LEAGUE_AVG,
)


def simulate_from_state(
    away_lineup: list[dict],
    home_lineup: list[dict],
    away_pitcher: dict | None,
    home_pitcher: dict | None,
    away_bp_pitcher: dict | None,
    home_bp_pitcher: dict | None,
    # Current game state
    away_score: int,
    home_score: int,
    inning: int,  # 1-9+
    is_top: bool,  # True = top of inning (away batting)
    outs: int,  # 0-2
    bases: list[int],  # [first, second, third] — 0 or 1
    away_batter_idx: int,  # 0-8 position in batting order
    home_batter_idx: int,
    # Pitcher state
    away_sp_pa_remaining: int,  # PA left before bullpen
    home_sp_pa_remaining: int,
    # Adjustment factors
    park_factor: float = 1.0,
    away_platoon_factors: dict[int, float] | None = None,
    home_platoon_factors: dict[int, float] | None = None,
    ump_factor: float = 1.0,
    weather_factor: float = 1.0,
    # Sim params
    n_sims: int = 5000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate from current game state to completion.

    Returns win probabilities and expected final scores.
    """
    rng = random.Random(seed)
    batter_projs = get_batter_projections()

    away_wins = 0
    home_wins = 0
    total_away_final = 0
    total_home_final = 0

    for _ in range(n_sims):
        # Copy game state
        a_score = away_score
        h_score = home_score
        cur_inning = inning
        cur_top = is_top
        cur_outs = outs
        cur_bases = bases.copy()
        a_idx = away_batter_idx
        h_idx = home_batter_idx
        a_sp_pa = away_sp_pa_remaining
        h_sp_pa = home_sp_pa_remaining

        def get_probs(player_id, pitcher, platoon_factors):
            proj = batter_projs.get(player_id)
            if proj is None:
                proj = {
                    "bb_pct": LEAGUE_AVG["bb_pct"], "k_pct": LEAGUE_AVG["k_pct"],
                    "pa": 500, "ab": 440, "h": 110, "hr": 15,
                    "double": 22, "triple": 3, "hbp": 5,
                }
            plat = 1.0
            if platoon_factors and player_id in platoon_factors:
                plat = platoon_factors[player_id]
            return compute_pa_probabilities(
                proj, pitcher, park_factor, plat,
                ump_factor, weather_factor, 1.0,
            )

        # Finish current half-inning if in progress
        def sim_rest_of_half(lineup, batter_idx, cur_outs, cur_bases,
                             starter, bp, sp_pa_left, platoon_factors):
            outs_local = cur_outs
            bases_local = cur_bases.copy()
            runs = 0
            pa_count = 0

            while outs_local < 3:
                batter = lineup[batter_idx % 9]
                pid = batter["id"]

                # Active pitcher
                if sp_pa_left > 0:
                    active_pitcher = starter
                    sp_pa_left -= 1
                else:
                    active_pitcher = bp

                pa_probs = get_probs(pid, active_pitcher, platoon_factors)
                outcome = simulate_pa(pa_probs, rng)
                bases_local, scored, outs_local = advance_runners(
                    bases_local, outcome, outs_local, rng
                )
                runs += scored
                batter_idx += 1
                pa_count += 1

            return runs, batter_idx, sp_pa_left

        # Complete the current half-inning
        if cur_top:
            # Away batting — finish this half
            runs, a_idx, h_sp_pa = sim_rest_of_half(
                away_lineup, a_idx, cur_outs, cur_bases,
                home_pitcher, home_bp_pitcher, h_sp_pa,
                away_platoon_factors,
            )
            a_score += runs

            # Check if home team already won (can't happen in top)
            # Now do bottom of current inning
            if cur_inning < 9 or a_score >= h_score:
                # Bottom half
                runs, h_idx, a_sp_pa = sim_rest_of_half(
                    home_lineup, h_idx, 0, [0, 0, 0],
                    away_pitcher, away_bp_pitcher, a_sp_pa,
                    home_platoon_factors,
                )
                h_score += runs

                # Walk-off check
                if cur_inning >= 9 and h_score > a_score:
                    home_wins += 1
                    total_away_final += a_score
                    total_home_final += h_score
                    continue

            cur_inning += 1
        else:
            # Bottom of inning — away is batting done, home batting
            runs, h_idx, a_sp_pa = sim_rest_of_half(
                home_lineup, h_idx, cur_outs, cur_bases,
                away_pitcher, away_bp_pitcher, a_sp_pa,
                home_platoon_factors,
            )
            h_score += runs

            # Walk-off check
            if cur_inning >= 9 and h_score > a_score:
                home_wins += 1
                total_away_final += a_score
                total_home_final += h_score
                continue

            cur_inning += 1

        # Simulate remaining full innings
        max_innings = 14  # extras cap
        while cur_inning <= max(9, max_innings):
            # Top: away bats
            runs, a_idx, h_sp_pa = sim_rest_of_half(
                away_lineup, a_idx, 0, [0, 0, 0],
                home_pitcher, home_bp_pitcher, h_sp_pa,
                away_platoon_factors,
            )
            a_score += runs

            # Bottom: home bats
            # Skip if home leads after 9+ and it's a regulation finish
            if cur_inning >= 9 and h_score > a_score:
                break

            runs, h_idx, a_sp_pa = sim_rest_of_half(
                home_lineup, h_idx, 0, [0, 0, 0],
                away_pitcher, away_bp_pitcher, a_sp_pa,
                home_platoon_factors,
            )
            h_score += runs

            # Walk-off
            if cur_inning >= 9 and h_score > a_score:
                break

            # End of regulation
            if cur_inning >= 9 and a_score != h_score:
                break

            cur_inning += 1

        # Tally
        if a_score > h_score:
            away_wins += 1
        else:
            home_wins += 1

        total_away_final += a_score
        total_home_final += h_score

    return {
        "away_win_pct": round(away_wins / n_sims * 100, 1),
        "home_win_pct": round(home_wins / n_sims * 100, 1),
        "avg_away_final": round(total_away_final / n_sims, 1),
        "avg_home_final": round(total_home_final / n_sims, 1),
        "n_sims": n_sims,
    }
