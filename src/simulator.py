"""Monte Carlo MLB game simulator — Full 10-Factor Engine.

Simulates games plate-appearance by plate-appearance using:
1. Individual batter projected rates (Steamer/ZiPS/THE BAT)
2. Opposing pitcher quality (ERA/FIP/K%/BB% adjustment)
3. L/R platoon splits (batter hand vs pitcher hand)
4. Umpire strike zone (K% and BB% adjustment)
5. Park factors (HR, XBH scaling)
6. Weather effects (temp, wind, humidity on hit rates)
7. Rest/travel fatigue (offense scaling)
8. Bullpen transition mid-game (starter exits based on IP/GS)

Tracks full baserunning state to produce realistic box scores
and Monte Carlo win probabilities.
"""

import json
import os
import random
from collections import defaultdict
from typing import Any

import numpy as np

from src.projections import load_all_pitcher_projections, get_league_avg_rpg
from src.features import get_park_factor

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROJ_DIR = os.path.join(DATA_DIR, "projections", "2026")

LEAGUE_AVG = {
    "bb_pct": 0.083,
    "k_pct": 0.225,
    "hbp_pct": 0.012,
    "babip": 0.295,
}


def _load_batter_projections() -> dict[int, dict[str, float]]:
    """Load and blend batter projections from all systems, keyed by MLBAM ID."""
    systems = ["steamer", "zips", "thebat"]
    all_data: dict[int, list[dict]] = defaultdict(list)

    for system in systems:
        filepath = os.path.join(PROJ_DIR, f"{system}_batters.json")
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            batters = json.load(f)
        for b in batters:
            mlbam_id = b.get("xMLBAMID")
            if not mlbam_id:
                continue
            pa = b.get("PA", 0) or 0
            if pa < 10:
                continue
            all_data[mlbam_id].append(b)

    blended = {}
    for mlbam_id, entries in all_data.items():
        n = len(entries)

        def avg(key):
            return sum(e.get(key, 0) or 0 for e in entries) / n

        blended[mlbam_id] = {
            "name": entries[0].get("PlayerName", "Unknown"),
            "team": entries[0].get("Team", ""),
            "pa": avg("PA"), "ab": avg("AB"), "h": avg("H"),
            "hr": avg("HR"), "bb": avg("BB"), "so": avg("SO"),
            "hbp": avg("HBP"), "double": avg("2B"), "triple": avg("3B"),
            "r": avg("R"), "rbi": avg("RBI"), "sb": avg("SB"),
            "avg": avg("AVG"), "obp": avg("OBP"), "slg": avg("SLG"),
            "bb_pct": avg("BB%"), "k_pct": avg("K%"),
        }

    return blended


_batter_proj_cache = None


def get_batter_projections() -> dict[int, dict[str, float]]:
    """Get cached batter projections."""
    global _batter_proj_cache
    if _batter_proj_cache is None:
        _batter_proj_cache = _load_batter_projections()
    return _batter_proj_cache


def compute_pa_probabilities(
    batter: dict[str, float],
    pitcher: dict[str, float] | None,
    park_factor: float = 1.0,
    platoon_factor: float = 1.0,
    ump_factor: float = 1.0,
    weather_factor: float = 1.0,
    rest_factor: float = 1.0,
) -> dict[str, float]:
    """Compute PA outcome probabilities with all adjustment factors.

    Args:
        batter: Batter projection dict
        pitcher: Pitcher projection dict (starter or reliever)
        park_factor: Venue park factor
        platoon_factor: L/R platoon adjustment (>1 = platoon advantage)
        ump_factor: Umpire zone adjustment (>1 = hitter-friendly)
        weather_factor: Weather run adjustment
        rest_factor: Rest/travel fatigue adjustment
    """
    # Batter base rates
    bb_rate = batter.get("bb_pct", LEAGUE_AVG["bb_pct"])
    k_rate = batter.get("k_pct", LEAGUE_AVG["k_pct"])

    pa = batter.get("pa", 600)
    hbp_rate = batter.get("hbp", 0) / pa if pa > 0 else LEAGUE_AVG["hbp_pct"]

    ab = batter.get("ab", 550)
    h = batter.get("h", 140)
    hr = batter.get("hr", 20)
    doubles = batter.get("double", 28)
    triples = batter.get("triple", 3)
    singles = h - hr - doubles - triples

    if pa > 0:
        hr_rate = hr / pa
        double_rate = doubles / pa
        triple_rate = triples / pa
        single_rate = singles / pa
    else:
        hr_rate, double_rate, triple_rate, single_rate = 0.030, 0.045, 0.004, 0.140

    # ── PITCHER ADJUSTMENT ──
    if pitcher and pitcher.get("era", 0) > 0:
        league_avg_rpg = get_league_avg_rpg()

        p_k_pct = pitcher.get("k_pct", LEAGUE_AVG["k_pct"])
        k_mult = p_k_pct / LEAGUE_AVG["k_pct"] if LEAGUE_AVG["k_pct"] > 0 else 1.0
        k_rate = k_rate * (0.6 + 0.4 * k_mult)

        p_bb_pct = pitcher.get("bb_pct", LEAGUE_AVG["bb_pct"])
        bb_mult = p_bb_pct / LEAGUE_AVG["bb_pct"] if LEAGUE_AVG["bb_pct"] > 0 else 1.0
        bb_rate = bb_rate * (0.6 + 0.4 * bb_mult)

        era = pitcher.get("era", 4.50)
        fip = pitcher.get("fip", era)
        blended = era * 0.4 + fip * 0.6
        raw_hit_mult = blended / league_avg_rpg if league_avg_rpg > 0 else 1.0

        # Dampen the hit multiplier — raw ERA ratio overstates the effect
        # A 3.09 ERA pitcher doesn't cut hits by 31% vs league avg
        # Real effect is ~40% of the raw ratio (10-15% suppression for an ace)
        hit_mult = 1.0 + (raw_hit_mult - 1.0) * 0.40

        hr_rate *= hit_mult
        double_rate *= hit_mult
        triple_rate *= hit_mult
        single_rate *= hit_mult

    # ── PLATOON ADJUSTMENT ──
    # Scales hit rates up/down based on batter/pitcher hand matchup
    hr_rate *= platoon_factor
    double_rate *= platoon_factor
    single_rate *= platoon_factor
    # K rate inversely affected (platoon advantage = fewer K's)
    k_rate *= (2.0 - platoon_factor)  # If platoon=1.04, K drops to 0.96x

    # ── UMPIRE ADJUSTMENT ──
    # Hitter-friendly umps (factor > 1): fewer K's, more BB's
    # Pitcher-friendly umps (factor < 1): more K's, fewer BB's
    k_rate *= (2.0 - ump_factor)  # ump_factor 1.02 → K rate * 0.98
    bb_rate *= ump_factor  # ump_factor 1.02 → BB rate * 1.02

    # ── PARK FACTOR ──
    hr_rate *= park_factor ** 0.5
    double_rate *= park_factor ** 0.3

    # ── WEATHER ──
    # Weather scales overall hit rates (already captures temp/wind/humidity)
    hr_rate *= weather_factor ** 0.6  # HR most affected
    double_rate *= weather_factor ** 0.3
    single_rate *= weather_factor ** 0.2

    # ── REST/TRAVEL FATIGUE ──
    # Fatigue reduces overall offensive output
    hr_rate *= rest_factor
    double_rate *= rest_factor
    single_rate *= rest_factor
    bb_rate *= rest_factor ** 0.5  # Plate discipline less affected

    # ── NORMALIZE ──
    total_event = bb_rate + hbp_rate + k_rate + hr_rate + double_rate + triple_rate + single_rate
    bip_out_rate = max(1.0 - total_event, 0.15)

    total = bb_rate + hbp_rate + k_rate + hr_rate + double_rate + triple_rate + single_rate + bip_out_rate
    return {
        "bb": bb_rate / total,
        "hbp": hbp_rate / total,
        "k": k_rate / total,
        "hr": hr_rate / total,
        "triple": triple_rate / total,
        "double": double_rate / total,
        "single": single_rate / total,
        "bip_out": bip_out_rate / total,
    }


def simulate_pa(probs: dict[str, float], rng: random.Random) -> str:
    """Simulate a single plate appearance outcome."""
    r = rng.random()
    cumulative = 0.0
    for outcome, prob in probs.items():
        cumulative += prob
        if r < cumulative:
            return outcome
    return "bip_out"


def advance_runners(
    bases: list[int], outcome: str, outs: int, rng: random.Random
) -> tuple[list[int], int, int]:
    """Advance baserunners and return (new_bases, runs_scored, new_outs)."""
    runs = 0
    new_bases = [0, 0, 0]

    if outcome == "hr":
        runs = sum(bases) + 1
        return [0, 0, 0], runs, outs

    if outcome == "triple":
        runs = sum(bases)
        return [0, 0, 1], runs, outs

    if outcome == "double":
        runs += bases[2] + bases[1]
        if bases[0]:
            if rng.random() < 0.40:
                runs += 1
            else:
                new_bases[2] = 1
        new_bases[1] = 1
        return new_bases, runs, outs

    if outcome == "single":
        runs += bases[2]
        if bases[1]:
            if rng.random() < 0.65:
                runs += 1
            else:
                new_bases[2] = 1
        if bases[0]:
            if new_bases[2]:
                new_bases[1] = 1
            else:
                if rng.random() < 0.30:
                    new_bases[2] = 1
                else:
                    new_bases[1] = 1
        new_bases[0] = 1
        return new_bases, runs, outs

    if outcome in ("bb", "hbp"):
        if bases[0]:
            if bases[1]:
                if bases[2]:
                    runs += 1
                new_bases[2] = 1
            new_bases[1] = 1
        else:
            new_bases[1] = bases[1]
            new_bases[2] = bases[2]
        new_bases[0] = 1
        return new_bases, runs, outs

    if outcome in ("k", "bip_out"):
        new_outs = outs + 1
        new_bases = bases.copy()
        if outcome == "bip_out" and outs < 2:
            if bases[2] and rng.random() < 0.50:
                runs += 1
                new_bases[2] = 0
            if bases[1] and not new_bases[2] and rng.random() < 0.30:
                new_bases[2] = 1
                new_bases[1] = 0
        return new_bases, runs, new_outs

    return bases.copy(), 0, outs + 1


def simulate_game(
    away_lineup: list[dict],
    home_lineup: list[dict],
    away_starter: dict | None,
    home_starter: dict | None,
    away_bp_pitcher: dict | None,
    home_bp_pitcher: dict | None,
    away_sp_ip: float,
    home_sp_ip: float,
    park_factor: float,
    away_platoon_factors: dict[int, float] | None,
    home_platoon_factors: dict[int, float] | None,
    ump_factor: float,
    weather_factor: float,
    away_rest: float,
    home_rest: float,
    rng: random.Random,
) -> dict[str, Any]:
    """Simulate a full 9+ inning game with all factors."""
    batter_projs = get_batter_projections()

    def init_stats(lineup):
        stats = {}
        for p in lineup:
            stats[p["id"]] = {
                "name": p["name"], "pos": p["pos"],
                "pa": 0, "ab": 0, "h": 0, "bb": 0, "k": 0, "hbp": 0,
                "hr": 0, "double": 0, "triple": 0, "single": 0,
                "r": 0, "rbi": 0,
            }
        return stats

    away_stats = init_stats(away_lineup)
    home_stats = init_stats(home_lineup)

    # Determine when starters exit (in terms of PA seen)
    # ~4.3 PA per inning, so a 6 IP starter faces ~26 batters
    away_sp_pa_limit = int(away_sp_ip * 4.3)
    home_sp_pa_limit = int(home_sp_ip * 4.3)

    def get_probs(player_id, pitcher, is_bullpen, platoon_factors, rest):
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
            ump_factor, weather_factor, rest,
        )

    # Game state
    linescore_away = []
    linescore_home = []
    away_idx = 0
    home_idx = 0
    away_total = 0
    home_total = 0
    away_pa_count = 0
    home_pa_count = 0

    def simulate_half_inning(lineup, stats, batter_idx, pa_count,
                             starter, bp_pitcher, sp_pa_limit,
                             platoon_factors, rest):
        outs = 0
        runs = 0
        bases = [0, 0, 0]

        while outs < 3:
            batter = lineup[batter_idx % 9]
            pid = batter["id"]

            # Determine active pitcher (starter or bullpen)
            if pa_count < sp_pa_limit:
                active_pitcher = starter
            else:
                active_pitcher = bp_pitcher

            is_bp = pa_count >= sp_pa_limit
            pa_probs = get_probs(pid, active_pitcher, is_bp, platoon_factors, rest)
            outcome = simulate_pa(pa_probs, rng)

            s = stats[pid]
            s["pa"] += 1
            pa_count += 1

            if outcome in ("k", "bip_out", "hr", "single", "double", "triple"):
                s["ab"] += 1
            if outcome == "bb":
                s["bb"] += 1
            elif outcome == "hbp":
                s["hbp"] += 1
            elif outcome == "k":
                s["k"] += 1
            elif outcome == "hr":
                s["h"] += 1; s["hr"] += 1
            elif outcome == "double":
                s["h"] += 1; s["double"] += 1
            elif outcome == "triple":
                s["h"] += 1; s["triple"] += 1
            elif outcome == "single":
                s["h"] += 1; s["single"] += 1

            bases, scored, outs = advance_runners(bases, outcome, outs, rng)

            if outcome in ("hr", "single", "double", "triple"):
                s["rbi"] += scored
            elif outcome in ("bb", "hbp") and scored > 0:
                s["rbi"] += scored

            runs += scored
            batter_idx += 1

        return runs, batter_idx, pa_count

    # Play 9 innings
    for inning in range(1, 10):
        inning_runs, away_idx, away_pa_count = simulate_half_inning(
            away_lineup, away_stats, away_idx, away_pa_count,
            home_starter, home_bp_pitcher, home_sp_pa_limit,
            away_platoon_factors, away_rest,
        )
        linescore_away.append(inning_runs)
        away_total += inning_runs

        if inning == 9 and home_total > away_total:
            linescore_home.append(0)
            break

        inning_runs, home_idx, home_pa_count = simulate_half_inning(
            home_lineup, home_stats, home_idx, home_pa_count,
            away_starter, away_bp_pitcher, away_sp_pa_limit,
            home_platoon_factors, home_rest,
        )
        linescore_home.append(inning_runs)
        home_total += inning_runs

        if inning == 9 and home_total > away_total:
            break

    # Extra innings
    max_extras = 5
    extra = 0
    while away_total == home_total and extra < max_extras:
        extra += 1
        inning_runs, away_idx, away_pa_count = simulate_half_inning(
            away_lineup, away_stats, away_idx, away_pa_count,
            home_starter, home_bp_pitcher, home_sp_pa_limit,
            away_platoon_factors, away_rest,
        )
        linescore_away.append(inning_runs)
        away_total += inning_runs

        inning_runs, home_idx, home_pa_count = simulate_half_inning(
            home_lineup, home_stats, home_idx, home_pa_count,
            away_starter, away_bp_pitcher, away_sp_pa_limit,
            home_platoon_factors, home_rest,
        )
        linescore_home.append(inning_runs)
        home_total += inning_runs
        if home_total > away_total:
            break

    # Distribute runs scored to players proportionally
    for stats_dict, total_r in [(away_stats, away_total), (home_stats, home_total)]:
        tob = sum(s["h"] + s["bb"] + s["hbp"] for s in stats_dict.values())
        if tob > 0:
            remainder = total_r
            sorted_p = sorted(stats_dict.values(), key=lambda s: s["h"]+s["bb"]+s["hbp"], reverse=True)
            for s in sorted_p:
                pt = s["h"] + s["bb"] + s["hbp"]
                s["r"] = round(total_r * pt / tob)
                remainder -= s["r"]
            for s in sorted_p:
                if remainder <= 0:
                    break
                s["r"] += 1
                remainder -= 1

    return {
        "away_runs": away_total,
        "home_runs": home_total,
        "linescore_away": linescore_away,
        "linescore_home": linescore_home,
        "away_stats": away_stats,
        "home_stats": home_stats,
        "innings": len(linescore_away),
    }


def run_simulations(
    away_lineup: list[dict],
    home_lineup: list[dict],
    away_starter: dict | None,
    home_starter: dict | None,
    park_factor: float,
    n_sims: int = 1000,
    seed: int | None = None,
    # New 10-factor params
    away_bp_pitcher: dict | None = None,
    home_bp_pitcher: dict | None = None,
    away_sp_ip: float = 5.5,
    home_sp_ip: float = 5.5,
    away_platoon_factors: dict[int, float] | None = None,
    home_platoon_factors: dict[int, float] | None = None,
    ump_factor: float = 1.0,
    weather_factor: float = 1.0,
    away_rest: float = 1.0,
    home_rest: float = 1.0,
) -> dict[str, Any]:
    """Run N simulations with all 10 factors and aggregate results."""
    rng = random.Random(seed)

    away_wins = 0
    total_away_runs = 0
    total_home_runs = 0
    score_counts: dict[tuple[int, int], int] = defaultdict(int)
    linescore_sums_away: list[float] = []
    linescore_sums_home: list[float] = []

    stat_keys = ["pa", "ab", "h", "bb", "k", "hbp", "hr", "double", "triple",
                 "single", "r", "rbi"]
    player_totals: dict[int, dict[str, float]] = {}

    for lineup in [away_lineup, home_lineup]:
        for p in lineup:
            player_totals[p["id"]] = {k: 0.0 for k in stat_keys}
            player_totals[p["id"]]["name"] = p["name"]
            player_totals[p["id"]]["pos"] = p["pos"]

    for _ in range(n_sims):
        result = simulate_game(
            away_lineup, home_lineup,
            away_starter, home_starter,
            away_bp_pitcher, home_bp_pitcher,
            away_sp_ip, home_sp_ip,
            park_factor,
            away_platoon_factors, home_platoon_factors,
            ump_factor, weather_factor,
            away_rest, home_rest,
            rng,
        )

        a_r = result["away_runs"]
        h_r = result["home_runs"]
        total_away_runs += a_r
        total_home_runs += h_r
        if a_r > h_r:
            away_wins += 1
        score_counts[(a_r, h_r)] += 1

        for i, runs in enumerate(result["linescore_away"]):
            while len(linescore_sums_away) <= i:
                linescore_sums_away.append(0.0)
            linescore_sums_away[i] += runs
        for i, runs in enumerate(result["linescore_home"]):
            while len(linescore_sums_home) <= i:
                linescore_sums_home.append(0.0)
            linescore_sums_home[i] += runs

        for stats_dict in [result["away_stats"], result["home_stats"]]:
            for pid, s in stats_dict.items():
                for k in stat_keys:
                    player_totals[pid][k] += s[k]

    # Averages
    player_avgs = {}
    for pid, totals in player_totals.items():
        avgs = {"name": totals["name"], "pos": totals["pos"]}
        for k in stat_keys:
            avgs[k] = round(totals[k] / n_sims, 2)
        avgs["avg"] = round(avgs["h"] / avgs["ab"], 3) if avgs["ab"] > 0 else 0.0
        avgs["obp"] = round((avgs["h"]+avgs["bb"]+avgs["hbp"])/avgs["pa"], 3) if avgs["pa"] > 0 else 0.0
        player_avgs[pid] = avgs

    top_scores = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "n_sims": n_sims,
        "away_win_pct": round(away_wins / n_sims * 100, 1),
        "home_win_pct": round((n_sims - away_wins) / n_sims * 100, 1),
        "avg_away_runs": round(total_away_runs / n_sims, 2),
        "avg_home_runs": round(total_home_runs / n_sims, 2),
        "avg_total": round((total_away_runs + total_home_runs) / n_sims, 2),
        "top_scores": [(a, h, round(c / n_sims * 100, 1)) for (a, h), c in top_scores],
        "avg_linescore_away": [round(x / n_sims, 2) for x in linescore_sums_away[:9]],
        "avg_linescore_home": [round(x / n_sims, 2) for x in linescore_sums_home[:9]],
        "player_avgs": player_avgs,
    }
