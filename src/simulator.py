"""Monte Carlo MLB game simulator.

Simulates games plate-appearance by plate-appearance using projected
batter rates adjusted for opposing pitcher quality. Tracks full
baserunning state to produce realistic box scores.

Each PA outcome is drawn from batter-specific probability distributions
derived from blended Steamer/ZiPS/THE BAT projections:
  - Strikeout, Walk, HBP, Single, Double, Triple, Home Run, BIP Out

Baserunner advancement uses simplified probabilistic rules based on
the type of hit and number of outs.
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

# League-average rates (2025 MLB)
LEAGUE_AVG = {
    "bb_pct": 0.083,
    "k_pct": 0.225,
    "hbp_pct": 0.012,
    "hr_per_h": 0.145,  # HR as fraction of hits
    "triple_per_h": 0.020,
    "double_per_h": 0.200,
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
        blended[mlbam_id] = {
            "name": entries[0].get("PlayerName", "Unknown"),
            "team": entries[0].get("Team", ""),
            "pa": sum(e.get("PA", 0) or 0 for e in entries) / n,
            "ab": sum(e.get("AB", 0) or 0 for e in entries) / n,
            "h": sum(e.get("H", 0) or 0 for e in entries) / n,
            "hr": sum(e.get("HR", 0) or 0 for e in entries) / n,
            "r": sum(e.get("R", 0) or 0 for e in entries) / n,
            "rbi": sum(e.get("RBI", 0) or 0 for e in entries) / n,
            "sb": sum(e.get("SB", 0) or 0 for e in entries) / n,
            "bb": sum(e.get("BB", 0) or 0 for e in entries) / n,
            "so": sum(e.get("SO", 0) or 0 for e in entries) / n,
            "hbp": sum(e.get("HBP", 0) or 0 for e in entries) / n,
            "double": sum(e.get("2B", 0) or 0 for e in entries) / n,
            "triple": sum(e.get("3B", 0) or 0 for e in entries) / n,
            "avg": sum(e.get("AVG", 0) or 0 for e in entries) / n,
            "obp": sum(e.get("OBP", 0) or 0 for e in entries) / n,
            "slg": sum(e.get("SLG", 0) or 0 for e in entries) / n,
            "bb_pct": sum(e.get("BB%", 0) or 0 for e in entries) / n,
            "k_pct": sum(e.get("K%", 0) or 0 for e in entries) / n,
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
) -> dict[str, float]:
    """Compute outcome probabilities for a single plate appearance.

    Adjusts batter's projected rates by opposing pitcher's quality
    relative to league average. Park factor scales hit/HR probability.

    Returns dict with probabilities summing to 1.0:
        {bb, hbp, k, hr, triple, double, single, bip_out}
    """
    # Batter base rates
    bb_rate = batter.get("bb_pct", LEAGUE_AVG["bb_pct"])
    k_rate = batter.get("k_pct", LEAGUE_AVG["k_pct"])

    pa = batter.get("pa", 600)
    hbp_rate = batter.get("hbp", 0) / pa if pa > 0 else LEAGUE_AVG["hbp_pct"]

    # Hit rates from counting stats
    ab = batter.get("ab", 550)
    h = batter.get("h", 140)
    hr = batter.get("hr", 20)
    doubles = batter.get("double", 28)
    triples = batter.get("triple", 3)
    singles = h - hr - doubles - triples

    # Rates per PA (not per AB)
    if pa > 0:
        hr_rate = hr / pa
        double_rate = doubles / pa
        triple_rate = triples / pa
        single_rate = singles / pa
    else:
        hr_rate = 0.030
        double_rate = 0.045
        triple_rate = 0.004
        single_rate = 0.140

    # Pitcher adjustment: shift rates toward pitcher's tendencies
    if pitcher and pitcher.get("era", 0) > 0:
        league_avg_rpg = get_league_avg_rpg()

        # Pitcher K% adjustment
        p_k_pct = pitcher.get("k_pct", LEAGUE_AVG["k_pct"])
        k_multiplier = p_k_pct / LEAGUE_AVG["k_pct"] if LEAGUE_AVG["k_pct"] > 0 else 1.0
        k_rate = k_rate * (0.6 + 0.4 * k_multiplier)  # 60% batter, 40% pitcher

        # Pitcher BB% adjustment
        p_bb_pct = pitcher.get("bb_pct", LEAGUE_AVG["bb_pct"])
        bb_multiplier = p_bb_pct / LEAGUE_AVG["bb_pct"] if LEAGUE_AVG["bb_pct"] > 0 else 1.0
        bb_rate = bb_rate * (0.6 + 0.4 * bb_multiplier)

        # Hit suppression based on pitcher ERA/FIP quality
        era = pitcher.get("era", 4.50)
        fip = pitcher.get("fip", era)
        blended = era * 0.4 + fip * 0.6
        hit_multiplier = blended / league_avg_rpg if league_avg_rpg > 0 else 1.0

        # Scale hit rates (better pitcher = fewer hits)
        hr_rate *= hit_multiplier * (0.85 + 0.15 * park_factor)
        double_rate *= hit_multiplier
        triple_rate *= hit_multiplier
        single_rate *= hit_multiplier

    # Apply park factor to power numbers
    hr_rate *= park_factor ** 0.5  # sqrt dampening — park affects HR less than linear
    double_rate *= (park_factor ** 0.3)

    # Normalize to ensure probabilities sum to 1.0
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
    """Advance baserunners and return (new_bases, runs_scored, new_outs).

    bases: [first, second, third] — 0 = empty, 1 = occupied
    Uses simplified probabilistic advancement rules.
    """
    runs = 0
    new_bases = [0, 0, 0]

    if outcome == "hr":
        runs = sum(bases) + 1  # Everyone scores including batter
        return [0, 0, 0], runs, outs

    if outcome == "triple":
        runs = sum(bases)
        return [0, 0, 1], runs, outs

    if outcome == "double":
        # Runners on 2nd/3rd score; runner on 1st goes to 3rd (80%) or scores (20%)
        runs += bases[2]  # 3rd scores
        runs += bases[1]  # 2nd scores
        if bases[0]:
            if rng.random() < 0.40:
                runs += 1
            else:
                new_bases[2] = 1
        new_bases[1] = 1  # Batter on 2nd
        return new_bases, runs, outs

    if outcome == "single":
        # 3rd scores; 2nd scores (65%) or goes to 3rd (35%); 1st goes to 2nd or 3rd
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
        new_bases[0] = 1  # Batter on 1st
        return new_bases, runs, outs

    if outcome in ("bb", "hbp"):
        # Forced advancement only
        if bases[0]:
            if bases[1]:
                if bases[2]:
                    runs += 1  # Bases loaded walk
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

        # On BIP out with < 2 outs: runner on 3rd may score on sac fly/groundout
        if outcome == "bip_out" and outs < 2:
            if bases[2] and rng.random() < 0.50:
                runs += 1
                new_bases[2] = 0
            # Runner on 2nd may advance to 3rd on groundout
            if bases[1] and not new_bases[2] and rng.random() < 0.30:
                new_bases[2] = 1
                new_bases[1] = 0

        return new_bases, runs, new_outs

    return bases.copy(), 0, outs + 1


def simulate_game(
    away_lineup: list[dict],
    home_lineup: list[dict],
    away_pitcher: dict | None,
    home_pitcher: dict | None,
    park_factor: float,
    rng: random.Random,
) -> dict[str, Any]:
    """Simulate a full 9+ inning game.

    Returns per-player stat lines and final linescore.
    """
    # Initialize player stats
    def init_stats(lineup):
        stats = {}
        for p in lineup:
            stats[p["id"]] = {
                "name": p["name"],
                "pos": p["pos"],
                "pa": 0, "ab": 0, "h": 0, "bb": 0, "k": 0, "hbp": 0,
                "hr": 0, "double": 0, "triple": 0, "single": 0,
                "r": 0, "rbi": 0,
            }
        return stats

    away_stats = init_stats(away_lineup)
    home_stats = init_stats(home_lineup)

    # Pre-compute PA probabilities for each batter
    batter_projs = get_batter_projections()

    def get_probs(player_id, opposing_pitcher):
        proj = batter_projs.get(player_id)
        if proj is None:
            # Fallback: league-average hitter
            proj = {
                "bb_pct": LEAGUE_AVG["bb_pct"],
                "k_pct": LEAGUE_AVG["k_pct"],
                "pa": 500, "ab": 440, "h": 110, "hr": 15,
                "double": 22, "triple": 3, "hbp": 5,
            }
        return compute_pa_probabilities(proj, opposing_pitcher, park_factor)

    away_probs = {p["id"]: get_probs(p["id"], home_pitcher) for p in away_lineup}
    home_probs = {p["id"]: get_probs(p["id"], away_pitcher) for p in home_lineup}

    # Game state
    linescore_away = []
    linescore_home = []
    away_idx = 0
    home_idx = 0
    away_total = 0
    home_total = 0

    def simulate_half_inning(lineup, probs, stats, batter_idx):
        outs = 0
        runs = 0
        bases = [0, 0, 0]

        while outs < 3:
            batter = lineup[batter_idx % 9]
            pid = batter["id"]
            pa_probs = probs[pid]

            outcome = simulate_pa(pa_probs, rng)

            # Record stats
            s = stats[pid]
            s["pa"] += 1

            if outcome in ("k", "bip_out", "hr", "single", "double", "triple"):
                s["ab"] += 1

            if outcome == "bb":
                s["bb"] += 1
            elif outcome == "hbp":
                s["hbp"] += 1
            elif outcome == "k":
                s["k"] += 1
            elif outcome == "hr":
                s["h"] += 1
                s["hr"] += 1
            elif outcome == "double":
                s["h"] += 1
                s["double"] += 1
            elif outcome == "triple":
                s["h"] += 1
                s["triple"] += 1
            elif outcome == "single":
                s["h"] += 1
                s["single"] += 1

            # Advance runners
            bases, scored, outs = advance_runners(bases, outcome, outs, rng)

            # Credit RBIs (not on errors or fielder's choice — simplified)
            if outcome in ("hr", "single", "double", "triple"):
                s["rbi"] += scored
            elif outcome in ("bb", "hbp") and scored > 0:
                s["rbi"] += scored

            # Credit runs
            runs += scored
            batter_idx += 1

        return runs, batter_idx

    # Play 9 innings
    for inning in range(1, 10):
        # Top: away bats
        inning_runs, away_idx = simulate_half_inning(
            away_lineup, away_probs, away_stats, away_idx
        )
        linescore_away.append(inning_runs)
        away_total += inning_runs

        # Bottom: home bats (skip if home leads in 9th)
        if inning == 9 and home_total > away_total:
            linescore_home.append(0)
            break

        inning_runs, home_idx = simulate_half_inning(
            home_lineup, home_probs, home_stats, home_idx
        )
        linescore_home.append(inning_runs)
        home_total += inning_runs

        # Walk-off in 9th
        if inning == 9 and home_total > away_total:
            break

    # Extra innings (simplified: runner on 2nd)
    max_extras = 5
    extra = 0
    while away_total == home_total and extra < max_extras:
        extra += 1
        inning_runs, away_idx = simulate_half_inning(
            away_lineup, away_probs, away_stats, away_idx
        )
        linescore_away.append(inning_runs)
        away_total += inning_runs

        inning_runs, home_idx = simulate_half_inning(
            home_lineup, home_probs, home_stats, home_idx
        )
        linescore_home.append(inning_runs)
        home_total += inning_runs

        if home_total > away_total:
            break

    # Credit runs to players who scored (distribute proportionally to OBP)
    # Simplified: distribute runs proportionally to times on base
    for stats_dict, total_r in [(away_stats, away_total), (home_stats, home_total)]:
        tob = sum(s["h"] + s["bb"] + s["hbp"] for s in stats_dict.values())
        if tob > 0:
            remainder = total_r
            sorted_players = sorted(
                stats_dict.values(), key=lambda s: s["h"] + s["bb"] + s["hbp"], reverse=True
            )
            for s in sorted_players:
                player_tob = s["h"] + s["bb"] + s["hbp"]
                s["r"] = round(total_r * player_tob / tob)
                remainder -= s["r"]
            # Distribute any rounding remainder
            for s in sorted_players:
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
    away_pitcher: dict | None,
    home_pitcher: dict | None,
    park_factor: float,
    n_sims: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run N simulations and aggregate into expected box scores.

    Returns aggregated stats for each player plus game-level summaries.
    """
    rng = random.Random(seed)

    # Accumulators
    away_wins = 0
    total_away_runs = 0
    total_home_runs = 0
    score_counts: dict[tuple[int, int], int] = defaultdict(int)
    linescore_sums_away: list[float] = []
    linescore_sums_home: list[float] = []

    # Per-player accumulators
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
            away_lineup, home_lineup, away_pitcher, home_pitcher, park_factor, rng
        )

        # Score tracking
        a_r = result["away_runs"]
        h_r = result["home_runs"]
        total_away_runs += a_r
        total_home_runs += h_r
        if a_r > h_r:
            away_wins += 1
        score_counts[(a_r, h_r)] += 1

        # Linescore tracking
        for i, runs in enumerate(result["linescore_away"]):
            while len(linescore_sums_away) <= i:
                linescore_sums_away.append(0.0)
            linescore_sums_away[i] += runs
        for i, runs in enumerate(result["linescore_home"]):
            while len(linescore_sums_home) <= i:
                linescore_sums_home.append(0.0)
            linescore_sums_home[i] += runs

        # Player stats accumulation
        for stats_dict in [result["away_stats"], result["home_stats"]]:
            for pid, s in stats_dict.items():
                for k in stat_keys:
                    player_totals[pid][k] += s[k]

    # Compute averages
    player_avgs = {}
    for pid, totals in player_totals.items():
        avgs = {"name": totals["name"], "pos": totals["pos"]}
        for k in stat_keys:
            avgs[k] = round(totals[k] / n_sims, 2)
        # Compute derived stats
        if avgs["ab"] > 0:
            avgs["avg"] = round(avgs["h"] / avgs["ab"], 3)
        else:
            avgs["avg"] = 0.0
        if avgs["pa"] > 0:
            avgs["obp"] = round((avgs["h"] + avgs["bb"] + avgs["hbp"]) / avgs["pa"], 3)
        else:
            avgs["obp"] = 0.0
        player_avgs[pid] = avgs

    # Top scores
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
