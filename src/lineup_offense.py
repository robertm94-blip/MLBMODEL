"""Lineup-specific offensive projection module.

Instead of using team-level offensive factors (which average across the full
roster), this module computes game-specific offensive strength from the
actual starting lineup's individual batter projections.

This captures:
- Lineup construction (top-heavy vs balanced)
- Rest-day substitutions (backup catcher vs starter)
- Platoon advantages when batter hand data is available
- Actual PA distribution across the 9 batting slots
"""

import json
import os
from collections import defaultdict
from typing import Any

PROJ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "projections", "2026")
PROJECTION_SYSTEMS = ["steamer", "zips", "thebat"]

_batter_cache: dict[int, dict] | None = None


def _load_all_batter_projections() -> dict[int, dict[str, float]]:
    """Load and blend batter projections keyed by MLBAM ID."""
    global _batter_cache
    if _batter_cache is not None:
        return _batter_cache

    all_data: dict[int, list[dict]] = defaultdict(list)

    for system in PROJECTION_SYSTEMS:
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
            vals = [e.get(key, 0) or 0 for e in entries]
            return sum(vals) / n

        pa = avg("PA")
        ab = avg("AB")
        h = avg("H")
        hr = avg("HR")
        bb = avg("BB")
        so = avg("SO")
        hbp = avg("HBP")
        doubles = avg("2B")
        triples = avg("3B")
        r = avg("R")
        rbi = avg("RBI")
        sb = avg("SB")

        # wOBA and wRC+ are the key offensive quality metrics
        woba = avg("wOBA")
        wrc_plus = avg("wRC+")
        ops = avg("OPS")

        blended[mlbam_id] = {
            "name": entries[0].get("PlayerName", "Unknown"),
            "team": entries[0].get("Team", ""),
            "pa": pa, "ab": ab, "h": h, "hr": hr, "bb": bb, "so": so,
            "hbp": hbp, "double": doubles, "triple": triples,
            "r": r, "rbi": rbi, "sb": sb,
            "woba": woba, "wrc_plus": wrc_plus, "ops": ops,
            "avg": h / ab if ab > 0 else 0.250,
            "obp": (h + bb + hbp) / pa if pa > 0 else 0.320,
            "slg": avg("SLG"),
            "k_pct": avg("K%"),
            "bb_pct": avg("BB%"),
            "iso": avg("ISO"),
            "babip": avg("BABIP"),
        }

    _batter_cache = blended
    return blended


# PA distribution by batting order position (MLB average)
# Leadoff gets ~4.8 PA/game, 9-hole gets ~3.8
PA_WEIGHTS = {
    1: 4.80, 2: 4.65, 3: 4.53, 4: 4.41,
    5: 4.30, 6: 4.19, 7: 4.08, 8: 3.97, 9: 3.87,
}
TOTAL_PA_PER_GAME = sum(PA_WEIGHTS.values())  # ~39.8

# League average wOBA for normalization
LEAGUE_AVG_WOBA = 0.315
LEAGUE_AVG_WRC_PLUS = 100.0


def compute_lineup_offensive_factor(
    lineup: list[dict],
    pitcher_hand: str | None = None,
    platoon_adjustments: dict[int, dict] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute offensive strength factor from actual lineup.

    Args:
        lineup: List of {id, name, position, batting_order} for 9 batters
        pitcher_hand: "L" or "R" for opposing pitcher (for platoon adjustments)
        platoon_adjustments: {mlbam_id: {vs_L: factor, vs_R: factor}}

    Returns:
        (offensive_factor, breakdown_dict)
        Factor of 1.0 = league average offense
    """
    batter_projs = _load_all_batter_projections()

    weighted_woba = 0.0
    weighted_wrc = 0.0
    total_weight = 0.0
    player_details = []

    for i, player in enumerate(lineup):
        pid = player.get("id")
        order = player.get("batting_order", i + 1)
        pa_weight = PA_WEIGHTS.get(order, 4.0)

        proj = batter_projs.get(pid)
        if proj is None:
            # Unknown batter: assume replacement level
            woba = 0.290
            wrc_plus = 80.0
            name = player.get("name", "Unknown")
        else:
            woba = proj["woba"]
            wrc_plus = proj["wrc_plus"]
            name = proj["name"]

            # Apply platoon adjustment if available
            if platoon_adjustments and pid in platoon_adjustments and pitcher_hand:
                adj = platoon_adjustments[pid]
                if pitcher_hand == "L":
                    plat_factor = adj.get("vs_L", 1.0)
                else:
                    plat_factor = adj.get("vs_R", 1.0)
                woba *= plat_factor
                wrc_plus *= plat_factor

        weighted_woba += woba * pa_weight
        weighted_wrc += wrc_plus * pa_weight
        total_weight += pa_weight

        player_details.append({
            "name": name,
            "order": order,
            "woba": round(woba, 3),
            "wrc_plus": round(wrc_plus, 1),
            "pa_weight": pa_weight,
            "has_projection": proj is not None,
        })

    if total_weight == 0:
        return 1.0, {"players": [], "lineup_woba": LEAGUE_AVG_WOBA}

    lineup_woba = weighted_woba / total_weight
    lineup_wrc = weighted_wrc / total_weight

    # Convert to offensive factor
    # wRC+ is already normalized to 100 = league average
    # Use 60% wRC+ signal, 40% wOBA signal for robustness
    wrc_factor = lineup_wrc / LEAGUE_AVG_WRC_PLUS
    woba_factor = lineup_woba / LEAGUE_AVG_WOBA
    offensive_factor = wrc_factor * 0.60 + woba_factor * 0.40

    breakdown = {
        "lineup_woba": round(lineup_woba, 3),
        "lineup_wrc_plus": round(lineup_wrc, 1),
        "offensive_factor": round(offensive_factor, 4),
        "players": player_details,
    }

    return offensive_factor, breakdown
