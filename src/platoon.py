"""Platoon split adjustments for batter vs pitcher handedness.

MLB hitters historically perform significantly differently against
same-hand vs opposite-hand pitchers:
- RHB vs LHP: ~.020 higher wOBA than vs RHP (platoon advantage)
- LHB vs RHP: ~.015 higher wOBA than vs LHP (platoon advantage)
- Switch hitters: minimal platoon effect

This module applies standard platoon adjustments when pitcher hand
is known, scaling each batter's projected wOBA/wRC+ up or down.

Sources:
- Tom Tango, "The Book" (2006) — foundational platoon research
- FanGraphs platoon splits database
"""

import json
import os
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Standard platoon adjustment factors (relative to overall projection)
# Based on historical MLB data: ~.020 wOBA swing for platoon advantage
PLATOON_FACTORS = {
    # (batter_hand, pitcher_hand) -> multiplier on wOBA/wRC+
    ("R", "L"): 1.035,   # RHB vs LHP = platoon advantage (+3.5%)
    ("R", "R"): 0.975,   # RHB vs RHP = platoon disadvantage (-2.5%)
    ("L", "R"): 1.040,   # LHB vs RHP = platoon advantage (+4.0%)
    ("L", "L"): 0.950,   # LHB vs LHP = platoon disadvantage (-5.0%)
    ("S", "R"): 1.005,   # Switch vs RHP = slight advantage (bats left)
    ("S", "L"): 1.010,   # Switch vs LHP = slight advantage (bats right)
}

# Pitcher hand data cache
_pitcher_hand_cache: dict[int, str] | None = None
_batter_hand_cache: dict[int, str] | None = None


def load_pitcher_hands(filepath: str | None = None) -> dict[int, str]:
    """Load pitcher handedness data.

    Returns: {mlbam_id: "L" or "R"}
    """
    global _pitcher_hand_cache
    if _pitcher_hand_cache is not None:
        return _pitcher_hand_cache

    if filepath is None:
        filepath = os.path.join(DATA_DIR, "pitcher_hands.json")

    if not os.path.exists(filepath):
        _pitcher_hand_cache = {}
        return _pitcher_hand_cache

    with open(filepath) as f:
        data = json.load(f)

    # Support both {id: "L"} and [{id: X, throws: "L"}] formats
    if isinstance(data, dict):
        _pitcher_hand_cache = {int(k): v for k, v in data.items()}
    elif isinstance(data, list):
        _pitcher_hand_cache = {}
        for entry in data:
            pid = entry.get("id") or entry.get("mlbam_id")
            hand = entry.get("throws") or entry.get("hand") or entry.get("pitchHand")
            if pid and hand:
                _pitcher_hand_cache[int(pid)] = hand[0].upper()  # "Left" -> "L"
    else:
        _pitcher_hand_cache = {}

    return _pitcher_hand_cache


def load_batter_hands(filepath: str | None = None) -> dict[int, str]:
    """Load batter handedness data.

    Returns: {mlbam_id: "L", "R", or "S"}
    """
    global _batter_hand_cache
    if _batter_hand_cache is not None:
        return _batter_hand_cache

    if filepath is None:
        filepath = os.path.join(DATA_DIR, "batter_hands.json")

    if not os.path.exists(filepath):
        _batter_hand_cache = {}
        return _batter_hand_cache

    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict):
        _batter_hand_cache = {int(k): v for k, v in data.items()}
    elif isinstance(data, list):
        _batter_hand_cache = {}
        for entry in data:
            pid = entry.get("id") or entry.get("mlbam_id")
            hand = entry.get("bats") or entry.get("hand") or entry.get("batSide")
            if pid and hand:
                h = hand[0].upper()
                if h == "B":  # "Both" = switch hitter
                    h = "S"
                _batter_hand_cache[int(pid)] = h
    else:
        _batter_hand_cache = {}

    return _batter_hand_cache


def get_platoon_factor(batter_hand: str, pitcher_hand: str) -> float:
    """Get the platoon adjustment factor for a batter/pitcher matchup.

    Returns multiplier on offensive production (1.0 = no adjustment).
    """
    return PLATOON_FACTORS.get((batter_hand, pitcher_hand), 1.0)


def compute_lineup_platoon_adjustments(
    lineup: list[dict],
    pitcher_hand: str,
) -> dict[int, dict[str, float]]:
    """Compute platoon adjustments for each batter in a lineup.

    Returns: {mlbam_id: {"vs_L": factor, "vs_R": factor, "active": factor}}
    """
    batter_hands = load_batter_hands()
    adjustments = {}

    for player in lineup:
        pid = player.get("id")
        if pid is None:
            continue

        bat_hand = batter_hands.get(pid, "R")  # Default to RHB if unknown

        vs_l = get_platoon_factor(bat_hand, "L")
        vs_r = get_platoon_factor(bat_hand, "R")
        active = get_platoon_factor(bat_hand, pitcher_hand)

        adjustments[pid] = {
            "vs_L": vs_l,
            "vs_R": vs_r,
            "active": active,
            "batter_hand": bat_hand,
        }

    return adjustments


def compute_team_platoon_factor(
    lineup: list[dict],
    pitcher_hand: str,
) -> float:
    """Compute aggregate platoon advantage/disadvantage for a lineup vs a pitcher.

    Returns a single multiplier representing the lineup-level platoon effect.
    """
    adjustments = compute_lineup_platoon_adjustments(lineup, pitcher_hand)

    if not adjustments:
        return 1.0

    # PA-weighted average of platoon factors
    pa_weights = {1: 4.8, 2: 4.65, 3: 4.53, 4: 4.41, 5: 4.30,
                  6: 4.19, 7: 4.08, 8: 3.97, 9: 3.87}

    total_weight = 0
    weighted_factor = 0

    for i, player in enumerate(lineup):
        pid = player.get("id")
        order = player.get("batting_order", i + 1)
        weight = pa_weights.get(order, 4.0)

        adj = adjustments.get(pid, {})
        factor = adj.get("active", 1.0)

        weighted_factor += factor * weight
        total_weight += weight

    return weighted_factor / total_weight if total_weight > 0 else 1.0
