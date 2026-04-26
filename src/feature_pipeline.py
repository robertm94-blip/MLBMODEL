"""MLB feature engineering pipeline.

Builds a per-(game, team) feature record by composing existing modules:
- src.lineup_offense   -> PA-weighted, platoon-adjusted lineup wRC+
- src.platoon          -> per-batter platoon multipliers vs the opposing starter's hand
- src.projections      -> blended (Steamer/ZiPS/THE BAT) starter FIP/ERA/IP/GS
- src.bullpen          -> IP-weighted team bullpen FIP/ERA, projected starter length
- src.park_factors     -> runs park factor from data/park_factors.json

Adds explicit Opener / bullpen-heavy flags. The pitching-FIP blend uses the
existing IP-based starter share (no hard switch on the flags).
"""

from __future__ import annotations

from typing import Any

from src.bullpen import (
    get_starter_projected_length,
    load_team_bullpen_projections,
)
from src.lineup_offense import compute_lineup_offensive_factor
from src.park_factors import get_park_factor
from src.platoon import compute_lineup_platoon_adjustments, load_pitcher_hands
from src.projections import load_all_pitcher_projections


OPENER_IP_PER_GS_THRESHOLD = 4.0
OPENER_GS_FLOOR = 5
BULLPEN_HEAVY_SHARE_THRESHOLD = 0.4
STARTER_SHARE_CAP = 0.85
INNINGS_PER_GAME = 9.0


def _starter_proj(starter_name: str | None) -> dict[str, Any] | None:
    if not starter_name:
        return None
    projections = load_all_pitcher_projections()
    return projections.get(starter_name)


def _starter_hand(starter_id: int | None) -> str | None:
    if starter_id is None:
        return None
    hands = load_pitcher_hands()
    return hands.get(int(starter_id))


def _detect_opener(starter_proj: dict[str, Any] | None) -> bool:
    if not starter_proj:
        return False
    gs = float(starter_proj.get("gs", 0) or 0)
    ip = float(starter_proj.get("ip", 0) or 0)
    if gs < OPENER_GS_FLOOR:
        return True
    if gs >= 1 and ip > 0 and (ip / max(gs, 1)) < OPENER_IP_PER_GS_THRESHOLD:
        return True
    return False


def build_team_features(
    *,
    sport: str = "mlb",
    game_id: str | int,
    side: str,
    game_date: str,
    team_id: str | int,
    team_name: str,
    opponent_team_id: str | int | None = None,
    opponent_team_name: str | None = None,
    venue_id: int | None,
    lineup: list[dict[str, Any]],
    starter_id: int | None,
    starter_name: str | None,
    opposing_starter_id: int | None,
    opposing_starter_name: str | None,
    lineup_source: str | None = None,
) -> dict[str, Any]:
    """Compute the full feature row for one team in one game."""
    opp_hand = _starter_hand(opposing_starter_id)
    own_starter_hand = _starter_hand(starter_id)

    # Offense (lineup wRC+ + platoon adjustment vs opposing starter's hand)
    platoon_adj = (
        compute_lineup_platoon_adjustments(lineup, opp_hand) if opp_hand else None
    )
    offensive_factor, offense_breakdown = compute_lineup_offensive_factor(
        lineup, pitcher_hand=opp_hand, platoon_adjustments=platoon_adj
    )
    platoon_factor = _aggregate_platoon_factor(lineup, platoon_adj) if platoon_adj else 1.0

    # Listed starter's blended projection (across Steamer/ZiPS/THE BAT)
    starter_proj = _starter_proj(starter_name)
    starter_fip = _safe(starter_proj, "fip")
    starter_era = _safe(starter_proj, "era")
    starter_systems_count = (
        int(starter_proj.get("systems_count", 0)) if starter_proj else 0
    )
    starter_ip = get_starter_projected_length(starter_name or "", load_all_pitcher_projections())

    # Bullpen blended projection
    bullpens = load_team_bullpen_projections()
    bullpen = bullpens.get(team_name) or {}
    bullpen_fip = _safe(bullpen, "fip")
    bullpen_era = _safe(bullpen, "era")
    bullpen_ip = _safe(bullpen, "total_ip")

    # Pitching-FIP blend (smooth IP-based weighting; matches src/bullpen.py)
    starter_share = min(starter_ip / INNINGS_PER_GAME, STARTER_SHARE_CAP)
    pitching_blended_fip = _blend_fip(starter_fip, bullpen_fip, starter_share)

    # Opener / bullpen-heavy flags (informational only; do not switch the blend)
    is_opener = _detect_opener(starter_proj)
    is_bullpen_heavy = (starter_share < BULLPEN_HEAVY_SHARE_THRESHOLD) or (
        is_opener and starter_share < 0.55
    )

    park_factor = get_park_factor(venue_id)

    return {
        "sport": sport,
        "game_id": str(game_id),
        "team_id": str(team_id),
        "side": side,
        "game_date": game_date,
        "venue_id": venue_id,
        "team_name": team_name,
        "opponent_team_id": opponent_team_id,
        "opponent_team_name": opponent_team_name,

        "lineup_wrc_plus": _round(offense_breakdown.get("lineup_wrc_plus"), 2),
        "offensive_factor": _round(offensive_factor, 4),
        "platoon_factor": _round(platoon_factor, 4),
        "lineup_size": len(lineup),
        "lineup_source": lineup_source,

        "starter_player_id": starter_id,
        "starter_name": starter_name,
        "starter_hand": own_starter_hand,
        "starter_fip": _round(starter_fip, 3),
        "starter_era": _round(starter_era, 3),
        "starter_projected_ip": _round(starter_ip, 2),
        "starter_systems_count": starter_systems_count or None,
        "bullpen_fip": _round(bullpen_fip, 3),
        "bullpen_era": _round(bullpen_era, 3),
        "bullpen_total_ip": _round(bullpen_ip, 1),
        "starter_share": _round(starter_share, 4),
        "pitching_blended_fip": _round(pitching_blended_fip, 3),

        "is_opener": bool(is_opener),
        "is_bullpen_heavy": bool(is_bullpen_heavy),

        "park_runs_factor": _round(park_factor, 4),

        "raw": {
            "opposing_starter_hand": opp_hand,
            "opposing_starter_name": opposing_starter_name,
            "offense_players": offense_breakdown.get("players", []),
            "bullpen_n_relievers": bullpen.get("n_relievers"),
            "starter_systems": (starter_proj or {}).get("systems"),
        },
    }


def build_game_features(
    *,
    sport: str = "mlb",
    game_id: str | int,
    game_date: str,
    venue_id: int | None,
    home: dict[str, Any],
    away: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build features for both sides of a game in one call.

    `home` and `away` should each contain: team_id, team_name, lineup,
    starter_id, starter_name, optional lineup_source.
    """
    home_features = build_team_features(
        sport=sport,
        game_id=game_id,
        side="home",
        game_date=game_date,
        venue_id=venue_id,
        team_id=home["team_id"],
        team_name=home["team_name"],
        opponent_team_id=away.get("team_id"),
        opponent_team_name=away.get("team_name"),
        lineup=home.get("lineup") or [],
        starter_id=home.get("starter_id"),
        starter_name=home.get("starter_name"),
        opposing_starter_id=away.get("starter_id"),
        opposing_starter_name=away.get("starter_name"),
        lineup_source=home.get("lineup_source"),
    )
    away_features = build_team_features(
        sport=sport,
        game_id=game_id,
        side="away",
        game_date=game_date,
        venue_id=venue_id,
        team_id=away["team_id"],
        team_name=away["team_name"],
        opponent_team_id=home.get("team_id"),
        opponent_team_name=home.get("team_name"),
        lineup=away.get("lineup") or [],
        starter_id=away.get("starter_id"),
        starter_name=away.get("starter_name"),
        opposing_starter_id=home.get("starter_id"),
        opposing_starter_name=home.get("starter_name"),
        lineup_source=away.get("lineup_source"),
    )
    return {"home": home_features, "away": away_features}


def _safe(d: dict[str, Any] | None, key: str) -> float | None:
    if not d:
        return None
    val = d.get(key)
    if val in (None, 0, 0.0):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, ndigits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _blend_fip(starter_fip: float | None, bullpen_fip: float | None, starter_share: float) -> float | None:
    if starter_fip is None and bullpen_fip is None:
        return None
    if starter_fip is None:
        return bullpen_fip
    if bullpen_fip is None:
        return starter_fip
    return starter_share * starter_fip + (1 - starter_share) * bullpen_fip


def _aggregate_platoon_factor(
    lineup: list[dict[str, Any]],
    platoon_adj: dict[int, dict[str, float]],
) -> float:
    pa_weights = {1: 4.80, 2: 4.65, 3: 4.53, 4: 4.41, 5: 4.30,
                  6: 4.19, 7: 4.08, 8: 3.97, 9: 3.87}
    total = 0.0
    weighted = 0.0
    for i, player in enumerate(lineup):
        order = player.get("batting_order") or (i + 1)
        weight = pa_weights.get(order, 4.0)
        adj = platoon_adj.get(player.get("id"), {})
        factor = adj.get("active", 1.0)
        weighted += weight * factor
        total += weight
    return weighted / total if total else 1.0
