"""Market edge detection and bankroll management module.

Compares model win probabilities against posted betting lines to
identify +EV (positive expected value) opportunities. Uses the
Kelly Criterion to size bets optimally.

Key concepts:
- Implied probability: what the market price says the true probability is
- Edge: difference between model probability and implied probability
- Vig: the bookmaker's margin (~4.5% on standard -110/-110 lines)
- Kelly fraction: optimal bet size based on edge and odds
- Half-Kelly: common conservative approach (half the Kelly fraction)
"""

from typing import Any


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


def american_to_implied_prob(american: int) -> float:
    """Convert American odds to implied probability (includes vig)."""
    if american < 0:
        return abs(american) / (abs(american) + 100)
    else:
        return 100 / (american + 100)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(round(-100 / (decimal_odds - 1)))


def remove_vig(away_odds: int, home_odds: int) -> tuple[float, float]:
    """Remove vig from a two-way market to get true implied probabilities.

    Uses the multiplicative method (most accurate for MLB).
    Returns (away_true_prob, home_true_prob) summing to 1.0.
    """
    away_implied = american_to_implied_prob(away_odds)
    home_implied = american_to_implied_prob(home_odds)
    total = away_implied + home_implied  # > 1.0 due to vig

    return away_implied / total, home_implied / total


def calculate_edge(model_prob: float, market_odds: int) -> dict[str, float]:
    """Calculate the edge between model probability and market price.

    Returns:
        {
            "model_prob": float,      # Our probability
            "implied_prob": float,    # Market's probability (with vig)
            "no_vig_prob": float,     # Market's probability (vig removed)
            "edge": float,            # Edge = model_prob - no_vig_prob
            "edge_pct": float,        # Edge as percentage
            "decimal_odds": float,    # Decimal odds
            "ev_per_unit": float,     # Expected value per $1 bet
        }
    """
    implied = american_to_implied_prob(market_odds)
    decimal_odds = american_to_decimal(market_odds)

    # Edge against the actual price (includes vig)
    ev_per_unit = (model_prob * decimal_odds) - 1.0

    return {
        "model_prob": round(model_prob, 4),
        "implied_prob": round(implied, 4),
        "edge": round(model_prob - implied, 4),
        "edge_pct": round((model_prob - implied) * 100, 2),
        "decimal_odds": round(decimal_odds, 3),
        "ev_per_unit": round(ev_per_unit, 4),
    }


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Calculate the Kelly Criterion optimal bet fraction.

    f* = (bp - q) / b
    where:
        b = decimal_odds - 1 (net payout)
        p = probability of winning
        q = 1 - p (probability of losing)
    """
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p

    if b <= 0:
        return 0.0

    f = (b * p - q) / b
    return max(0.0, f)


def half_kelly(model_prob: float, decimal_odds: float) -> float:
    """Half-Kelly: more conservative bet sizing (standard practice)."""
    return kelly_fraction(model_prob, decimal_odds) / 2


def analyze_game(
    away_team: str,
    home_team: str,
    model_away_prob: float,
    model_home_prob: float,
    away_odds: int | None = None,
    home_odds: int | None = None,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Full edge analysis for a single game.

    Args:
        model_away_prob: Model's probability for away team (0-1)
        model_home_prob: Model's probability for home team (0-1)
        away_odds: Posted American odds for away team (e.g., +150)
        home_odds: Posted American odds for home team (e.g., -170)
        bankroll: Current bankroll for bet sizing

    Returns comprehensive edge analysis.
    """
    result = {
        "away_team": away_team,
        "home_team": home_team,
        "model_away_prob": round(model_away_prob * 100, 1),
        "model_home_prob": round(model_home_prob * 100, 1),
        "has_market_odds": away_odds is not None and home_odds is not None,
        "edges": [],
        "best_bet": None,
    }

    if away_odds is None or home_odds is None:
        # No market odds available — just show model probabilities
        return result

    # No-vig probabilities
    no_vig_away, no_vig_home = remove_vig(away_odds, home_odds)
    result["no_vig_away"] = round(no_vig_away * 100, 1)
    result["no_vig_home"] = round(no_vig_home * 100, 1)

    # Analyze away side
    away_edge = calculate_edge(model_away_prob, away_odds)
    away_edge["side"] = "away"
    away_edge["team"] = away_team
    away_edge["odds"] = away_odds
    away_edge["kelly"] = round(kelly_fraction(model_away_prob, away_edge["decimal_odds"]) * 100, 2)
    away_edge["half_kelly"] = round(half_kelly(model_away_prob, away_edge["decimal_odds"]) * 100, 2)
    away_edge["bet_amount"] = round(half_kelly(model_away_prob, away_edge["decimal_odds"]) * bankroll, 2)
    result["edges"].append(away_edge)

    # Analyze home side
    home_edge = calculate_edge(model_home_prob, home_odds)
    home_edge["side"] = "home"
    home_edge["team"] = home_team
    home_edge["odds"] = home_odds
    home_edge["kelly"] = round(kelly_fraction(model_home_prob, home_edge["decimal_odds"]) * 100, 2)
    home_edge["half_kelly"] = round(half_kelly(model_home_prob, home_edge["decimal_odds"]) * 100, 2)
    home_edge["bet_amount"] = round(half_kelly(model_home_prob, home_edge["decimal_odds"]) * bankroll, 2)
    result["edges"].append(home_edge)

    # Determine best bet (if any has positive EV)
    positive_ev = [e for e in result["edges"] if e["ev_per_unit"] > 0]
    if positive_ev:
        best = max(positive_ev, key=lambda e: e["ev_per_unit"])
        result["best_bet"] = {
            "side": best["side"],
            "team": best["team"],
            "odds": best["odds"],
            "edge_pct": best["edge_pct"],
            "ev_per_unit": best["ev_per_unit"],
            "half_kelly_pct": best["half_kelly"],
            "bet_amount": best["bet_amount"],
            "rating": _edge_rating(best["edge_pct"]),
        }

    return result


def _edge_rating(edge_pct: float) -> str:
    """Rate the strength of an edge."""
    if edge_pct >= 8.0:
        return "STRONG"
    elif edge_pct >= 5.0:
        return "GOOD"
    elif edge_pct >= 3.0:
        return "LEAN"
    elif edge_pct >= 1.0:
        return "MARGINAL"
    else:
        return "NO EDGE"


def analyze_total(
    model_total: float,
    posted_total: float,
    over_odds: int = -110,
    under_odds: int = -110,
    model_over_pct: float = 0.5,
    model_under_pct: float = 0.5,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Analyze edge on over/under total."""
    over_edge = calculate_edge(model_over_pct, over_odds)
    under_edge = calculate_edge(model_under_pct, under_odds)

    result = {
        "model_total": model_total,
        "posted_total": posted_total,
        "diff": round(model_total - posted_total, 1),
        "over_edge": over_edge,
        "under_edge": under_edge,
        "best_bet": None,
    }

    if over_edge["ev_per_unit"] > under_edge["ev_per_unit"] and over_edge["ev_per_unit"] > 0:
        result["best_bet"] = {
            "side": "OVER",
            "edge_pct": over_edge["edge_pct"],
            "ev_per_unit": over_edge["ev_per_unit"],
            "half_kelly_pct": round(half_kelly(model_over_pct, over_edge["decimal_odds"]) * 100, 2),
            "bet_amount": round(half_kelly(model_over_pct, over_edge["decimal_odds"]) * bankroll, 2),
        }
    elif under_edge["ev_per_unit"] > 0:
        result["best_bet"] = {
            "side": "UNDER",
            "edge_pct": under_edge["edge_pct"],
            "ev_per_unit": under_edge["ev_per_unit"],
            "half_kelly_pct": round(half_kelly(model_under_pct, under_edge["decimal_odds"]) * 100, 2),
            "bet_amount": round(half_kelly(model_under_pct, under_edge["decimal_odds"]) * bankroll, 2),
        }

    return result
