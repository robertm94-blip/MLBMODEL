"""Backtesting engine for MLB prediction model.

Runs the model against historical game results to measure:
- Overall accuracy (% correct at various confidence thresholds)
- Log-loss (calibration quality — lower is better)
- Brier score (probability accuracy — lower is better)
- Calibration curve (do 60% predictions win 60% of the time?)
- ROI by edge threshold (profitability at -110 juice)
- Edge-based filtering (only bet when model edge > X%)

This is the critical validation step before risking real money.
"""

import math
import json
from collections import defaultdict
from typing import Any


def load_results(filepath: str) -> list[dict]:
    """Load historical game results."""
    with open(filepath) as f:
        return json.load(f)


# ─── CORE METRICS ───────────────────────────────────────────────────────────


def log_loss(predicted_prob: float, actual_outcome: int) -> float:
    """Calculate log-loss for a single prediction.

    predicted_prob: probability assigned to the outcome that occurred
    actual_outcome: 1 if the predicted team won, 0 otherwise

    Lower is better. Perfect = 0, random coin flip = 0.693
    """
    p = max(min(predicted_prob, 0.999), 0.001)  # Clip to avoid log(0)
    if actual_outcome == 1:
        return -math.log(p)
    else:
        return -math.log(1 - p)


def brier_score(predicted_prob: float, actual_outcome: int) -> float:
    """Brier score for a single prediction.

    Lower is better. Perfect = 0, coin flip = 0.25
    """
    return (predicted_prob - actual_outcome) ** 2


def compute_accuracy(predictions: list[dict]) -> dict[str, Any]:
    """Compute comprehensive accuracy metrics from prediction results.

    Each prediction dict should have:
        - home_win_prob: model's probability that home team wins (0-1)
        - home_won: 1 if home team actually won, 0 if away won
        - away_win_prob: 1 - home_win_prob
    """
    if not predictions:
        return {"error": "No predictions to evaluate"}

    n = len(predictions)
    total_log_loss = 0
    total_brier = 0
    correct_picks = 0
    correct_by_confidence = defaultdict(lambda: {"correct": 0, "total": 0})

    # Calibration buckets (0-100% in 5% increments)
    cal_buckets = defaultdict(lambda: {"wins": 0, "total": 0, "prob_sum": 0})

    for pred in predictions:
        home_prob = pred["home_win_prob"]
        away_prob = 1 - home_prob
        home_won = pred["home_won"]

        # Log-loss (from home team perspective)
        ll = log_loss(home_prob, home_won)
        total_log_loss += ll

        # Brier score
        bs = brier_score(home_prob, home_won)
        total_brier += bs

        # Straight-up accuracy (pick the team with higher probability)
        if home_prob >= 0.5:
            picked_home = True
            pick_correct = (home_won == 1)
            confidence = home_prob
        else:
            picked_home = False
            pick_correct = (home_won == 0)
            confidence = away_prob

        if pick_correct:
            correct_picks += 1

        # Accuracy by confidence band
        conf_band = int(confidence * 100 // 5) * 5  # 50, 55, 60, ...
        correct_by_confidence[conf_band]["total"] += 1
        if pick_correct:
            correct_by_confidence[conf_band]["correct"] += 1

        # Calibration: bucket by predicted probability
        bucket = int(home_prob * 100 // 5) * 5  # 0, 5, 10, ... 95
        cal_buckets[bucket]["total"] += 1
        cal_buckets[bucket]["prob_sum"] += home_prob
        if home_won:
            cal_buckets[bucket]["wins"] += 1

    avg_log_loss = total_log_loss / n
    avg_brier = total_brier / n
    accuracy = correct_picks / n

    # Build calibration curve
    calibration = []
    for bucket in sorted(cal_buckets.keys()):
        data = cal_buckets[bucket]
        if data["total"] >= 5:  # Min sample size
            avg_pred = data["prob_sum"] / data["total"]
            actual_rate = data["wins"] / data["total"]
            calibration.append({
                "predicted_pct": round(avg_pred * 100, 1),
                "actual_pct": round(actual_rate * 100, 1),
                "sample_size": data["total"],
                "deviation": round((actual_rate - avg_pred) * 100, 1),
            })

    # Accuracy by confidence
    confidence_bands = []
    for band in sorted(correct_by_confidence.keys()):
        data = correct_by_confidence[band]
        if data["total"] >= 10:
            rate = data["correct"] / data["total"]
            confidence_bands.append({
                "confidence_range": f"{band}-{band+5}%",
                "accuracy": round(rate * 100, 1),
                "sample_size": data["total"],
            })

    return {
        "n_games": n,
        "accuracy": round(accuracy * 100, 2),
        "log_loss": round(avg_log_loss, 4),
        "brier_score": round(avg_brier, 4),
        "calibration": calibration,
        "confidence_bands": confidence_bands,
        "correct_picks": correct_picks,
        "baseline_accuracy": 50.0,  # Coin flip
        "baseline_log_loss": 0.6931,  # -ln(0.5)
        "baseline_brier": 0.2500,
    }


# ─── PROFITABILITY ANALYSIS ─────────────────────────────────────────────────


def simulate_betting(
    predictions: list[dict],
    min_edge: float = 0.0,
    odds: int = -110,
    kelly_fraction: float = 0.5,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Simulate betting on all games meeting the edge threshold.

    Args:
        predictions: list of prediction dicts with home_win_prob, home_won
        min_edge: minimum edge (model_prob - implied_prob) to place bet
        odds: standard odds for all bets (e.g., -110)
        kelly_fraction: fraction of Kelly to bet (0.5 = half-Kelly)
        bankroll: starting bankroll

    Returns profitability metrics.
    """
    if odds < 0:
        implied_prob = abs(odds) / (abs(odds) + 100)
        decimal_odds = (100 / abs(odds)) + 1
    else:
        implied_prob = 100 / (odds + 100)
        decimal_odds = (odds / 100) + 1

    payout = decimal_odds - 1  # Net payout per $1

    bets = []
    current_bankroll = bankroll
    peak_bankroll = bankroll
    max_drawdown = 0

    for pred in predictions:
        home_prob = pred["home_win_prob"]
        away_prob = 1 - home_prob

        # Check both sides for edge
        home_edge = home_prob - implied_prob
        away_edge = away_prob - implied_prob

        bet_side = None
        bet_prob = 0

        if home_edge >= min_edge and home_edge >= away_edge:
            bet_side = "home"
            bet_prob = home_prob
        elif away_edge >= min_edge:
            bet_side = "away"
            bet_prob = away_prob

        if bet_side is None:
            continue

        # Kelly sizing
        b = payout
        p = bet_prob
        q = 1 - p
        kelly = max(0, (b * p - q) / b) * kelly_fraction

        bet_amount = current_bankroll * kelly
        if bet_amount < 1:  # Minimum $1 bet
            continue

        # Resolve
        won = (bet_side == "home" and pred["home_won"] == 1) or \
              (bet_side == "away" and pred["home_won"] == 0)

        profit = bet_amount * payout if won else -bet_amount
        current_bankroll += profit

        peak_bankroll = max(peak_bankroll, current_bankroll)
        drawdown = (peak_bankroll - current_bankroll) / peak_bankroll
        max_drawdown = max(max_drawdown, drawdown)

        bets.append({
            "bet_prob": round(bet_prob, 3),
            "edge": round(bet_prob - implied_prob, 3),
            "bet_amount": round(bet_amount, 2),
            "won": won,
            "profit": round(profit, 2),
            "bankroll": round(current_bankroll, 2),
        })

    if not bets:
        return {
            "min_edge": min_edge,
            "n_bets": 0,
            "message": "No bets met the edge threshold",
        }

    wins = sum(1 for b in bets if b["won"])
    total_wagered = sum(b["bet_amount"] for b in bets)
    total_profit = current_bankroll - bankroll

    return {
        "min_edge": round(min_edge * 100, 1),
        "n_bets": len(bets),
        "wins": wins,
        "losses": len(bets) - wins,
        "win_rate": round(wins / len(bets) * 100, 1),
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_wagered * 100, 2) if total_wagered > 0 else 0,
        "final_bankroll": round(current_bankroll, 2),
        "bankroll_growth": round((current_bankroll / bankroll - 1) * 100, 1),
        "max_drawdown": round(max_drawdown * 100, 1),
        "avg_edge": round(sum(b["edge"] for b in bets) / len(bets) * 100, 1),
        "break_even_rate": round(implied_prob * 100, 1),
    }


def simulate_betting_with_odds(
    predictions: list[dict],
    min_edge: float = 0.0,
    kelly_fraction: float = 0.5,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Simulate betting using actual per-game closing odds.

    Each prediction dict must have:
        - home_win_prob: model probability (0-1)
        - home_won: 1 if home won, 0 if away won
        - home_odds: American closing odds for home team (e.g., -150)
        - away_odds: American closing odds for away team (e.g., +130)

    Games without odds (home_odds=None) are skipped.
    Edge is computed against the no-vig implied probability.
    """
    bets = []
    current_bankroll = bankroll
    peak_bankroll = bankroll
    max_drawdown = 0
    skipped_no_odds = 0

    for pred in predictions:
        home_odds = pred.get("home_odds")
        away_odds = pred.get("away_odds")

        if home_odds is None or away_odds is None:
            skipped_no_odds += 1
            continue

        home_prob = pred["home_win_prob"]
        away_prob = 1 - home_prob

        # Compute no-vig implied probabilities
        home_implied = _american_to_implied(home_odds)
        away_implied = _american_to_implied(away_odds)
        total_implied = home_implied + away_implied
        home_no_vig = home_implied / total_implied
        away_no_vig = away_implied / total_implied

        # Edge against no-vig line
        home_edge = home_prob - home_no_vig
        away_edge = away_prob - away_no_vig

        bet_side = None
        bet_prob = 0.0
        bet_odds = 0

        if home_edge >= min_edge and home_edge >= away_edge:
            bet_side = "home"
            bet_prob = home_prob
            bet_odds = home_odds
        elif away_edge >= min_edge:
            bet_side = "away"
            bet_prob = away_prob
            bet_odds = away_odds

        if bet_side is None:
            continue

        # Decimal odds for payout calculation
        if bet_odds < 0:
            decimal_odds = (100 / abs(bet_odds)) + 1
        else:
            decimal_odds = (bet_odds / 100) + 1
        payout = decimal_odds - 1

        # Kelly sizing against actual odds
        b = payout
        p = bet_prob
        q = 1 - p
        kelly = max(0, (b * p - q) / b) * kelly_fraction

        # Cap bet at 2% of bankroll to prevent runaway compounding
        kelly = min(kelly, 0.02)
        bet_amount = current_bankroll * kelly
        if bet_amount < 1:
            continue

        won = (bet_side == "home" and pred["home_won"] == 1) or \
              (bet_side == "away" and pred["home_won"] == 0)

        profit = bet_amount * payout if won else -bet_amount
        current_bankroll += profit

        peak_bankroll = max(peak_bankroll, current_bankroll)
        drawdown = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

        implied = _american_to_implied(bet_odds)
        bets.append({
            "bet_side": bet_side,
            "bet_prob": round(bet_prob, 3),
            "bet_odds": bet_odds,
            "implied_prob": round(implied, 3),
            "edge": round(bet_prob - implied, 3),
            "bet_amount": round(bet_amount, 2),
            "won": won,
            "profit": round(profit, 2),
            "bankroll": round(current_bankroll, 2),
        })

    if not bets:
        return {
            "min_edge": round(min_edge * 100, 1),
            "n_bets": 0,
            "skipped_no_odds": skipped_no_odds,
            "message": "No bets met the edge threshold",
        }

    wins = sum(1 for b in bets if b["won"])
    total_wagered = sum(b["bet_amount"] for b in bets)
    total_profit = current_bankroll - bankroll

    fav_bets = [b for b in bets if b["bet_odds"] < 0]
    dog_bets = [b for b in bets if b["bet_odds"] > 0]

    return {
        "min_edge": round(min_edge * 100, 1),
        "n_bets": len(bets),
        "skipped_no_odds": skipped_no_odds,
        "wins": wins,
        "losses": len(bets) - wins,
        "win_rate": round(wins / len(bets) * 100, 1),
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_wagered * 100, 2) if total_wagered > 0 else 0,
        "final_bankroll": round(current_bankroll, 2),
        "bankroll_growth": round((current_bankroll / bankroll - 1) * 100, 1),
        "max_drawdown": round(max_drawdown * 100, 1),
        "avg_edge": round(sum(b["edge"] for b in bets) / len(bets) * 100, 1),
        "avg_odds": round(sum(b["bet_odds"] for b in bets) / len(bets)),
        "fav_bets": len(fav_bets),
        "dog_bets": len(dog_bets),
        "bets": bets,
    }


def _american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def simulate_flat_betting_with_odds(
    predictions: list[dict],
    min_edge: float = 0.0,
    bet_size: float = 100.0,
) -> dict[str, Any]:
    """Simulate flat-bet strategy using actual per-game closing odds.

    Simpler than Kelly — bet a fixed amount on every game meeting the edge threshold.
    """
    bets = []
    total_wagered = 0.0
    total_profit = 0.0

    for pred in predictions:
        home_odds = pred.get("home_odds")
        away_odds = pred.get("away_odds")
        if home_odds is None or away_odds is None:
            continue

        home_prob = pred["home_win_prob"]
        away_prob = 1 - home_prob

        # No-vig implied
        hi = _american_to_implied(home_odds)
        ai = _american_to_implied(away_odds)
        total = hi + ai
        home_nv = hi / total
        away_nv = ai / total

        home_edge = home_prob - home_nv
        away_edge = away_prob - away_nv

        bet_side = None
        bet_odds = 0

        if home_edge >= min_edge and home_edge >= away_edge:
            bet_side = "home"
            bet_odds = home_odds
        elif away_edge >= min_edge:
            bet_side = "away"
            bet_odds = away_odds

        if bet_side is None:
            continue

        # Resolve at actual odds
        if bet_odds < 0:
            win_payout = bet_size * (100 / abs(bet_odds))
        else:
            win_payout = bet_size * (bet_odds / 100)

        won = (bet_side == "home" and pred["home_won"] == 1) or \
              (bet_side == "away" and pred["home_won"] == 0)

        profit = win_payout if won else -bet_size
        total_wagered += bet_size
        total_profit += profit

        bets.append({
            "date": pred.get("date", ""),
            "home_team": pred.get("home_team", ""),
            "away_team": pred.get("away_team", ""),
            "bet_side": bet_side,
            "bet_odds": bet_odds,
            "won": won,
            "profit": round(profit, 2),
        })

    if not bets:
        return {"n_bets": 0, "message": "No bets"}

    wins = sum(1 for b in bets if b["won"])

    return {
        "min_edge": round(min_edge * 100, 1),
        "n_bets": len(bets),
        "wins": wins,
        "losses": len(bets) - wins,
        "win_rate": round(wins / len(bets) * 100, 1),
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_wagered * 100, 2) if total_wagered > 0 else 0,
        "units": round(total_profit / bet_size, 1),
    }


def run_edge_sweep(
    predictions: list[dict],
    edge_thresholds: list[float] | None = None,
) -> list[dict]:
    """Run profitability analysis at multiple edge thresholds."""
    if edge_thresholds is None:
        edge_thresholds = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

    results = []
    for threshold in edge_thresholds:
        result = simulate_betting(predictions, min_edge=threshold)
        results.append(result)

    return results


def run_edge_sweep_with_odds(
    predictions: list[dict],
    edge_thresholds: list[float] | None = None,
) -> list[dict]:
    """Run profitability analysis at multiple edge thresholds using actual odds."""
    if edge_thresholds is None:
        edge_thresholds = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

    results = []
    for threshold in edge_thresholds:
        result = simulate_betting_with_odds(predictions, min_edge=threshold)
        # Strip individual bets for summary
        summary = {k: v for k, v in result.items() if k != "bets"}
        results.append(summary)

    return results


# ─── DISPLAY ────────────────────────────────────────────────────────────────


def print_backtest_report(
    metrics: dict,
    edge_sweep: list[dict],
    season: str = "2025",
) -> None:
    """Print comprehensive backtest report."""
    print(f"\n{'=' * 80}")
    print(f"  ⚾ BACKTEST REPORT — {season} MLB Season")
    print(f"{'=' * 80}\n")

    # Core metrics
    print(f"  CORE METRICS ({metrics['n_games']} games)")
    print(f"  {'─' * 50}")
    print(f"  {'Metric':<25} {'Model':>10} {'Baseline':>10} {'Delta':>10}")
    print(f"  {'─' * 50}")

    acc_delta = metrics["accuracy"] - metrics["baseline_accuracy"]
    ll_delta = metrics["log_loss"] - metrics["baseline_log_loss"]
    bs_delta = metrics["brier_score"] - metrics["baseline_brier"]

    print(f"  {'Accuracy':<25} {metrics['accuracy']:>9.1f}% {metrics['baseline_accuracy']:>9.1f}% {acc_delta:>+9.1f}%")
    print(f"  {'Log-Loss':<25} {metrics['log_loss']:>10.4f} {metrics['baseline_log_loss']:>10.4f} {ll_delta:>+10.4f}")
    print(f"  {'Brier Score':<25} {metrics['brier_score']:>10.4f} {metrics['baseline_brier']:>10.4f} {bs_delta:>+10.4f}")

    # Calibration
    print(f"\n  CALIBRATION CURVE")
    print(f"  {'─' * 55}")
    print(f"  {'Predicted':>10} {'Actual':>10} {'Sample':>8} {'Deviation':>10}")
    print(f"  {'─' * 55}")
    for cal in metrics.get("calibration", []):
        dev_str = f"{cal['deviation']:+.1f}%"
        bar = "█" * max(0, int(abs(cal["deviation"]) / 2))
        direction = "→" if abs(cal["deviation"]) < 2 else ("↑" if cal["deviation"] > 0 else "↓")
        print(f"  {cal['predicted_pct']:>9.1f}% {cal['actual_pct']:>9.1f}% {cal['sample_size']:>7} {dev_str:>9} {direction} {bar}")

    # Confidence bands
    print(f"\n  ACCURACY BY CONFIDENCE")
    print(f"  {'─' * 40}")
    for band in metrics.get("confidence_bands", []):
        bar = "█" * int(band["accuracy"] / 2)
        print(f"  {band['confidence_range']:>10}  {band['accuracy']:>5.1f}%  n={band['sample_size']:<5} {bar}")

    # Edge sweep (profitability)
    print(f"\n  PROFITABILITY BY MINIMUM EDGE (at -110 juice, half-Kelly)")
    print(f"  {'─' * 75}")
    print(f"  {'Min Edge':>9} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Profit':>10} {'Final $':>10} {'MaxDD':>7}")
    print(f"  {'─' * 75}")

    for r in edge_sweep:
        if r.get("n_bets", 0) == 0:
            print(f"  {r['min_edge']:>8.1f}% {'—':>6} {'—':>7} {'—':>8} {'—':>10} {'—':>10} {'—':>7}")
            continue

        roi_marker = "✓" if r["roi"] > 0 else "✗"
        print(f"  {r['min_edge']:>8.1f}% {r['n_bets']:>6} {r['win_rate']:>6.1f}% "
              f"{r['roi']:>+7.1f}% {r['total_profit']:>+9.2f} {r['final_bankroll']:>9.2f} "
              f"{r['max_drawdown']:>6.1f}% {roi_marker}")

    # Key finding
    print(f"\n  {'=' * 75}")
    profitable = [r for r in edge_sweep if r.get("roi", 0) > 0 and r.get("n_bets", 0) >= 50]
    if profitable:
        best = max(profitable, key=lambda r: r["roi"])
        print(f"  BEST EDGE THRESHOLD: ≥{best['min_edge']:.1f}%")
        print(f"    → {best['n_bets']} bets, {best['win_rate']:.1f}% win rate, "
              f"{best['roi']:+.1f}% ROI, ${best['total_profit']:+.2f} profit")
        print(f"    → Max drawdown: {best['max_drawdown']:.1f}%")
    else:
        best_attempt = min(edge_sweep, key=lambda r: abs(r.get("roi", -100)))
        print(f"  ⚠ No reliably profitable edge threshold found with n≥50 bets")
        print(f"    Closest: {best_attempt.get('min_edge', 0):.1f}% edge → "
              f"{best_attempt.get('roi', 0):+.1f}% ROI on {best_attempt.get('n_bets', 0)} bets")
        print(f"    Consider: recalibrating model weights or increasing edge threshold")

    print(f"  {'=' * 75}\n")


def _compact_money(amount: float) -> str:
    """Format large dollar amounts compactly."""
    if abs(amount) >= 1_000_000_000:
        return f"${amount/1_000_000_000:,.1f}B"
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:,.1f}M"
    if abs(amount) >= 10_000:
        return f"${amount/1_000:,.1f}K"
    return f"${amount:,.2f}"


def print_odds_backtest_report(
    metrics: dict,
    edge_sweep: list[dict],
    flat_sweep: list[dict],
    season: str = "2025",
    odds_coverage: float = 0.0,
) -> None:
    """Print backtest report using actual closing odds."""
    print(f"\n{'=' * 80}")
    print(f"  ⚾ BACKTEST vs CLOSING ODDS — {season}")
    print(f"{'=' * 80}\n")

    # Core metrics (same as standard report)
    print(f"  CORE METRICS ({metrics['n_games']} games, {odds_coverage:.0f}% with closing odds)")
    print(f"  {'─' * 50}")
    print(f"  {'Metric':<25} {'Model':>10} {'Baseline':>10} {'Delta':>10}")
    print(f"  {'─' * 50}")

    acc_delta = metrics["accuracy"] - metrics["baseline_accuracy"]
    ll_delta = metrics["log_loss"] - metrics["baseline_log_loss"]
    bs_delta = metrics["brier_score"] - metrics["baseline_brier"]

    print(f"  {'Accuracy':<25} {metrics['accuracy']:>9.1f}% {metrics['baseline_accuracy']:>9.1f}% {acc_delta:>+9.1f}%")
    print(f"  {'Log-Loss':<25} {metrics['log_loss']:>10.4f} {metrics['baseline_log_loss']:>10.4f} {ll_delta:>+10.4f}")
    print(f"  {'Brier Score':<25} {metrics['brier_score']:>10.4f} {metrics['baseline_brier']:>10.4f} {bs_delta:>+10.4f}")

    # Calibration
    print(f"\n  CALIBRATION CURVE")
    print(f"  {'─' * 55}")
    print(f"  {'Predicted':>10} {'Actual':>10} {'Sample':>8} {'Deviation':>10}")
    print(f"  {'─' * 55}")
    for cal in metrics.get("calibration", []):
        dev_str = f"{cal['deviation']:+.1f}%"
        direction = "→" if abs(cal["deviation"]) < 2 else ("↑" if cal["deviation"] > 0 else "↓")
        bar = "█" * max(0, int(abs(cal["deviation"]) / 2))
        print(f"  {cal['predicted_pct']:>9.1f}% {cal['actual_pct']:>9.1f}% {cal['sample_size']:>7} {dev_str:>9} {direction} {bar}")

    # Kelly betting with actual odds
    print(f"\n  KELLY BETTING vs ACTUAL CLOSING ODDS (half-Kelly, 2% max bet)")
    print(f"  {'─' * 80}")
    print(f"  {'Min Edge':>9} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Final $':>12} {'Growth':>9} {'MaxDD':>7} {'Fav/Dog':>9}")
    print(f"  {'─' * 80}")

    for r in edge_sweep:
        if r.get("n_bets", 0) == 0:
            print(f"  {r.get('min_edge', 0):>8.1f}% {'—':>6} {'—':>7} {'—':>8} {'—':>12} {'—':>9} {'—':>7} {'—':>9}")
            continue

        roi_marker = "✓" if r["roi"] > 0 else "✗"
        fav_dog = f"{r.get('fav_bets', 0)}/{r.get('dog_bets', 0)}"
        final = r['final_bankroll']
        final_str = _compact_money(final)
        growth = r.get('bankroll_growth', 0)
        growth_str = f"{growth:+.0f}%" if abs(growth) < 10000 else f"{growth/100:+.0f}x"
        print(f"  {r['min_edge']:>8.1f}% {r['n_bets']:>6} {r['win_rate']:>6.1f}% "
              f"{r['roi']:>+7.1f}% {final_str:>12} {growth_str:>9} "
              f"{r['max_drawdown']:>6.1f}% {fav_dog:>9} {roi_marker}")

    # Flat betting with actual odds
    print(f"\n  FLAT BET $100 vs ACTUAL CLOSING ODDS")
    print(f"  {'─' * 65}")
    print(f"  {'Min Edge':>9} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Units':>8} {'Profit':>10}")
    print(f"  {'─' * 65}")

    for r in flat_sweep:
        if r.get("n_bets", 0) == 0:
            continue
        roi_marker = "✓" if r["roi"] > 0 else "✗"
        print(f"  ≥{r['min_edge']:>5.1f}% {r['n_bets']:>6} {r['win_rate']:>6.1f}% "
              f"{r['roi']:>+7.2f}% {r.get('units', 0):>+7.1f}u "
              f"{r['total_profit']:>+9.2f} {roi_marker}")

    print(f"  {'─' * 65}")

    # Key findings
    print(f"\n  {'=' * 80}")
    profitable_kelly = [r for r in edge_sweep if r.get("roi", 0) > 0 and r.get("n_bets", 0) >= 50]
    profitable_flat = [r for r in flat_sweep if r.get("roi", 0) > 0 and r.get("n_bets", 0) >= 50]

    if profitable_kelly:
        best = max(profitable_kelly, key=lambda r: r["roi"])
        print(f"  BEST KELLY THRESHOLD: ≥{best['min_edge']:.1f}% edge")
        print(f"    → {best['n_bets']} bets, {best['win_rate']:.1f}% win, "
              f"{best['roi']:+.1f}% ROI, {_compact_money(best['final_bankroll'])} final")

    if profitable_flat:
        best = max(profitable_flat, key=lambda r: r["roi"])
        print(f"  BEST FLAT-BET THRESHOLD: ≥{best['min_edge']:.1f}% edge")
        print(f"    → {best['n_bets']} bets, {best['win_rate']:.1f}% win, "
              f"{best['roi']:+.1f}% ROI, {best.get('units', 0):+.1f}u")

    if not profitable_kelly and not profitable_flat:
        print(f"  ⚠ No reliably profitable threshold found with n≥50 bets")
        print(f"    The model may need recalibration against market lines")

    print(f"  {'=' * 80}\n")
