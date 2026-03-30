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
