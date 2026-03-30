"""Poisson-based MLB score prediction model.

Uses expected runs (lambda) to generate:
- Most likely final scores
- Win probabilities
- Over/under estimates
- Score distributions
"""

import numpy as np
from scipy.stats import poisson


def predict_score_distribution(
    away_lambda: float, home_lambda: float, max_runs: int = 16
) -> np.ndarray:
    """Generate a 2D probability matrix of (away_runs, home_runs) outcomes.

    Uses independent Poisson distributions for each team's run scoring.
    Returns a (max_runs+1) x (max_runs+1) matrix where entry [i][j] is
    P(away=i, home=j).
    """
    away_probs = poisson.pmf(np.arange(max_runs + 1), away_lambda)
    home_probs = poisson.pmf(np.arange(max_runs + 1), home_lambda)

    # Outer product gives joint probability (assuming independence)
    return np.outer(away_probs, home_probs)


def most_likely_score(score_matrix: np.ndarray) -> tuple[int, int, float]:
    """Find the single most likely final score from the distribution."""
    idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    return int(idx[0]), int(idx[1]), float(score_matrix[idx])


def top_n_scores(score_matrix: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
    """Return the top N most likely scores."""
    flat = score_matrix.flatten()
    top_indices = np.argsort(flat)[::-1][:n]
    results = []
    for idx in top_indices:
        away, home = np.unravel_index(idx, score_matrix.shape)
        results.append((int(away), int(home), float(flat[idx])))
    return results


def win_probability(score_matrix: np.ndarray) -> dict[str, float]:
    """Calculate win probability for away team, home team, and tie/extras."""
    max_runs = score_matrix.shape[0]
    away_win = 0.0
    home_win = 0.0
    tie = 0.0

    for i in range(max_runs):
        for j in range(max_runs):
            p = score_matrix[i][j]
            if i > j:
                away_win += p
            elif j > i:
                home_win += p
            else:
                tie += p

    # Distribute tie probability (extra innings) — slight home advantage
    home_extras_edge = 0.52
    home_win += tie * home_extras_edge
    away_win += tie * (1 - home_extras_edge)

    return {
        "away_win": round(away_win, 4),
        "home_win": round(home_win, 4),
    }


def over_under_probability(
    score_matrix: np.ndarray, total_line: float
) -> dict[str, float]:
    """Calculate probability of over/under a given total runs line."""
    max_runs = score_matrix.shape[0]
    over = 0.0
    under = 0.0

    for i in range(max_runs):
        for j in range(max_runs):
            total = i + j
            p = score_matrix[i][j]
            if total > total_line:
                over += p
            elif total < total_line:
                under += p
            # Exact = push, split between over/under
            else:
                over += p * 0.5
                under += p * 0.5

    return {"over": round(over, 4), "under": round(under, 4)}


def expected_total(away_lambda: float, home_lambda: float) -> float:
    """Simple expected total runs."""
    return round(away_lambda + home_lambda, 2)


def generate_prediction(
    away_lambda: float,
    home_lambda: float,
    away_team: str,
    home_team: str,
) -> dict:
    """Generate a complete prediction for a single game."""
    matrix = predict_score_distribution(away_lambda, home_lambda)
    away_score, home_score, prob = most_likely_score(matrix)
    top_scores = top_n_scores(matrix, n=5)
    win_probs = win_probability(matrix)
    total = expected_total(away_lambda, home_lambda)
    ou_probs = over_under_probability(matrix, round(total * 2) / 2)  # nearest 0.5

    return {
        "away_team": away_team,
        "home_team": home_team,
        "away_expected_runs": round(away_lambda, 2),
        "home_expected_runs": round(home_lambda, 2),
        "predicted_score": {"away": away_score, "home": home_score},
        "score_probability": round(prob * 100, 2),
        "top_5_scores": [
            {"away": a, "home": h, "probability": round(p * 100, 2)}
            for a, h, p in top_scores
        ],
        "win_probability": {
            "away": round(win_probs["away_win"] * 100, 1),
            "home": round(win_probs["home_win"] * 100, 1),
        },
        "expected_total": total,
        "over_under": {
            "line": round(total * 2) / 2,
            "over_pct": round(ou_probs["over"] * 100, 1),
            "under_pct": round(ou_probs["under"] * 100, 1),
        },
    }
