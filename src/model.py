"""Negative Binomial MLB score prediction model.

Replaces Poisson with Negative Binomial distribution which better
captures baseball's overdispersion (variance > mean in run scoring).

Baseball run distributions have heavier tails than Poisson predicts:
- Multi-run innings (rallies, grand slams) create positive skew
- Bullpen meltdowns create blowout risk Poisson underestimates
- The NB's extra shape parameter (r) controls this overdispersion

The NB is parameterized by:
- mu (mean) = expected runs (same as Poisson lambda)
- r (shape) = controls overdispersion. Lower r = more variance.
  r → infinity recovers Poisson. For MLB, r ≈ 4-6 fits best.

Also retains Poisson mode for backward compatibility.
"""

import numpy as np
from scipy.stats import poisson, nbinom


# MLB-calibrated overdispersion parameter
# Fitted from historical run distributions: variance/mean ratio ≈ 1.15-1.25
# r = mu^2 / (variance - mu)
# For typical MLB: ~4.5 R/G with var ~5.5-6.0 → r ≈ 10-15
# r=5.5 over-compresses favorites (61% when should be 64%)
# r=12 matches market pricing more closely while still capturing
# the heavier tails that Poisson misses (blowouts, multi-run innings)
NB_SHAPE_R = 12.0


def _nb_params(mu: float, r: float = NB_SHAPE_R) -> tuple[float, float]:
    """Convert mean (mu) and shape (r) to scipy nbinom parameters.

    scipy.stats.nbinom uses (n, p) parameterization where:
    - n = r (number of successes)
    - p = r / (r + mu) (probability of success)
    """
    p = r / (r + mu) if (r + mu) > 0 else 0.5
    return r, p


def predict_score_distribution(
    away_lambda: float, home_lambda: float, max_runs: int = 18,
    use_negative_binomial: bool = True,
) -> np.ndarray:
    """Generate a 2D probability matrix of (away_runs, home_runs) outcomes.

    Uses Negative Binomial by default for better tail modeling.
    Falls back to Poisson if use_negative_binomial=False.
    """
    k = np.arange(max_runs + 1)

    if use_negative_binomial:
        r_a, p_a = _nb_params(away_lambda)
        r_h, p_h = _nb_params(home_lambda)
        away_probs = nbinom.pmf(k, r_a, p_a)
        home_probs = nbinom.pmf(k, r_h, p_h)
    else:
        away_probs = poisson.pmf(k, away_lambda)
        home_probs = poisson.pmf(k, home_lambda)

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
    """Generate a complete prediction for a single game using Negative Binomial."""
    matrix = predict_score_distribution(away_lambda, home_lambda)
    away_score, home_score, prob = most_likely_score(matrix)
    top_scores = top_n_scores(matrix, n=5)
    win_probs = win_probability(matrix)
    total = expected_total(away_lambda, home_lambda)
    ou_probs = over_under_probability(matrix, round(total * 2) / 2)

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
        "model": "negative_binomial",
    }
