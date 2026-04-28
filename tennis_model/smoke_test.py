"""End-to-end smoke test on synthetic data, no network required.

Generates a small fake tennis dataset, then runs:
  - matches_to_p1p2
  - build_features
  - walk_forward_train (with smaller windows)
  - select_bets + simulate

This catches schema/logic bugs without paying the multi-minute download cost.
Run: python -m tennis_model.smoke_test
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from tennis_model.backtest import BacktestConfig, select_bets, simulate
from tennis_model.data_ingestion import matches_to_p1p2
from tennis_model.features import FeatureConfig, build_features
from tennis_model.model import ModelConfig, walk_forward_train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def synth_dataset(n_players: int = 60, n_matches: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthesize a tennis-data-style winner/loser frame.

    True skills are drawn N(0,1); win probability follows logistic on skill diff
    plus a small surface affinity. Odds embed the true probability + a 5%
    overround and small noise (so the bookmaker is roughly efficient).
    """
    rng = np.random.default_rng(seed)
    skills = rng.normal(size=n_players)
    surface_affinity = rng.normal(scale=0.5, size=(n_players, 4))  # Hard, Clay, Grass, Carpet
    surfaces = ["Hard", "Clay", "Grass", "Carpet"]
    surface_idx = {s: i for i, s in enumerate(surfaces)}

    dates = pd.date_range("2014-01-01", periods=n_matches, freq="D")
    rows = []
    for i in range(n_matches):
        a, b = rng.choice(n_players, size=2, replace=False)
        surface = rng.choice(surfaces, p=[0.55, 0.30, 0.10, 0.05])
        diff = (skills[a] + surface_affinity[a, surface_idx[surface]]) - \
               (skills[b] + surface_affinity[b, surface_idx[surface]])
        p_a = 1.0 / (1 + np.exp(-1.2 * diff))
        winner_idx = a if rng.random() < p_a else b
        loser_idx = b if winner_idx == a else a
        # Bookmaker odds: noisy version of true probability + 5% overround.
        true_p_w = p_a if winner_idx == a else 1 - p_a
        noisy = np.clip(true_p_w + rng.normal(scale=0.04), 0.05, 0.95)
        # Add overround: inv_w + inv_l = 1.05
        w_odds = 1 / (noisy * 1.05)
        l_odds = 1 / ((1 - noisy) * 1.05)
        rows.append({
            "date": dates[i],
            "tour": rng.choice(["ATP", "WTA"], p=[0.55, 0.45]),
            "tournament": "Test Open",
            "surface": surface,
            "level": rng.choice(["Grand Slam", "Masters 1000", "ATP 250"], p=[0.05, 0.20, 0.75]),
            "round": rng.choice(["1st Round", "2nd Round", "Quarterfinals", "Final"], p=[0.5, 0.3, 0.15, 0.05]),
            "best_of": 3,
            "winner": f"player_{winner_idx:03d}",
            "loser": f"player_{loser_idx:03d}",
            "w_rank": int(rng.integers(1, 200)),
            "l_rank": int(rng.integers(1, 200)),
            "w_pts": int(rng.integers(100, 9000)),
            "l_pts": int(rng.integers(100, 9000)),
            "w_odds": max(w_odds, 1.05),
            "l_odds": max(l_odds, 1.05),
            "odds_source": "PSW_PSL",
            "comment": "Completed",
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main():
    print("Building synthetic dataset...")
    full = synth_dataset()
    print(f"  {len(full)} matches across {full['date'].min().date()}..{full['date'].max().date()}")

    p1p2 = matches_to_p1p2(full)
    print("Building features...")
    feats, state = build_features(p1p2, full, FeatureConfig())
    print(f"  features: {feats.shape}")
    assert feats["y"].isin([0, 1]).all(), "label sanity"
    # Symmetry check: market_p1 + market_p2 should equal 1 (devigged).
    # We only carry market_p1 in features; the column name carries it.

    print("Walk-forward training...")
    cfg = ModelConfig(train_window_years=1, step_months=3, min_train_rows=300,
                      xgb_params={"objective": "binary:logistic", "max_depth": 3,
                                  "learning_rate": 0.1, "n_estimators": 100,
                                  "tree_method": "hist", "eval_metric": "logloss"},
                      calibration="isotonic")
    oof, folds, art = walk_forward_train(feats, cfg)
    print(f"  {len(folds)} folds, {len(oof)} OOF predictions")
    assert len(oof) > 0, "no OOF predictions"
    assert "pred" in oof.columns and oof["pred"].between(0, 1).all()

    print("\nFold metrics:")
    for f in folds[:5]:
        print(f"  fold {f.fold_id}: model_LL={f.log_loss:.4f} market_LL={f.market_log_loss:.4f} acc={f.accuracy:.3f}")
    if len(folds) > 5:
        print(f"  ... ({len(folds)-5} more)")

    print("\nBacktesting at edge_threshold=0.04...")
    bt = BacktestConfig(edge_threshold=0.04)
    bets = select_bets(oof, bt)
    bets, metrics = simulate(bets, bt)
    print(f"  bets selected: {metrics.get('n_bets', 0)}")
    if metrics.get("n_bets", 0) > 0:
        print(f"  yield (flat): {metrics['yield_pct']:+.2f}%  win_rate: {metrics['win_rate']*100:.1f}%")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
