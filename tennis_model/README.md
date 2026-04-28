# tennis_model — ATP + WTA pre-match betting model

A walk-forward-validated tennis prediction framework that:

- Pulls real bookmaker pre-match odds from [tennis-data.co.uk](http://www.tennis-data.co.uk) (2000–2026)
- Enriches with [Jeff Sackmann's](https://github.com/JeffSackmann) match metadata (serve/return splits, ages, heights, handedness)
- Maintains a strictly-causal Elo state (overall + per-surface, blended)
- Builds rolling form, rest, head-to-head, and career-surface features without any future leakage
- Trains an XGBoost classifier with isotonic calibration in a walk-forward loop (retrain every 3 months on a 4-year window by default)
- Selects bets at a configurable edge threshold and reports flat-stake yield, fractional-Kelly bankroll, drawdown, and per-tour/surface/year breakdowns
- Persists the final model + feature state for live `predict_match()` use

## Honest expectations

**No model in this repository is guaranteed to be profitable in live play.** Closing odds at sharp books (Pinnacle in particular) are highly efficient — beating them sustainably is the rare exception, not the rule. What this codebase delivers is a *correct* backtesting harness. Whether the resulting yield is positive, near zero, or negative when you run it depends on:

- which odds source you select (`PSW_PSL` is sharpest and hardest to beat; `B365W_B365L` is softer)
- the years you train and test on
- the edge threshold you require
- whether the soft-book prices you'd actually get in live play match what's recorded historically

If you see a backtest yield above ~10% on a few hundred bets, treat it as overfitting until proven otherwise. A 2–5% flat yield across thousands of bets and multiple years is the realistic professional bar.

## Installation

```bash
pip install -e .                     # from repo root, uses pyproject.toml
# or
pip install -r tennis_model/requirements.txt
```

## Run the full pipeline

```bash
python -m tennis_model.run --config tennis_model/config.yaml
```

This will:

1. Download and cache yearly Excel/CSV files (one-time cost; subsequent runs read from `tennis_model/data/cache/`).
2. Build features in chronological order with the running Elo / form / H2H state.
3. Run walk-forward training (default: 4-year train window, 3-month step).
4. Backtest every out-of-sample prediction at the configured edge threshold.
5. Print headline metrics + per-tour / per-surface / per-year breakdowns.
6. Save:
   - `tennis_model/artifacts/model.joblib` — final model + feature state for live use
   - `tennis_model/artifacts/backtest_bets.csv` — per-bet ledger
   - `tennis_model/artifacts/metrics.json` — full metrics dict
   - `tennis_model/artifacts/folds.csv` — per-fold log-loss / Brier vs market
   - `tennis_model/artifacts/run.log` — pipeline log

## Live prediction

Build a CSV of upcoming matches with these columns:

```
date, tour, tournament, surface, level, round, best_of, player1, player2,
p1_rank, p2_rank, p1_pts, p2_pts, p1_odds, p2_odds
```

Optional columns: `p1_age, p2_age, p1_ht, p2_ht, p1_hand, p2_hand`.

Player names should be in tennis-data.co.uk style (`federer r.`, `nadal r.`) so they match the Elo/form state. Then:

```bash
python -m tennis_model.run --action predict --upcoming upcoming_matches.csv --edge 0.04
```

The output CSV adds: `model_p1, model_p2, market_p1, market_p2, edge_p1, edge_p2, recommended_side, recommended_edge, kelly_full`.

## Tuning knobs

All in `tennis_model/config.yaml`:

| Section | Key | Effect |
|---|---|---|
| `elo` | `surface_blend` | Weight on surface Elo (0.65 default); raise on clay/grass-heavy seasons |
| `model` | `train_window_years` | Larger windows = more stable but slower to react to regime change |
| `model` | `step_months` | Smaller = more frequent retrains but more compute |
| `backtest` | `edge_threshold` | Higher = fewer, sharper bets; lower = more bets, lower yield |
| `backtest` | `odds_source` | `PSW_PSL` (Pinnacle, sharpest), `AvgW_AvgL` (average), `B365W_B365L` (soft) |
| `backtest` | `kelly_fraction` | Fractional Kelly multiplier (0.25 = quarter-Kelly) |

## Architecture

```
tennis_model/
  config.yaml          # all tunable knobs
  data_ingestion.py    # download tennis-data.co.uk + Sackmann, normalize, merge
  features.py          # FeatureState (causal Elo/form/H2H) + build_features()
  model.py             # encode_features() + walk_forward_train()
  backtest.py          # select_bets() + simulate() with flat + Kelly
  run.py               # orchestrator + predict_match() entry point
  artifacts/           # model.joblib, metrics.json, bets.csv, folds.csv, log
  data/cache/          # cached Excel/CSV downloads
```

## Anti-leakage guarantees

- Features for match `i` are computed from `FeatureState` *before* the row's outcome is applied. The state is updated *after* the feature row is emitted.
- Walk-forward training uses a strictly past training window: a fold predicting matches in `[t, t+step)` is trained only on `date < t`.
- Calibration uses the last 10% of the training window (still in the past relative to the test fold).
- `market_p1` (the bookmaker's implied probability) is **not** fed to the model. Including it would let the model trivially learn to copy the market and any backtest "edge" would just be the dev-vig difference between odds sources.

## Reading the output

Headline numbers to focus on:

- **Yield (flat)** — profit ÷ total stakes. Stable estimator, doesn't compound.
- **Per-year yield** — does positive yield persist across regimes, or is it concentrated in one year?
- **Per-tour / per-surface yield** — is the edge real, or all from one slice?
- **Approx Sharpe** — risk-adjusted; numbers above 2 with thousands of bets are good, above 5 are suspicious.
- **Calibration table** — model probability vs actual win rate. If the model says 60% and actuals are 55%, calibration is off and edges are inflated.

## Things this framework deliberately does not do

- It does not bet in-play. Pre-match only.
- It does not model player injuries, news, or court schedules. Add these features yourself if you have the data.
- It does not auto-stake into a book. Output is bet recommendations; execution and bankroll management are your problem.
- It does not iterate the model "until profitable." Doing so on a fixed dataset is overfitting. If you want to extend it, the right move is: hold out the last 12 months entirely, iterate on data prior to that, then evaluate the held-out year exactly once.
