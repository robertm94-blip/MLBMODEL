"""End-to-end pipeline + live prediction entry point.

Usage:
    python -m tennis_model.run --config tennis_model/config.yaml
    python -m tennis_model.run --action predict --upcoming upcoming_matches.csv

The script does:
  1. Load config.
  2. Download data (cached on disk) and merge sources.
  3. Build features in chronological order.
  4. Run walk-forward training and produce OOF predictions.
  5. Run the backtest with the configured edge threshold.
  6. Persist the final model + feature state for live inference.
  7. Print a metrics table.

For live inference, point `--upcoming` at a CSV with columns
(date, tour, tournament, surface, level, round, best_of, player1, player2,
 p1_rank, p2_rank, p1_pts, p2_pts, p1_odds, p2_odds). Optional Sackmann
 columns (p1_age, p2_age, p1_ht, p2_ht, p1_hand, p2_hand) are honored.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from tennis_model.backtest import (
    BacktestConfig,
    calibration_table,
    select_bets,
    simulate,
)
from tennis_model.data_ingestion import IngestionConfig, load_all, matches_to_p1p2
from tennis_model.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FeatureConfig,
    FeatureState,
    _devig_two_way,
    _level_bucket,
    _round_bucket,
    build_features,
)
from tennis_model.model import (
    ModelConfig,
    encode_features,
    summarize_folds,
    walk_forward_train,
)


def _setup_logging(level: str, file: str | None):
    Path(file).parent.mkdir(parents=True, exist_ok=True) if file else None
    handlers = [logging.StreamHandler()]
    if file:
        handlers.append(logging.FileHandler(file))
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers, force=True,
    )


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ingestion_cfg(c: dict) -> IngestionConfig:
    d = c["data"]
    return IngestionConfig(
        td_atp_url_template=d["td_atp_url_template"],
        td_wta_url_template=d["td_wta_url_template"],
        sackmann_atp_matches=d["sackmann_atp_matches"],
        sackmann_wta_matches=d["sackmann_wta_matches"],
        sackmann_atp_players=d["sackmann_atp_players"],
        sackmann_wta_players=d["sackmann_wta_players"],
        cache_dir=d["cache_dir"],
        start_year=int(d["start_year"]),
        end_year=int(d["end_year"]),
        tours=d["tours"],
        fallback_odds=c["backtest"]["fallback_odds"],
    )


def _feature_cfg(c: dict) -> FeatureConfig:
    e = c["elo"]
    f = c["features"]
    return FeatureConfig(
        k_initial=e["k_initial"], k_min=e["k_min"], k_decay=e["k_decay"],
        surface_blend=e["surface_blend"], prior_rating=e["prior_rating"],
        recent_form_windows=tuple(f["recent_form_windows"]),
        rest_cap_days=f["rest_cap_days"],
        min_career_matches=f["min_career_matches"],
    )


def _model_cfg(c: dict) -> ModelConfig:
    m = c["model"]
    return ModelConfig(
        train_window_years=m["train_window_years"],
        step_months=m["step_months"],
        min_train_rows=m["min_train_rows"],
        xgb_params=m["xgb_params"],
        calibration=m.get("calibration"),
    )


def _backtest_cfg(c: dict) -> BacktestConfig:
    b = c["backtest"]
    return BacktestConfig(
        edge_threshold=b["edge_threshold"],
        prob_min=b["prob_min"], prob_max=b["prob_max"],
        kelly_fraction=b["kelly_fraction"], kelly_cap=b["kelly_cap"],
        flat_stake=b["flat_stake"], initial_bankroll=b["initial_bankroll"],
    )


def _print_metrics(metrics: dict):
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS  (out-of-sample, walk-forward)")
    print("=" * 70)
    headline = [
        ("# bets", metrics["n_bets"]),
        ("Win rate", f"{metrics['win_rate']*100:.2f}%"),
        ("Avg edge", f"{metrics['avg_edge']*100:.2f}%"),
        ("Avg odds", f"{metrics['avg_odds']:.2f}"),
        ("Yield (flat)", f"{metrics['yield_pct']:.2f}%"),
        ("Profit (flat units)", f"{metrics['flat_profit']:.2f}"),
        ("Max DD (units)", f"{metrics['max_drawdown_units']:.2f}"),
        ("Approx Sharpe", f"{metrics['approx_sharpe']:.2f}"),
        ("Kelly final bankroll", f"{metrics['kelly_final_bankroll']:.2f}"),
    ]
    for k, v in headline:
        print(f"  {k:<22} {v}")

    print("\nBy tour:")
    for k, v in metrics["by_tour"].items():
        print(f"  {k:<6} n={v['n']:>5}  win_rate={v['win_rate']*100:5.1f}%  yield={v['yield_pct']:+6.2f}%")

    print("\nBy surface:")
    for k, v in metrics["by_surface"].items():
        print(f"  {k:<10} n={v['n']:>5}  win_rate={v['win_rate']*100:5.1f}%  yield={v['yield_pct']:+6.2f}%")

    print("\nBy year:")
    for year in sorted(metrics["yearly"].keys()):
        v = metrics["yearly"][year]
        print(f"  {year}  n={v['n']:>5}  yield={v['yield_pct']:+6.2f}%  profit={v['profit_units']:+7.2f}u")
    print("=" * 70)
    print(
        "Interpretation: yield is profit divided by total stakes. A flat-stake "
        "yield of +2-5% on >1000 bets across multiple years is the realistic "
        "professional target. Backtest yields above ~10% on small samples are "
        "almost always overfit or sourced from non-replicable closing prices."
    )


def run_pipeline(config_path: str) -> dict:
    cfg = _load_config(config_path)
    _setup_logging(cfg["logging"]["level"], cfg["logging"]["file"])
    log = logging.getLogger("tennis_model.run")

    art_dir = Path(cfg["artifacts"]["model_path"]).parent
    art_dir.mkdir(parents=True, exist_ok=True)

    log.info("Step 1/5: ingesting data")
    full = load_all(_ingestion_cfg(cfg))
    p1p2 = matches_to_p1p2(full)

    log.info("Step 2/5: building features")
    feat_cfg = _feature_cfg(cfg)
    features, state = build_features(p1p2, full, feat_cfg)

    log.info("Step 3/5: walk-forward training")
    oof, folds, artifacts = walk_forward_train(features, _model_cfg(cfg))

    log.info("Step 4/5: backtesting")
    bt_cfg = _backtest_cfg(cfg)
    bets = select_bets(oof, bt_cfg)
    bets, metrics = simulate(bets, bt_cfg)
    bets.to_csv(cfg["artifacts"]["backtest_csv"], index=False)

    # Persist the full OOF frame so multi-source comparison runs (odds_compare)
    # don't have to retrain. Parquet keeps the per-source odds columns intact.
    oof_path = Path(cfg["artifacts"]["model_path"]).parent / "oof_predictions.parquet"
    try:
        oof.to_parquet(oof_path, index=False)
        log.info("Saved OOF frame: %s (%d rows)", oof_path, len(oof))
    except Exception as e:
        # Parquet may not be available in all envs; CSV fallback.
        log.warning("Parquet save failed (%s); writing CSV instead", e)
        oof.to_csv(oof_path.with_suffix(".csv"), index=False)

    calib = calibration_table(oof)
    print("\nCalibration (model vs market vs actual):")
    print(calib.to_string(index=False))

    log.info("Step 5/5: persisting artifacts")
    joblib.dump({
        "model": artifacts["final_model"],
        "feature_columns": artifacts["feature_columns"],
        "categories": artifacts["categories"],
        "feature_state": state,
        "feature_config": feat_cfg,
        "config": cfg,
    }, cfg["artifacts"]["model_path"])

    folds_df = summarize_folds(folds)
    folds_df.to_csv(art_dir / "folds.csv", index=False)
    with open(cfg["artifacts"]["metrics_json"], "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    _print_metrics(metrics)
    return metrics


# --------- live prediction ----------

def _row_features_from_state(state: FeatureState, row: pd.Series, cfg: FeatureConfig) -> dict:
    """Build the feature dict for an upcoming match using current state."""
    surface = row["surface"] if isinstance(row["surface"], str) else "Hard"
    p1, p2 = row["player1"], row["player2"]
    date = pd.Timestamp(row["date"])

    p1_eo, p1_es, p1_eb = state.blended_elo(p1, surface)
    p2_eo, p2_es, p2_eb = state.blended_elo(p2, surface)
    rank_p1 = float(row["p1_rank"]) if pd.notna(row.get("p1_rank")) else 500.0
    rank_p2 = float(row["p2_rank"]) if pd.notna(row.get("p2_rank")) else 500.0
    pts_p1 = float(row["p1_pts"]) if pd.notna(row.get("p1_pts")) else 0.0
    pts_p2 = float(row["p2_pts"]) if pd.notna(row.get("p2_pts")) else 0.0

    feat = {
        "tour": row["tour"], "surface": surface,
        "level_bucket": _level_bucket(row.get("level")),
        "round_num": _round_bucket(row.get("round")),
        "best_of": float(row["best_of"]) if pd.notna(row.get("best_of")) else 3.0,
        "elo_overall_diff": p1_eo - p2_eo,
        "elo_surface_diff": p1_es - p2_es,
        "elo_blend_diff": p1_eb - p2_eb,
        "elo_blend_p1": p1_eb, "elo_blend_p2": p2_eb,
        "rank_p1": rank_p1, "rank_p2": rank_p2,
        "rank_diff": rank_p1 - rank_p2,
        "log_rank_ratio": np.log(rank_p2 + 1) - np.log(rank_p1 + 1),
        "pts_p1": pts_p1, "pts_p2": pts_p2,
        "pts_diff": pts_p1 - pts_p2,
        "log_pts_ratio": np.log(pts_p1 + 1) - np.log(pts_p2 + 1),
    }
    for w in cfg.recent_form_windows:
        f1, f2 = state.form_pct(p1, w), state.form_pct(p2, w)
        feat[f"form_overall_{w}_diff"] = (f1 - f2) if (f1 is not None and f2 is not None) else 0.0
        f1s, f2s = state.form_pct(p1, w, surface), state.form_pct(p2, w, surface)
        feat[f"form_surface_{w}_diff"] = (f1s - f2s) if (f1s is not None and f2s is not None) else 0.0
    c1, c2 = state.career_surface_winpct(p1, surface), state.career_surface_winpct(p2, surface)
    feat["career_surface_winpct_diff"] = (c1 - c2) if (c1 is not None and c2 is not None) else 0.0
    r1, r2 = state.days_since_last(p1, date), state.days_since_last(p2, date)
    feat["rest_days_p1"], feat["rest_days_p2"], feat["rest_diff"] = r1, r2, r1 - r2
    h1, h2 = state.get_h2h(p1, p2)
    feat["h2h_total"] = h1 + h2
    feat["h2h_p1_winpct"] = (h1 / (h1 + h2)) if (h1 + h2) > 0 else 0.5
    sh1, sh2 = state.get_h2h(p1, p2, surface)
    feat["h2h_surface_total"] = sh1 + sh2
    feat["h2h_surface_p1_winpct"] = (sh1 / (sh1 + sh2)) if (sh1 + sh2) > 0 else 0.5
    s1s, s1r = state.serve_avg(p1)
    s2s, s2r = state.serve_avg(p2)
    feat["serve_winpct_diff"] = (s1s - s2s) if (s1s is not None and s2s is not None) else 0.0
    feat["return_winpct_diff"] = (s1r - s2r) if (s1r is not None and s2r is not None) else 0.0
    for col in ["p1_age", "p2_age", "p1_ht", "p2_ht"]:
        feat[col] = float(row[col]) if (col in row.index and pd.notna(row[col])) else np.nan
    feat["age_diff"] = (feat["p1_age"] - feat["p2_age"]) if not (np.isnan(feat["p1_age"]) or np.isnan(feat["p2_age"])) else 0.0
    feat["height_diff"] = (feat["p1_ht"] - feat["p2_ht"]) if not (np.isnan(feat["p1_ht"]) or np.isnan(feat["p2_ht"])) else 0.0
    h1, h2 = row.get("p1_hand"), row.get("p2_hand")
    feat["same_hand"] = 1.0 if (isinstance(h1, str) and isinstance(h2, str) and h1 == h2) else (-1.0 if (isinstance(h1, str) and isinstance(h2, str)) else 0.0)
    return feat


def predict_match(model_artifact_path: str, upcoming: pd.DataFrame,
                  edge_threshold: float = 0.04) -> pd.DataFrame:
    """Score a frame of upcoming matches.

    Returns a DataFrame with model_p1, model_p2, market_p1, market_p2, edges,
    recommended side, and Kelly fraction (using config defaults).
    """
    art = joblib.load(model_artifact_path)
    model = art["model"]
    columns = art["feature_columns"]
    categories = art["categories"]
    state: FeatureState = art["feature_state"]
    feat_cfg: FeatureConfig = art["feature_config"]

    rows = []
    for _, r in upcoming.iterrows():
        f = _row_features_from_state(state, r, feat_cfg)
        rows.append(f)
    feat_df = pd.DataFrame(rows)
    X, _ = encode_features(feat_df, categories)
    for col in columns:
        if col not in X.columns:
            X[col] = 0
    X = X[columns]

    proba = model.predict_proba(X)[:, 1]

    out = upcoming.copy().reset_index(drop=True)
    out["model_p1"] = proba
    out["model_p2"] = 1 - proba
    if "p1_odds" in out.columns and "p2_odds" in out.columns:
        market = out.apply(lambda r: _devig_two_way(r["p1_odds"], r["p2_odds"]), axis=1, result_type="expand")
        market.columns = ["market_p1", "market_p2"]
        out = pd.concat([out, market], axis=1)
        out["edge_p1"] = out["model_p1"] * out["p1_odds"] - 1.0
        out["edge_p2"] = out["model_p2"] * out["p2_odds"] - 1.0
        out["recommended_side"] = np.where(
            (out["edge_p1"] > edge_threshold) & (out["edge_p1"] >= out["edge_p2"]), "p1",
            np.where(out["edge_p2"] > edge_threshold, "p2", "no_bet"),
        )
        out["recommended_edge"] = np.where(out["recommended_side"] == "p1", out["edge_p1"],
                                           np.where(out["recommended_side"] == "p2", out["edge_p2"], 0.0))
        # Kelly
        def _k_for_row(r):
            if r["recommended_side"] == "p1":
                p, o = r["model_p1"], r["p1_odds"]
            elif r["recommended_side"] == "p2":
                p, o = r["model_p2"], r["p2_odds"]
            else:
                return 0.0
            b = o - 1
            return float(np.clip((p * (b + 1) - 1) / b, 0.0, 1.0)) if b > 0 else 0.0
        out["kelly_full"] = out.apply(_k_for_row, axis=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="tennis_model/config.yaml")
    p.add_argument("--action", choices=["train", "predict", "compare-odds"], default="train")
    p.add_argument("--upcoming", help="CSV of upcoming matches for --action predict")
    p.add_argument("--edge", type=float, default=0.04, help="edge threshold for predict")
    args = p.parse_args()

    if args.action == "train":
        run_pipeline(args.config)
    elif args.action == "compare-odds":
        from tennis_model.odds_compare import run_comparison
        run_comparison(args.config)
    else:
        if not args.upcoming:
            raise SystemExit("--upcoming CSV required for predict action")
        cfg = _load_config(args.config)
        _setup_logging(cfg["logging"]["level"], cfg["logging"]["file"])
        upcoming = pd.read_csv(args.upcoming, parse_dates=["date"])
        result = predict_match(cfg["artifacts"]["model_path"], upcoming, edge_threshold=args.edge)
        out_path = "tennis_model/artifacts/predictions.csv"
        result.to_csv(out_path, index=False)
        print(result.to_string(index=False))
        print(f"\nWrote {len(result)} predictions to {out_path}")


if __name__ == "__main__":
    main()
