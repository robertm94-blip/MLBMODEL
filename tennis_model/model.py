"""Model layer: feature pipeline, walk-forward training, and calibration.

Why walk-forward over k-fold:
  Tennis match outcomes are temporally correlated (player form, ranking momentum,
  surface seasons, rule/equipment changes). Random k-fold leaks future info into
  the training set. Walk-forward — train on [t0, t_train_end), evaluate on
  [t_train_end, t_train_end + step) — is the correct simulation of how the model
  would have been used in production at each historical point.

Why XGBoost:
  Robust on tabular data with missing values, fast to retrain inside a
  walk-forward loop, and gives well-behaved probability outputs that respond
  well to isotonic calibration.

We train one model across both tours by default (ATP and WTA share most signal
sources — Elo, form, surface — and combining roughly doubles the training set).
A `tour` one-hot is included so the model can shift if needed. The flag
`per_tour=True` switches to two separate models if you'd rather isolate them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

# sklearn >=1.6 deprecated `cv="prefit"` in favor of FrozenEstimator. Support both.
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False

from tennis_model.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

log = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except ImportError:
    _XGB_OK = False


@dataclass
class ModelConfig:
    train_window_years: int = 4
    step_months: int = 3
    min_train_rows: int = 5000
    xgb_params: dict | None = None
    calibration: str | None = "isotonic"


def encode_features(df: pd.DataFrame, train_categories: dict[str, list] | None = None
                    ) -> tuple[pd.DataFrame, dict[str, list]]:
    """One-hot encode categoricals using a fixed category list.

    `train_categories` lets us re-use the categories learned on the training
    set when encoding a future fold, so columns are stable.
    """
    out = df[FEATURE_COLUMNS].copy()
    cats: dict[str, list] = {}
    for col in CATEGORICAL_COLUMNS:
        series = df[col].fillna("Unknown").astype(str)
        if train_categories is not None and col in train_categories:
            categories = train_categories[col]
        else:
            categories = sorted(series.unique().tolist())
        cats[col] = categories
        for cat in categories:
            out[f"{col}__{cat}"] = (series == cat).astype(np.int8)
    return out, cats


def _build_xgb(params: dict):
    if not _XGB_OK:
        raise RuntimeError("xgboost is not installed; pip install xgboost")
    return XGBClassifier(**params)


def fit_one_fold(X_train: pd.DataFrame, y_train: np.ndarray,
                 X_calib: pd.DataFrame | None, y_calib: np.ndarray | None,
                 cfg: ModelConfig):
    """Fit XGBoost on the training window, then optionally calibrate on a hold-out."""
    params = dict(cfg.xgb_params or {})
    # XGBoost wants n_estimators / learning_rate at constructor time.
    base = _build_xgb(params)
    base.fit(X_train, y_train, verbose=False)

    if cfg.calibration and X_calib is not None and len(X_calib) >= 200:
        # Wrap a "prefit" calibrator around the trained XGB.
        if _HAS_FROZEN:
            calib = CalibratedClassifierCV(FrozenEstimator(base), method=cfg.calibration)
        else:
            calib = CalibratedClassifierCV(base, method=cfg.calibration, cv="prefit")
        calib.fit(X_calib, y_calib)
        return calib
    return base


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    log_loss: float
    brier: float
    accuracy: float
    market_log_loss: float
    market_brier: float


def walk_forward_train(features: pd.DataFrame, cfg: ModelConfig,
                       per_tour: bool = False
                       ) -> tuple[pd.DataFrame, list[FoldResult], dict]:
    """Run the walk-forward loop.

    Returns:
        oof_df: features dataframe with extra columns 'pred', 'fold' for every
                row that fell in some test fold (i.e. has out-of-sample preds).
        fold_results: list of per-fold metrics
        artifacts: {'final_model': last fitted model, 'categories': cat dict,
                    'feature_columns': list of columns the model expects}

    The "final" model is refit on *all* available data at the end so it can be
    used for live prediction on future matches.
    """
    if per_tour:
        results = []
        all_oof = []
        artifacts = {}
        for tour in features["tour"].unique():
            sub = features[features["tour"] == tour].reset_index(drop=True)
            oof, folds, art = walk_forward_train(sub, cfg, per_tour=False)
            results.extend([(tour, f) for f in folds])
            all_oof.append(oof)
            artifacts[tour] = art
        oof_df = pd.concat(all_oof, ignore_index=True).sort_values("date").reset_index(drop=True)
        return oof_df, [r[1] for r in results], artifacts

    df = features.sort_values("date").reset_index(drop=True).copy()
    df["pred"] = np.nan
    df["fold"] = -1

    train_window = pd.DateOffset(years=cfg.train_window_years)
    step = pd.DateOffset(months=cfg.step_months)

    # Walk-forward windows
    if df.empty:
        raise RuntimeError("No features to train on.")
    start_date = df["date"].min() + train_window
    end_date = df["date"].max()

    fold_results: list[FoldResult] = []
    fold_id = 0
    cursor = start_date

    last_categories: dict[str, list] | None = None

    while cursor < end_date:
        train_mask = df["date"] < cursor
        test_mask = (df["date"] >= cursor) & (df["date"] < cursor + step)
        if train_mask.sum() < cfg.min_train_rows or test_mask.sum() == 0:
            cursor = cursor + step
            continue

        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]

        # Reserve the last 10% of the training window as a calibration set
        # to avoid feeding the calibrator data the base model has memorized.
        calib_split = int(len(train_df) * 0.9)
        fit_df = train_df.iloc[:calib_split]
        calib_df = train_df.iloc[calib_split:]

        X_fit, last_categories = encode_features(fit_df, last_categories)
        y_fit = fit_df["y"].values
        X_calib, _ = encode_features(calib_df, last_categories)
        y_calib = calib_df["y"].values
        X_test, _ = encode_features(test_df, last_categories)
        y_test = test_df["y"].values

        # Align columns (some categories may be unseen in fit set)
        for col in X_fit.columns:
            if col not in X_test.columns:
                X_test[col] = 0
            if col not in X_calib.columns:
                X_calib[col] = 0
        X_test = X_test[X_fit.columns]
        X_calib = X_calib[X_fit.columns]

        log.info(
            "Fold %d: train=[%s..%s] (%d rows), test=[%s..%s] (%d rows)",
            fold_id, train_df["date"].min().date(), train_df["date"].max().date(),
            len(train_df),
            test_df["date"].min().date(), test_df["date"].max().date(),
            len(test_df),
        )

        clf = fit_one_fold(X_fit, y_fit, X_calib, y_calib, cfg)
        proba = clf.predict_proba(X_test)[:, 1]

        df.loc[test_df.index, "pred"] = proba
        df.loc[test_df.index, "fold"] = fold_id

        market = test_df["market_p1"].values
        try:
            ll = log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6))
            mll = log_loss(y_test, np.clip(market, 1e-6, 1 - 1e-6))
        except ValueError:
            ll = mll = float("nan")
        br = brier_score_loss(y_test, proba)
        mbr = brier_score_loss(y_test, market)
        acc = float(((proba >= 0.5) == (y_test == 1)).mean())

        fold_results.append(FoldResult(
            fold_id=fold_id,
            train_start=train_df["date"].min(),
            train_end=train_df["date"].max(),
            test_start=test_df["date"].min(),
            test_end=test_df["date"].max(),
            n_train=len(train_df),
            n_test=len(test_df),
            log_loss=ll, brier=br, accuracy=acc,
            market_log_loss=mll, market_brier=mbr,
        ))
        log.info(
            "  log_loss model=%.4f market=%.4f | brier model=%.4f market=%.4f | acc=%.3f",
            ll, mll, br, mbr, acc,
        )

        fold_id += 1
        cursor = cursor + step

    # Refit on ALL data for the final live model.
    log.info("Refitting final model on full dataset (%d rows)", len(df))
    X_all, all_cats = encode_features(df, last_categories)
    y_all = df["y"].values
    cut = int(len(X_all) * 0.9)
    final_model = fit_one_fold(X_all.iloc[:cut], y_all[:cut],
                               X_all.iloc[cut:], y_all[cut:], cfg)

    artifacts = {
        "final_model": final_model,
        "categories": all_cats,
        "feature_columns": list(X_all.columns),
    }
    oof_df = df.dropna(subset=["pred"]).copy()
    return oof_df, fold_results, artifacts


def summarize_folds(folds: list[FoldResult]) -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in folds])
