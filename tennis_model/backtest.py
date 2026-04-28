"""Backtesting engine for the out-of-sample predictions.

We operate on the OOF dataframe produced by walk-forward training. For each
match we have:
  * model probability for player1 (`pred`)
  * decimal odds for both sides (`p1_odds`, `p2_odds`)
  * the realized label (`y` = 1 iff player1 won)

For each side we compute edge = (model_prob * odds) - 1, which is the expected
return per unit staked. We bet the side with the positive edge (if any) when
edge > threshold and the model probability is in [prob_min, prob_max].

Two staking modes:
  * Flat (1 unit/bet) - cleanest measure of yield.
  * Fractional Kelly - sizes by edge and odds, capped to limit ruin risk.

Output: per-bet ledger, equity curve, and a metrics dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    edge_threshold: float = 0.04
    prob_min: float = 0.10
    prob_max: float = 0.90
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.05
    flat_stake: float = 1.0
    initial_bankroll: float = 100.0


def _kelly(p: float, odds: float) -> float:
    """Kelly fraction for a binary bet at decimal odds. Clipped at [0, 1]."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * (b + 1) - 1) / b
    return float(np.clip(f, 0.0, 1.0))


def select_bets(oof: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Build a ledger of bets to place from out-of-sample predictions."""
    df = oof.copy()
    df["pred_p2"] = 1.0 - df["pred"]
    df["edge_p1"] = df["pred"] * df["p1_odds"] - 1.0
    df["edge_p2"] = df["pred_p2"] * df["p2_odds"] - 1.0

    # Pick the side with the larger positive edge, if any.
    take_p1 = (df["edge_p1"] > cfg.edge_threshold) & (df["edge_p1"] >= df["edge_p2"]) & \
              (df["pred"] >= cfg.prob_min) & (df["pred"] <= cfg.prob_max)
    take_p2 = (df["edge_p2"] > cfg.edge_threshold) & (df["edge_p2"] > df["edge_p1"]) & \
              (df["pred_p2"] >= cfg.prob_min) & (df["pred_p2"] <= cfg.prob_max)

    bets = []
    for idx in df.index[take_p1 | take_p2]:
        row = df.loc[idx]
        if take_p1.loc[idx]:
            side, prob, odds, edge = "p1", row["pred"], row["p1_odds"], row["edge_p1"]
            won = (row["y"] == 1)
        else:
            side, prob, odds, edge = "p2", row["pred_p2"], row["p2_odds"], row["edge_p2"]
            won = (row["y"] == 0)
        kelly_full = _kelly(prob, odds)
        kelly_size = min(cfg.kelly_fraction * kelly_full, cfg.kelly_cap)
        bets.append({
            "date": row["date"],
            "tour": row["tour"],
            "surface": row["surface"],
            "level_bucket": row["level_bucket"],
            "player1": row.get("player1"),
            "player2": row.get("player2"),
            "side": side,
            "model_prob": prob,
            "market_prob": row["market_p1"] if side == "p1" else (1 - row["market_p1"]),
            "odds": odds,
            "edge": edge,
            "kelly_fraction": kelly_size,
            "won": int(won),
            "fold": int(row["fold"]) if pd.notna(row["fold"]) else -1,
        })
    return pd.DataFrame(bets).sort_values("date").reset_index(drop=True)


def simulate(bets: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    """Add P&L columns and compute summary metrics for both staking modes."""
    if bets.empty:
        log.warning("No bets selected at edge_threshold=%.3f", cfg.edge_threshold)
        return bets, {"n_bets": 0}

    b = bets.copy()
    # Flat staking
    b["flat_stake"] = cfg.flat_stake
    b["flat_pnl"] = np.where(b["won"] == 1, b["flat_stake"] * (b["odds"] - 1), -b["flat_stake"])
    b["flat_cum_pnl"] = b["flat_pnl"].cumsum()

    # Kelly: stake = kelly_fraction * current_bankroll
    bankroll = cfg.initial_bankroll
    kelly_pnl = []
    bankroll_curve = []
    for _, row in b.iterrows():
        stake = bankroll * row["kelly_fraction"]
        if row["won"] == 1:
            pnl = stake * (row["odds"] - 1)
        else:
            pnl = -stake
        bankroll += pnl
        kelly_pnl.append(pnl)
        bankroll_curve.append(bankroll)
    b["kelly_pnl"] = kelly_pnl
    b["bankroll"] = bankroll_curve

    # Metrics
    n_bets = len(b)
    flat_turnover = b["flat_stake"].sum()
    flat_profit = b["flat_pnl"].sum()
    yield_pct = flat_profit / flat_turnover if flat_turnover > 0 else 0.0

    # Max drawdown on the flat curve.
    cum = b["flat_cum_pnl"].values
    high_watermark = np.maximum.accumulate(cum)
    drawdown = high_watermark - cum
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    # Per-bet "Sharpe": mean / std of flat P&L * sqrt(bets/year). Ballpark only.
    if b["flat_pnl"].std(ddof=0) > 0:
        bets_per_year = n_bets / max((b["date"].max() - b["date"].min()).days / 365.25, 1.0)
        sharpe = (b["flat_pnl"].mean() / b["flat_pnl"].std(ddof=0)) * np.sqrt(bets_per_year)
    else:
        sharpe = 0.0

    metrics = {
        "n_bets": int(n_bets),
        "win_rate": float(b["won"].mean()),
        "avg_edge": float(b["edge"].mean()),
        "avg_odds": float(b["odds"].mean()),
        "flat_turnover": float(flat_turnover),
        "flat_profit": float(flat_profit),
        "yield_pct": float(yield_pct * 100),
        "roi_flat_pct": float(flat_profit / flat_turnover * 100) if flat_turnover > 0 else 0.0,
        "max_drawdown_units": max_dd,
        "approx_sharpe": float(sharpe),
        "kelly_final_bankroll": float(bankroll),
        "kelly_roi_pct": float((bankroll - cfg.initial_bankroll) / cfg.initial_bankroll * 100),
        "by_tour": _group_metrics(b, "tour"),
        "by_surface": _group_metrics(b, "surface"),
        "by_level": _group_metrics(b, "level_bucket"),
        "yearly": _yearly_metrics(b),
    }
    return b, metrics


def _group_metrics(b: pd.DataFrame, col: str) -> dict:
    out = {}
    for key, sub in b.groupby(col):
        if sub.empty:
            continue
        turn = sub["flat_stake"].sum()
        out[str(key)] = {
            "n": int(len(sub)),
            "win_rate": float(sub["won"].mean()),
            "yield_pct": float(sub["flat_pnl"].sum() / turn * 100) if turn > 0 else 0.0,
        }
    return out


def _yearly_metrics(b: pd.DataFrame) -> dict:
    out = {}
    for year, sub in b.groupby(b["date"].dt.year):
        turn = sub["flat_stake"].sum()
        out[int(year)] = {
            "n": int(len(sub)),
            "yield_pct": float(sub["flat_pnl"].sum() / turn * 100) if turn > 0 else 0.0,
            "profit_units": float(sub["flat_pnl"].sum()),
        }
    return out


def calibration_table(oof: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram bins for the OOF predictions."""
    df = oof.copy()
    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["pred"], bins, include_lowest=True)
    out = df.groupby("bin").agg(
        n=("y", "size"),
        avg_pred=("pred", "mean"),
        actual_rate=("y", "mean"),
        avg_market=("market_p1", "mean"),
    ).reset_index()
    return out
