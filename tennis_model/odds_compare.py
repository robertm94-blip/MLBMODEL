"""Compare model performance against multiple odds sources without retraining.

Loads the OOF predictions saved by run.py and re-runs select_bets + simulate
against each odds source independently. Reports yield, year-by-year stability,
ATP/WTA split, surface split, and a CLV diagnostic showing whether the model
is systematically picking spots where the chosen source offers better prices
than Pinnacle.

Usage:
    python -m tennis_model.odds_compare --config tennis_model/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tennis_model.backtest import BacktestConfig, select_bets, simulate

log = logging.getLogger(__name__)

SOURCES = ("pinnacle", "max", "avg", "b365")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _backtest_cfg(c: dict) -> BacktestConfig:
    b = c["backtest"]
    return BacktestConfig(
        edge_threshold=b["edge_threshold"],
        prob_min=b["prob_min"], prob_max=b["prob_max"],
        kelly_fraction=b["kelly_fraction"], kelly_cap=b["kelly_cap"],
        flat_stake=b["flat_stake"], initial_bankroll=b["initial_bankroll"],
    )


def _load_oof(model_path: str) -> pd.DataFrame:
    """Load the OOF frame saved next to model.joblib."""
    base = Path(model_path).parent
    pq = base / "oof_predictions.parquet"
    csv = base / "oof_predictions.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["date"])
    raise FileNotFoundError(
        f"No OOF frame found at {pq} or {csv}. Run `python -m tennis_model.run` first."
    )


def _summarize(bets: pd.DataFrame, n_universe: int) -> dict:
    if bets.empty:
        return {"n_bets": 0, "win_rate": np.nan, "yield_pct": np.nan,
                "profit": 0.0, "max_dd": 0.0, "n_universe": n_universe}
    flat_pnl = np.where(bets["won"] == 1, bets["odds"] - 1, -1.0)
    profit = float(flat_pnl.sum())
    cum = np.cumsum(flat_pnl)
    high = np.maximum.accumulate(cum)
    max_dd = float((high - cum).max()) if len(cum) else 0.0
    return {
        "n_bets": int(len(bets)),
        "win_rate": float(bets["won"].mean()),
        "yield_pct": profit / len(bets) * 100,
        "profit": profit,
        "max_dd": max_dd,
        "avg_odds": float(bets["odds"].mean()),
        "avg_edge_pct": float(bets["edge"].mean() * 100),
        "n_universe": n_universe,
    }


def _yearly(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {}
    flat_pnl = np.where(bets["won"] == 1, bets["odds"] - 1, -1.0)
    bets = bets.assign(_pnl=flat_pnl)
    out = {}
    for y, g in bets.groupby(bets["date"].dt.year):
        out[int(y)] = float(g["_pnl"].sum() / len(g) * 100)
    return out


def _clv_vs_pinnacle(oof: pd.DataFrame, bets: pd.DataFrame, source: str) -> float:
    """Average implied-prob gap (source - pinnacle) on the side the model picked.

    A positive value means the chosen source consistently offers higher implied
    prob (i.e. shorter odds) on the model's pick than Pinnacle did, which would
    be bad - we'd prefer the chosen source to offer *lower* implied prob (longer
    odds, better price). So a *negative* number is the favorable direction.

    To make the metric intuitively read positive=good, we report the inverse:
    average (pinnacle_implied - source_implied) on the bet side. Positive = the
    softer source pays more than Pinnacle on the spots the model picks.
    """
    if bets.empty:
        return 0.0
    # Re-merge by index would be brittle; rebuild using the unique (date, p1, p2)
    # tuple. Fast enough for ~50k rows.
    key_cols = ["date", "player1", "player2"]
    merged = bets.merge(
        oof[key_cols + ["p1_odds_pinnacle", "p2_odds_pinnacle",
                        f"p1_odds_{source}", f"p2_odds_{source}"]],
        on=key_cols, how="left",
    )
    diffs = []
    for _, r in merged.iterrows():
        if r["side"] == "p1":
            pin_o, src_o = r["p1_odds_pinnacle"], r[f"p1_odds_{source}"]
        else:
            pin_o, src_o = r["p2_odds_pinnacle"], r[f"p2_odds_{source}"]
        if pd.notna(pin_o) and pd.notna(src_o) and pin_o > 1 and src_o > 1:
            # Implied prob from each (un-devigged here, since it's a side comparison;
            # the overrounds roughly cancel for a same-row comparison).
            diffs.append((1.0 / pin_o) - (1.0 / src_o))
    return float(np.mean(diffs)) if diffs else 0.0


def run_comparison(config_path: str) -> dict:
    cfg = _load_config(config_path)
    logging.basicConfig(
        level=cfg["logging"]["level"],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    bt_cfg = _backtest_cfg(cfg)
    oof = _load_oof(cfg["artifacts"]["model_path"])
    log.info("Loaded OOF frame: %d rows", len(oof))

    missing = [s for s in SOURCES
               if f"p1_odds_{s}" not in oof.columns or f"p2_odds_{s}" not in oof.columns]
    if missing:
        raise RuntimeError(
            f"OOF frame missing per-source columns for {missing}. "
            "Re-run `python -m tennis_model.run` after the multi-source upgrade."
        )

    rows = []
    yearly = {}
    by_tour = {}
    for src in SOURCES:
        n_universe = int((oof[f"p1_odds_{src}"].notna() & oof[f"p2_odds_{src}"].notna()).sum())
        bets = select_bets(oof, bt_cfg, odds_source=src)
        bets, _ = simulate(bets, bt_cfg)
        summary = _summarize(bets, n_universe)
        summary["source"] = src
        summary["clv_vs_pinnacle_implied"] = _clv_vs_pinnacle(oof, bets, src) if src != "pinnacle" else 0.0
        rows.append(summary)
        yearly[src] = _yearly(bets)
        by_tour[src] = {
            t: _summarize(g, len(g))["yield_pct"]
            for t, g in bets.groupby("tour")
        } if not bets.empty else {}

    # --- print headline table ---
    print("\n" + "=" * 110)
    print("SOFT-LINE COMPARISON  (same model probabilities; only the odds source changes)")
    print("=" * 110)
    header = f"{'source':<10} {'universe':>9} {'n_bets':>8} {'win%':>6} {'avg_odds':>9} {'avg_edge':>9} {'yield%':>8} {'profit_u':>10} {'max_dd':>8} {'CLV_vs_pin':>11}"
    print(header)
    print("-" * 110)
    for r in rows:
        print(f"{r['source']:<10} {r['n_universe']:>9d} {r['n_bets']:>8d} "
              f"{r['win_rate']*100:>5.1f}% {r.get('avg_odds', float('nan')):>9.2f} "
              f"{r.get('avg_edge_pct', float('nan')):>8.1f}% {r['yield_pct']:>+7.2f}% "
              f"{r['profit']:>+10.2f} {r['max_dd']:>8.1f} {r['clv_vs_pinnacle_implied']*100:>+10.2f}%")

    # --- year-by-year ---
    all_years = sorted({y for src_yearly in yearly.values() for y in src_yearly})
    print("\nYear-by-year yield % (positive = profit):")
    print(f"  {'year':<6} " + " ".join(f"{s:>10}" for s in SOURCES))
    for y in all_years:
        cells = [f"{yearly[s].get(y, float('nan')):>+9.2f}%" for s in SOURCES]
        print(f"  {y:<6} " + " ".join(cells))

    # --- per tour ---
    print("\nYield % by tour:")
    for s in SOURCES:
        cells = " ".join(f"{t}={by_tour[s].get(t, float('nan')):>+5.2f}%" for t in ("ATP", "WTA"))
        print(f"  {s:<10} {cells}")

    # --- decision summary ---
    print("\n" + "-" * 110)
    print("Decision rule (from plan): yield ≥ +2% AND ≥7/13 years positive AND CLV > 0 → soft-line viable")
    for r in rows:
        if r["source"] == "pinnacle":
            continue
        n_pos_years = sum(1 for v in yearly[r["source"]].values() if v > 0)
        viable = (r["yield_pct"] >= 2.0
                  and n_pos_years >= 7
                  and r["clv_vs_pinnacle_implied"] > 0)
        verdict = "VIABLE" if viable else "not viable"
        print(f"  {r['source']:<10} yield={r['yield_pct']:+6.2f}% pos_years={n_pos_years}/{len(yearly[r['source']])} "
              f"clv_implied_pp={r['clv_vs_pinnacle_implied']*100:+5.2f}  -> {verdict}")
    print("=" * 110)
    return {"rows": rows, "yearly": yearly, "by_tour": by_tour}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="tennis_model/config.yaml")
    args = p.parse_args()
    run_comparison(args.config)


if __name__ == "__main__":
    main()
