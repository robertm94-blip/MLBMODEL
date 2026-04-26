"""Live Edge Dashboard — Streamlit app.

Reads:
- `team_features` (from feature_engineer.py) for projected win% and totals
- `data/odds/{date}.csv` (optional, manual drop-in) for market lines
- `odds_snapshots` (from storage.py) for line-movement history

Renders:
- Filters: Sport (MLB / NBA / All), Minimum Edge, Date
- `st.dataframe` betting slate with fair lines, market lines (when present),
  edge %, and Kelly stake %
- Per-game line-movement chart (Chart.js via st.components.v1.html)

Run:
    streamlit run app.py
    streamlit run app.py -- --db /path/to/data_ingestion.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.data_ingestion.storage import DEFAULT_DB_PATH, SQLiteStore
from src.edge import (
    american_to_implied_prob,
    half_kelly,
    kelly_fraction,
    remove_vig,
)

ODDS_DIR = os.path.join("data", "odds")
SUPPORTED_SPORTS = ["mlb", "nba"]


# =====================================================================
# CLI args (Streamlit forwards anything after `--`)
# =====================================================================

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


CLI = _parse_cli()


# =====================================================================
# Data loaders (cached so Streamlit reruns are cheap)
# =====================================================================

@st.cache_resource
def get_store(db_path: str) -> SQLiteStore:
    return SQLiteStore(db_path)


@st.cache_data(ttl=60)
def load_features(db_path: str, sport: str, game_date: str) -> pd.DataFrame:
    """Load per-team features and pivot to one row per game."""
    store = get_store(db_path)
    rows = store.conn.execute(
        """
        SELECT * FROM team_features
        WHERE sport=? AND game_date=?
        ORDER BY game_id, side
        """,
        (sport, game_date),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    home = df[df["side"] == "home"].add_prefix("home_").rename(
        columns={"home_game_id": "game_id", "home_game_date": "game_date",
                 "home_sport": "sport", "home_venue_id": "venue_id"}
    )
    away = df[df["side"] == "away"].add_prefix("away_").rename(
        columns={"away_game_id": "game_id", "away_game_date": "game_date",
                 "away_sport": "sport", "away_venue_id": "venue_id"}
    )
    keep_cols_home = ["sport", "game_id", "game_date", "venue_id"]
    keep_cols_away = ["game_id"]
    home = home[keep_cols_home + [c for c in home.columns if c.startswith("home_")
                                  and c not in {"home_side"}]]
    away = away[keep_cols_away + [c for c in away.columns if c.startswith("away_")
                                  and c not in {"away_side"}]]
    return home.merge(away, on="game_id", how="inner")


@st.cache_data(ttl=60)
def load_market_odds(game_date: str) -> pd.DataFrame:
    """Optional manual drop-in: data/odds/{date}.csv

    Expected columns:
        game_id, away_ml, home_ml, total_line, over_odds, under_odds
    Any missing column is OK; downstream code treats it as NaN.
    """
    path = os.path.join(ODDS_DIR, f"{game_date}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"game_id": str})
    return df


@st.cache_data(ttl=300)
def available_dates(db_path: str, sport: str) -> list[str]:
    store = get_store(db_path)
    rows = store.conn.execute(
        """
        SELECT DISTINCT game_date FROM team_features
        WHERE sport=? ORDER BY game_date DESC
        """,
        (sport,),
    ).fetchall()
    return [r["game_date"] for r in rows]


@st.cache_data(ttl=30)
def load_line_movement(db_path: str, sport: str, game_id: str) -> pd.DataFrame:
    store = get_store(db_path)
    snaps = store.query_odds_snapshots(sport, game_id)
    if not snaps:
        return pd.DataFrame()
    df = pd.DataFrame(snaps)
    df["captured_at"] = pd.to_datetime(df["captured_at"])
    return df


# =====================================================================
# Edge computation
# =====================================================================

def _safe_int(v: Any) -> int | None:
    try:
        if v is None or pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def attach_edges(slate: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Compute edge%, Kelly%, and best-side per game when market odds are present."""
    if slate.empty:
        return slate
    out = slate.copy()
    for col in (
        "away_market_ml", "home_market_ml", "total_market_line",
        "over_market_odds", "under_market_odds",
        "away_no_vig_prob", "home_no_vig_prob",
        "away_edge_pct", "home_edge_pct",
        "away_kelly_pct", "home_kelly_pct",
        "away_half_kelly_pct", "home_half_kelly_pct",
        "best_side", "best_edge_pct", "best_half_kelly_pct",
    ):
        out[col] = pd.NA

    if odds.empty:
        return out

    odds_idx = odds.set_index("game_id")
    for i, row in out.iterrows():
        gid = str(row["game_id"])
        if gid not in odds_idx.index:
            continue
        o = odds_idx.loc[gid]
        away_ml = _safe_int(o.get("away_ml"))
        home_ml = _safe_int(o.get("home_ml"))
        if away_ml is None or home_ml is None:
            continue
        out.at[i, "away_market_ml"] = away_ml
        out.at[i, "home_market_ml"] = home_ml

        no_vig_away, no_vig_home = remove_vig(away_ml, home_ml)
        out.at[i, "away_no_vig_prob"] = round(no_vig_away * 100, 1)
        out.at[i, "home_no_vig_prob"] = round(no_vig_home * 100, 1)

        model_away = float(row.get("away_lineup_wrc_plus") or 0)  # placeholder
        # Win% comes out of the projection engine, not team_features.
        # We approximate from the per-team `offensive_factor` symmetry: callers
        # who want exact prob should plug in projection_engine output. Here we
        # use a round-trip from the implied no-vig prob if win% is missing.
        # In practice you'll join the projection CSV (column away_win_pct).
        model_away_prob = float(row.get("away_win_pct") or 0) / 100.0 if row.get("away_win_pct") is not None else None
        model_home_prob = float(row.get("home_win_pct") or 0) / 100.0 if row.get("home_win_pct") is not None else None
        if model_away_prob is None or model_home_prob is None:
            continue

        edge_away = model_away_prob - no_vig_away
        edge_home = model_home_prob - no_vig_home
        out.at[i, "away_edge_pct"] = round(edge_away * 100, 2)
        out.at[i, "home_edge_pct"] = round(edge_home * 100, 2)

        # Kelly against the actual posted American odds (vig-included)
        away_dec = (away_ml / 100 + 1) if away_ml > 0 else (100 / abs(away_ml) + 1)
        home_dec = (home_ml / 100 + 1) if home_ml > 0 else (100 / abs(home_ml) + 1)
        out.at[i, "away_kelly_pct"] = round(kelly_fraction(model_away_prob, away_dec) * 100, 2)
        out.at[i, "home_kelly_pct"] = round(kelly_fraction(model_home_prob, home_dec) * 100, 2)
        out.at[i, "away_half_kelly_pct"] = round(half_kelly(model_away_prob, away_dec) * 100, 2)
        out.at[i, "home_half_kelly_pct"] = round(half_kelly(model_home_prob, home_dec) * 100, 2)

        # Best side
        if edge_away > edge_home and edge_away > 0:
            out.at[i, "best_side"] = "away"
            out.at[i, "best_edge_pct"] = round(edge_away * 100, 2)
            out.at[i, "best_half_kelly_pct"] = round(half_kelly(model_away_prob, away_dec) * 100, 2)
        elif edge_home > 0:
            out.at[i, "best_side"] = "home"
            out.at[i, "best_edge_pct"] = round(edge_home * 100, 2)
            out.at[i, "best_half_kelly_pct"] = round(half_kelly(model_home_prob, home_dec) * 100, 2)

    return out


# =====================================================================
# Chart.js component
# =====================================================================

CHART_TEMPLATE = """
<!doctype html>
<html><head>
<meta charset="utf-8" />
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         color: #ddd; background: #0e1117; margin: 0; }}
  .empty {{ padding: 32px; text-align: center; color: #888; }}
</style>
</head><body>
<canvas id="lm" height="320"></canvas>
<script>
  const datasets = {datasets_json};
  if (!datasets || datasets.length === 0) {{
    document.body.innerHTML = '<div class="empty">No line-movement snapshots yet.<br>'
      + 'Write rows to the <code>odds_snapshots</code> table to populate this chart.</div>';
  }} else {{
    const ctx = document.getElementById('lm').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{ datasets: datasets }},
      options: {{
        responsive: true,
        animation: false,
        plugins: {{
          legend: {{ labels: {{ color: '#ddd' }} }},
          title: {{ display: true, color: '#ddd', text: '{title}' }}
        }},
        scales: {{
          x: {{ type: 'time', time: {{ unit: 'hour' }},
                ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }},
          y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }},
                title: {{ display: true, text: 'American odds', color: '#aaa' }} }}
        }}
      }}
    }});
  }}
</script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
</body></html>
"""


def render_line_movement(df: pd.DataFrame, title: str) -> str:
    """Pivot snapshot rows into Chart.js dataset shape."""
    datasets: list[dict[str, Any]] = []
    if df.empty:
        return CHART_TEMPLATE.format(datasets_json="[]", title=title)
    palette = [
        "#4ade80", "#60a5fa", "#f472b6", "#facc15",
        "#a78bfa", "#fb923c", "#f87171", "#34d399",
    ]
    for i, ((market, side, book), grp) in enumerate(
        df.groupby(["market", "side", "sportsbook"], dropna=False)
    ):
        label = f"{market}/{side}"
        if book:
            label += f" ({book})"
        points = [
            {"x": ts.isoformat(), "y": int(o)}
            for ts, o in zip(grp["captured_at"], grp["american_odds"])
            if pd.notna(o)
        ]
        if not points:
            continue
        datasets.append({
            "label": label,
            "data": points,
            "borderColor": palette[i % len(palette)],
            "backgroundColor": palette[i % len(palette)] + "44",
            "tension": 0.2,
            "stepped": "before",
            "pointRadius": 3,
        })
    return CHART_TEMPLATE.format(
        datasets_json=json.dumps(datasets), title=title.replace("'", "\\'")
    )


# =====================================================================
# Layout
# =====================================================================

def main() -> None:
    st.set_page_config(page_title="Live Edge Dashboard", layout="wide")
    st.title("⚾🏀 Live Edge Dashboard")
    st.caption(f"DB: `{CLI.db}`")

    # ---- sidebar filters --------------------------------------------
    with st.sidebar:
        st.header("Filters")
        sport_label = st.selectbox(
            "Sport", ["MLB", "NBA", "All"], index=0,
        )
        sport_keys = (
            ["mlb", "nba"] if sport_label == "All"
            else [sport_label.lower()]
        )
        # Date selector: union of available dates across selected sports.
        all_dates = sorted({d for s in sport_keys for d in available_dates(CLI.db, s)},
                           reverse=True)
        if not all_dates:
            st.warning(f"No team_features rows in DB for: {', '.join(sport_keys)}")
            st.info(
                "Run the feature pipeline first:\n"
                "```\npython feature_engineer.py --date YYYY-MM-DD --write-db\n```"
            )
            return
        chosen_date = st.selectbox("Date", all_dates, index=0)
        min_edge_pct = st.slider(
            "Minimum edge %", min_value=0.0, max_value=15.0, value=0.0, step=0.5,
            help="Filters the slate by max(|away_edge|, |home_edge|). "
                 "Has no effect until market odds are loaded.",
        )

    # ---- load slate -------------------------------------------------
    frames = []
    for sport in sport_keys:
        f = load_features(CLI.db, sport, chosen_date)
        if not f.empty:
            frames.append(f)
    slate = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if slate.empty:
        st.warning(f"No games for {sport_label} on {chosen_date}.")
        return

    # ---- attach win% from projection engine if available ------------
    # team_features doesn't store win% (the projection engine does that).
    # If a CSV exists for this date, join its win% columns in.
    proj_csv = os.path.join("data", "projections_csv", f"projections_{chosen_date}.csv")
    if os.path.exists(proj_csv):
        proj_df = pd.read_csv(proj_csv, dtype={"game_id": str})
        keep = ["game_id", "away_win_pct", "home_win_pct",
                "away_fair_line", "home_fair_line",
                "expected_total_nb", "projected_total", "ou_line", "totals_confidence"]
        keep = [c for c in keep if c in proj_df.columns]
        slate = slate.merge(proj_df[keep], on="game_id", how="left")
    else:
        st.info(
            f"No projection CSV at `{proj_csv}` — running "
            "`python projection_engine.py --date {date}` will populate fair lines, "
            "win%, and totals.".format(date=chosen_date)
        )

    # ---- attach market odds + edges ---------------------------------
    odds_df = load_market_odds(chosen_date)
    if odds_df.empty:
        st.info(
            f"No market odds at `{ODDS_DIR}/{chosen_date}.csv` — "
            "edge and Kelly columns are empty until you drop a CSV "
            "(`game_id, away_ml, home_ml, total_line, over_odds, under_odds`)."
        )
    slate = attach_edges(slate, odds_df)

    # ---- apply Minimum Edge filter ----------------------------------
    if "best_edge_pct" in slate.columns and min_edge_pct > 0:
        before = len(slate)
        slate = slate[slate["best_edge_pct"].fillna(-999) >= min_edge_pct]
        st.caption(
            f"Filtered to edges ≥ {min_edge_pct:.1f}% — {len(slate)}/{before} games"
        )

    # ---- main slate table -------------------------------------------
    display_cols = [
        ("matchup", "Matchup"),
        ("away_win_pct", "Away win%"),
        ("home_win_pct", "Home win%"),
        ("away_fair_line", "Away fair"),
        ("home_fair_line", "Home fair"),
        ("away_market_ml", "Away mkt"),
        ("home_market_ml", "Home mkt"),
        ("away_edge_pct", "Away edge%"),
        ("home_edge_pct", "Home edge%"),
        ("away_half_kelly_pct", "Away ½K%"),
        ("home_half_kelly_pct", "Home ½K%"),
        ("best_side", "Best"),
        ("projected_total", "Proj total"),
        ("ou_line", "OU line"),
        ("totals_confidence", "Conf"),
        ("home_is_opener", "Home opener"),
        ("away_is_opener", "Away opener"),
    ]
    slate["matchup"] = (
        slate["away_team_name"].astype(str)
        + " @ " + slate["home_team_name"].astype(str)
    )
    visible = [c for c, _ in display_cols if c in slate.columns]
    rename = {c: lbl for c, lbl in display_cols if c in slate.columns}
    table = slate[visible].rename(columns=rename)

    st.subheader(f"Slate — {chosen_date}")
    st.dataframe(table, use_container_width=True, hide_index=True)

    # ---- per-game line movement -------------------------------------
    st.subheader("Line movement")
    matchups = slate[["game_id", "sport", "matchup"]].drop_duplicates()
    if matchups.empty:
        st.info("No games to chart.")
        return
    options = {
        f"{row['matchup']}  [{row['sport']} {row['game_id']}]": (row["sport"], row["game_id"])
        for _, row in matchups.iterrows()
    }
    selection = st.selectbox("Select a game", list(options.keys()))
    sel_sport, sel_game_id = options[selection]
    movement_df = load_line_movement(CLI.db, sel_sport, sel_game_id)
    if movement_df.empty:
        st.caption(
            f"No `odds_snapshots` rows for {sel_sport} game {sel_game_id}. "
            "The chart loads but is empty until snapshots are recorded."
        )
    html = render_line_movement(movement_df, title=selection)
    st.components.v1.html(html, height=380)


if __name__ == "__main__":
    main()
