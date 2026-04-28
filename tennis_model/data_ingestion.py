"""Data ingestion for tennis-data.co.uk and Jeff Sackmann's repos.

The two sources are complementary:
  * tennis-data.co.uk has real bookmaker pre-match odds (Bet365, Pinnacle, average)
    which is what we need to compute backtest P&L. Coverage is best from ~2005.
  * Sackmann has richer match metadata (serve/return splits, IDs, ages, heights)
    but no odds. We merge on (date, player names) to enrich the odds rows.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

# Canonical column names produced by load_tennis_data().
CANON_COLUMNS = [
    "date", "tour", "tournament", "surface", "level", "round",
    "best_of", "winner", "loser", "w_rank", "l_rank", "w_pts", "l_pts",
    "w_odds", "l_odds", "odds_source",
    "comment",
]


@dataclass
class IngestionConfig:
    td_atp_url_template: str
    td_wta_url_template: str
    sackmann_atp_matches: str
    sackmann_wta_matches: str
    sackmann_atp_players: str
    sackmann_wta_players: str
    cache_dir: str
    start_year: int
    end_year: int
    tours: list[str]
    fallback_odds: list[str]


def _http_get(url: str, retries: int = 3, timeout: int = 30) -> bytes | None:
    """GET with simple exponential backoff. Returns None on 404 / final failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "tennis-model/0.1"})
            if r.status_code == 404:
                log.debug("404 for %s", url)
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("GET %s failed (%s); retrying in %ds", url, e, wait)
            time.sleep(wait)
    log.error("Giving up on %s after %d attempts", url, retries)
    return None


def _cached_download(url: str, cache_path: Path) -> bytes | None:
    """Download once, then read from disk. Empty cache files indicate prior 404s."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = cache_path.read_bytes()
        return data if len(data) > 0 else None
    blob = _http_get(url)
    cache_path.write_bytes(blob if blob is not None else b"")
    return blob


def _td_url(template: str, year: int) -> tuple[str, str]:
    """tennis-data.co.uk uses .xls before 2013 and .xlsx after."""
    ext = "xls" if year <= 2012 else "xlsx"
    return template.format(year=year, ext=ext), ext


def download_tennis_data(cfg: IngestionConfig) -> pd.DataFrame:
    """Pull tennis-data.co.uk yearly files and concatenate."""
    cache = Path(cfg.cache_dir) / "tennis_data"
    frames: list[pd.DataFrame] = []
    for tour in cfg.tours:
        template = cfg.td_atp_url_template if tour == "ATP" else cfg.td_wta_url_template
        for year in range(cfg.start_year, cfg.end_year + 1):
            url, ext = _td_url(template, year)
            cache_path = cache / f"{tour}_{year}.{ext}"
            blob = _cached_download(url, cache_path)
            if blob is None:
                # try the alternate extension once (some years are mis-published)
                alt_ext = "xlsx" if ext == "xls" else "xls"
                alt_url = template.format(year=year, ext=alt_ext)
                blob = _cached_download(alt_url, cache / f"{tour}_{year}.{alt_ext}")
            if blob is None:
                log.info("No tennis-data file for %s %d", tour, year)
                continue
            try:
                df = pd.read_excel(io.BytesIO(blob))
            except Exception as e:
                log.warning("Failed to parse %s %d: %s", tour, year, e)
                continue
            df["tour"] = tour
            df["source_year"] = year
            frames.append(df)
    if not frames:
        raise RuntimeError(
            "No tennis-data.co.uk files were downloaded. Check network access."
        )
    raw = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Loaded %d raw rows from tennis-data.co.uk", len(raw))
    return raw


def _pick_odds(row: pd.Series, sources: Iterable[str]) -> tuple[float | None, float | None, str | None]:
    """Pick the first available (winner_odds, loser_odds) pair."""
    for src in sources:
        wcol, lcol = src.split("_") if "_" in src else (src, src)
        # Map our config tags to actual columns
        wcol, lcol = _ODDS_MAP.get(src, (wcol, lcol))
        wo = row.get(wcol)
        lo = row.get(lcol)
        if pd.notna(wo) and pd.notna(lo) and wo > 1.0 and lo > 1.0:
            return float(wo), float(lo), src
    return None, None, None


_ODDS_MAP = {
    "PSW_PSL": ("PSW", "PSL"),       # Pinnacle
    "AvgW_AvgL": ("AvgW", "AvgL"),   # Bookmaker average
    "B365W_B365L": ("B365W", "B365L"),
    "MaxW_MaxL": ("MaxW", "MaxL"),
}


def standardize_tennis_data(raw: pd.DataFrame, fallback_odds: list[str]) -> pd.DataFrame:
    """Map tennis-data columns onto our canonical schema."""
    df = raw.copy()

    # Date column is "Date" across years.
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Names: 'Winner' and 'Loser' (e.g. 'Federer R.').
    df["winner"] = df["Winner"].astype(str).map(_normalize_name)
    df["loser"] = df["Loser"].astype(str).map(_normalize_name)

    df["tournament"] = df.get("Tournament")
    df["surface"] = df.get("Surface")
    df["level"] = df.get("Series", df.get("Tier"))   # ATP uses Series, WTA uses Tier
    df["round"] = df.get("Round")
    df["best_of"] = pd.to_numeric(df.get("Best of"), errors="coerce")

    # Ranks / points are "WRank", "LRank", "WPts", "LPts".
    df["w_rank"] = pd.to_numeric(df.get("WRank"), errors="coerce")
    df["l_rank"] = pd.to_numeric(df.get("LRank"), errors="coerce")
    df["w_pts"] = pd.to_numeric(df.get("WPts"), errors="coerce")
    df["l_pts"] = pd.to_numeric(df.get("LPts"), errors="coerce")

    # Odds: pick first available source for every row.
    odds = df.apply(lambda r: _pick_odds(r, fallback_odds), axis=1, result_type="expand")
    odds.columns = ["w_odds", "l_odds", "odds_source"]
    df[["w_odds", "l_odds", "odds_source"]] = odds

    # Comment includes "Completed", "Retired", "Walkover".
    df["comment"] = df.get("Comment", "Completed").fillna("Completed")

    out = df[CANON_COLUMNS].copy()
    out = out.dropna(subset=["date", "winner", "loser"])
    # Drop retirements/walkovers - the result is recorded but the match wasn't fully
    # contested, so the implied skill comparison is noisy.
    out = out[out["comment"].str.lower().str.startswith(("completed", "comp"))].copy()
    out = out.sort_values("date").reset_index(drop=True)
    log.info("Standardized to %d completed matches with names+date", len(out))
    return out


def _normalize_name(name: str) -> str:
    """Strip accents and standardize 'Lastname F.' style names for fuzzy joins."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def download_sackmann(cfg: IngestionConfig) -> dict[str, pd.DataFrame]:
    """Pull Sackmann match + player CSVs into a dict keyed by tour."""
    cache = Path(cfg.cache_dir) / "sackmann"
    out: dict[str, pd.DataFrame] = {}
    for tour in cfg.tours:
        match_template = cfg.sackmann_atp_matches if tour == "ATP" else cfg.sackmann_wta_matches
        player_url = cfg.sackmann_atp_players if tour == "ATP" else cfg.sackmann_wta_players

        # Players file (one-shot)
        players_blob = _cached_download(player_url, cache / f"{tour}_players.csv")
        players = (
            pd.read_csv(io.BytesIO(players_blob), low_memory=False)
            if players_blob
            else pd.DataFrame()
        )

        match_frames: list[pd.DataFrame] = []
        for year in range(cfg.start_year, cfg.end_year + 1):
            blob = _cached_download(
                match_template.format(year=year), cache / f"{tour}_matches_{year}.csv"
            )
            if blob is None:
                continue
            try:
                m = pd.read_csv(io.BytesIO(blob), low_memory=False)
                match_frames.append(m)
            except Exception as e:
                log.warning("Sackmann %s %d unreadable: %s", tour, year, e)
        matches = (
            pd.concat(match_frames, ignore_index=True, sort=False)
            if match_frames
            else pd.DataFrame()
        )
        if not matches.empty:
            matches["tourney_date"] = pd.to_datetime(
                matches["tourney_date"], format="%Y%m%d", errors="coerce"
            )
        out[tour] = matches
        out[f"{tour}_players"] = players
        log.info("Sackmann %s: %d matches, %d players", tour, len(matches), len(players))
    return out


def _sackmann_lastname_initial(full_name: str) -> str:
    """Convert 'Roger Federer' -> 'federer r.' to match tennis-data style."""
    if not isinstance(full_name, str):
        return ""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return _normalize_name(full_name)
    first = parts[0]
    last = " ".join(parts[1:])
    return _normalize_name(f"{last} {first[0]}.")


def merge_with_sackmann(td: pd.DataFrame, sack: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join Sackmann match metadata onto tennis-data rows where we can.

    The match key is (tour, date, normalized_winner, normalized_loser). Sackmann
    names are 'Roger Federer'; we convert to 'federer r.' to match tennis-data.
    Many rows won't match (different tournament inclusion, name variants), but
    we only need *partial* enrichment - the core features come from the odds
    file, and Sackmann adds bonus features when available.
    """
    enriched_frames: list[pd.DataFrame] = []
    for tour in td["tour"].unique():
        td_t = td[td["tour"] == tour].copy()
        s = sack.get(tour, pd.DataFrame())
        if s.empty:
            enriched_frames.append(td_t)
            continue
        s = s.copy()
        s["winner_key"] = s["winner_name"].map(_sackmann_lastname_initial)
        s["loser_key"] = s["loser_name"].map(_sackmann_lastname_initial)
        s["date"] = s["tourney_date"]

        keep_cols = [
            "date", "winner_key", "loser_key",
            "winner_id", "loser_id", "winner_age", "loser_age",
            "winner_ht", "loser_ht", "winner_hand", "loser_hand",
            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
            "w_SvGms", "w_bpSaved", "w_bpFaced",
            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
            "l_SvGms", "l_bpSaved", "l_bpFaced",
            "minutes",
        ]
        keep_cols = [c for c in keep_cols if c in s.columns]
        s_small = s[keep_cols].copy()

        td_t["winner_key"] = td_t["winner"]
        td_t["loser_key"] = td_t["loser"]
        merged = td_t.merge(s_small, on=["date", "winner_key", "loser_key"], how="left")
        merged.drop(columns=["winner_key", "loser_key"], inplace=True)
        enriched_frames.append(merged)
    out = pd.concat(enriched_frames, ignore_index=True, sort=False)
    out = out.sort_values("date").reset_index(drop=True)

    # Diagnostic: how many rows got Sackmann enrichment?
    sack_hit = out["winner_id"].notna().mean() if "winner_id" in out.columns else 0.0
    log.info("Sackmann match rate: %.1f%% of rows", sack_hit * 100)
    return out


def load_all(cfg: IngestionConfig) -> pd.DataFrame:
    """End-to-end: download, standardize, merge. Returns one big DataFrame."""
    raw = download_tennis_data(cfg)
    td = standardize_tennis_data(raw, cfg.fallback_odds)
    sack = download_sackmann(cfg)
    full = merge_with_sackmann(td, sack)
    full = full.dropna(subset=["w_odds", "l_odds"]).reset_index(drop=True)
    log.info("Final dataset: %d rows with odds", len(full))
    return full


def matches_to_p1p2(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Re-shape winner/loser rows into player1/player2 with a label.

    We randomize the side assignment so the model can't trivially learn
    "the winner column always wins" through any leaked signal. Label is
    1 iff player1 won.
    """
    rng = np.random.default_rng(seed)
    flip = rng.random(len(df)) < 0.5

    out = pd.DataFrame({
        "date": df["date"].values,
        "tour": df["tour"].values,
        "tournament": df["tournament"].values,
        "surface": df["surface"].values,
        "level": df["level"].values,
        "round": df["round"].values,
        "best_of": df["best_of"].values,
    })
    out["player1"] = np.where(flip, df["loser"], df["winner"])
    out["player2"] = np.where(flip, df["winner"], df["loser"])
    out["p1_rank"] = np.where(flip, df["l_rank"], df["w_rank"])
    out["p2_rank"] = np.where(flip, df["w_rank"], df["l_rank"])
    out["p1_pts"] = np.where(flip, df["l_pts"], df["w_pts"])
    out["p2_pts"] = np.where(flip, df["w_pts"], df["l_pts"])
    out["p1_odds"] = np.where(flip, df["l_odds"], df["w_odds"])
    out["p2_odds"] = np.where(flip, df["w_odds"], df["l_odds"])
    out["odds_source"] = df["odds_source"].values
    out["y"] = (~flip).astype(int)   # 1 if player1 (= original winner when not flipped) wins

    # Carry through Sackmann columns if present, swapping w_/l_ to p1_/p2_ semantics.
    sack_pairs = {
        "winner_age": "loser_age",
        "winner_ht": "loser_ht",
        "winner_hand": "loser_hand",
        "winner_id": "loser_id",
        "w_ace": "l_ace", "w_df": "l_df", "w_svpt": "l_svpt",
        "w_1stIn": "l_1stIn", "w_1stWon": "l_1stWon", "w_2ndWon": "l_2ndWon",
        "w_SvGms": "l_SvGms", "w_bpSaved": "l_bpSaved", "w_bpFaced": "l_bpFaced",
    }
    for wcol, lcol in sack_pairs.items():
        if wcol in df.columns and lcol in df.columns:
            p1_col = wcol.replace("winner", "p1").replace("w_", "p1_")
            p2_col = lcol.replace("loser", "p2").replace("l_", "p2_")
            out[p1_col] = np.where(flip, df[lcol], df[wcol])
            out[p2_col] = np.where(flip, df[wcol], df[lcol])

    return out
