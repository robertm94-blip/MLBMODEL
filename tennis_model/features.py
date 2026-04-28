"""Strictly-causal feature engineering for tennis matches.

Every feature for a given match is computed using *only* information available
strictly before that match's date. The pattern is: process matches in time
order, emit features for the current match using running state, then update
the state with the actual outcome.

Key features:
  * Elo (overall + per-surface, blended)
  * Rank / log-rank / points difference
  * Recent form windows (5/10/20 matches, surface-specific)
  * Head-to-head record (overall + surface)
  * Rest days
  * Career surface win%
  * Sackmann serve/return rolling averages (when available)
  * Tournament level / round / best_of one-hots
  * Implied market probability from odds (devigged)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SURFACES = ("Hard", "Clay", "Grass", "Carpet")


# Module-level factories so PlayerState is picklable (lambdas are not).
def _form_deque() -> deque:
    return deque(maxlen=20)


def _surface_form_factory() -> dict:
    # plain dict; we initialize entries lazily in update()
    return {}


def _int_dict() -> dict:
    return {}


@dataclass
class FeatureConfig:
    k_initial: float = 32.0
    k_min: float = 12.0
    k_decay: float = 0.0015
    surface_blend: float = 0.65
    prior_rating: float = 1500.0
    recent_form_windows: tuple[int, ...] = (5, 10, 20)
    rest_cap_days: int = 21
    min_career_matches: int = 10


@dataclass
class PlayerState:
    elo_overall: float
    elo_by_surface: dict[str, float]
    matches_played: int = 0
    matches_by_surface: dict[str, int] = field(default_factory=_int_dict)
    wins_by_surface: dict[str, int] = field(default_factory=_int_dict)
    last_match_date: pd.Timestamp | None = None
    # Rolling form: deque of 1/0 outcomes per surface and overall.
    form_overall: deque = field(default_factory=_form_deque)
    form_by_surface: dict[str, deque] = field(default_factory=_surface_form_factory)
    # Rolling serve stats (Sackmann).
    serve_pts_won: deque = field(default_factory=_form_deque)
    return_pts_won: deque = field(default_factory=_form_deque)


@dataclass
class FeatureState:
    """All cross-match running state. Pickleable for incremental inference."""
    cfg: FeatureConfig
    players: dict[str, PlayerState] = field(default_factory=dict)
    h2h: dict[tuple[str, str], list[int]] = field(default_factory=dict)  # winners count -> [p1_wins, p2_wins]
    h2h_by_surface: dict[tuple[str, str, str], list[int]] = field(default_factory=dict)
    last_processed_date: pd.Timestamp | None = None

    def _player(self, name: str) -> PlayerState:
        ps = self.players.get(name)
        if ps is None:
            ps = PlayerState(
                elo_overall=self.cfg.prior_rating,
                elo_by_surface={s: self.cfg.prior_rating for s in SURFACES},
            )
            self.players[name] = ps
        return ps

    def _k(self, matches_played: int) -> float:
        # Decaying K-factor: more uncertainty about new players, less about veterans.
        k = self.cfg.k_initial * math.exp(-self.cfg.k_decay * matches_played)
        return max(k, self.cfg.k_min)

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def _h2h_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def get_h2h(self, a: str, b: str, surface: str | None = None) -> tuple[int, int]:
        """Return (a_wins, b_wins) head-to-head before the current match."""
        key = self._h2h_key(a, b)
        if surface is None:
            counts = self.h2h.get(key, [0, 0])
        else:
            counts = self.h2h_by_surface.get((surface, *key), [0, 0])
        # Counts are ordered by sorted player names; remap if needed.
        if a < b:
            return counts[0], counts[1]
        return counts[1], counts[0]

    def update(self, winner: str, loser: str, surface: str, date: pd.Timestamp,
               serve_stats: dict | None = None) -> None:
        """Apply the outcome of a match to the running state."""
        w = self._player(winner)
        l = self._player(loser)
        surface = surface if surface in SURFACES else "Hard"

        # Elo update - blended surface + overall, but we keep them stored separately
        # and update each. The "rating" used for prediction is the blend.
        k_w = self._k(w.matches_played)
        k_l = self._k(l.matches_played)

        ew_o = self._expected(w.elo_overall, l.elo_overall)
        w.elo_overall += k_w * (1 - ew_o)
        l.elo_overall += k_l * (0 - (1 - ew_o))

        ew_s = self._expected(w.elo_by_surface[surface], l.elo_by_surface[surface])
        w.elo_by_surface[surface] += k_w * (1 - ew_s)
        l.elo_by_surface[surface] += k_l * (0 - (1 - ew_s))

        # Counts (plain dicts; init lazily)
        w.matches_played += 1
        l.matches_played += 1
        w.matches_by_surface[surface] = w.matches_by_surface.get(surface, 0) + 1
        l.matches_by_surface[surface] = l.matches_by_surface.get(surface, 0) + 1
        w.wins_by_surface[surface] = w.wins_by_surface.get(surface, 0) + 1
        l.wins_by_surface.setdefault(surface, 0)
        w.last_match_date = date
        l.last_match_date = date

        # Form
        w.form_overall.append(1)
        l.form_overall.append(0)
        w.form_by_surface.setdefault(surface, deque(maxlen=20)).append(1)
        l.form_by_surface.setdefault(surface, deque(maxlen=20)).append(0)

        # Serve stats
        if serve_stats is not None:
            ws = serve_stats.get("w_serve_pts_won_pct")
            wr = serve_stats.get("w_return_pts_won_pct")
            ls = serve_stats.get("l_serve_pts_won_pct")
            lr = serve_stats.get("l_return_pts_won_pct")
            if ws is not None and not math.isnan(ws):
                w.serve_pts_won.append(ws)
            if wr is not None and not math.isnan(wr):
                w.return_pts_won.append(wr)
            if ls is not None and not math.isnan(ls):
                l.serve_pts_won.append(ls)
            if lr is not None and not math.isnan(lr):
                l.return_pts_won.append(lr)

        # H2H
        key = self._h2h_key(winner, loser)
        if key not in self.h2h:
            self.h2h[key] = [0, 0]
        self.h2h[key][0 if winner == key[0] else 1] += 1

        skey = (surface, *key)
        if skey not in self.h2h_by_surface:
            self.h2h_by_surface[skey] = [0, 0]
        self.h2h_by_surface[skey][0 if winner == key[0] else 1] += 1

        self.last_processed_date = date

    # -- prediction-time accessors --

    def blended_elo(self, name: str, surface: str) -> tuple[float, float, float]:
        """Return (overall, surface, blended) elo for a player."""
        ps = self._player(name)
        surface = surface if surface in SURFACES else "Hard"
        s = ps.elo_by_surface.get(surface, self.cfg.prior_rating)
        o = ps.elo_overall
        b = self.cfg.surface_blend * s + (1 - self.cfg.surface_blend) * o
        return o, s, b

    def form_pct(self, name: str, window: int, surface: str | None = None) -> float | None:
        ps = self._player(name)
        if surface is None:
            seq = list(ps.form_overall)[-window:]
        else:
            surface = surface if surface in SURFACES else "Hard"
            seq = list(ps.form_by_surface.get(surface, deque()))[-window:]
        if len(seq) < max(3, window // 4):  # need a minimum sample
            return None
        return float(np.mean(seq))

    def career_surface_winpct(self, name: str, surface: str) -> float | None:
        ps = self._player(name)
        surface = surface if surface in SURFACES else "Hard"
        n = ps.matches_by_surface.get(surface, 0)
        if n < self.cfg.min_career_matches:
            return None
        return ps.wins_by_surface.get(surface, 0) / n

    def days_since_last(self, name: str, date: pd.Timestamp) -> float:
        ps = self._player(name)
        if ps.last_match_date is None:
            return float(self.cfg.rest_cap_days)
        d = (date - ps.last_match_date).days
        if d < 0:
            d = 0
        return float(min(d, self.cfg.rest_cap_days))

    def serve_avg(self, name: str) -> tuple[float | None, float | None]:
        ps = self._player(name)
        s = float(np.mean(ps.serve_pts_won)) if len(ps.serve_pts_won) >= 3 else None
        r = float(np.mean(ps.return_pts_won)) if len(ps.return_pts_won) >= 3 else None
        return s, r


def _serve_stats_from_row(row: pd.Series) -> dict | None:
    """Compute serve/return win % from Sackmann columns if present.

    Returns None if the columns aren't there or are NaN.
    """
    needed = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
    if not all(c in row.index and pd.notna(row[c]) for c in needed):
        return None
    w_pts = row["w_svpt"]
    l_pts = row["l_svpt"]
    if w_pts <= 0 or l_pts <= 0:
        return None
    w_serve_won = (row["w_1stWon"] + row["w_2ndWon"]) / w_pts
    l_serve_won = (row["l_1stWon"] + row["l_2ndWon"]) / l_pts
    # Return win % = 1 - opponent's serve win %.
    return {
        "w_serve_pts_won_pct": w_serve_won,
        "w_return_pts_won_pct": 1 - l_serve_won,
        "l_serve_pts_won_pct": l_serve_won,
        "l_return_pts_won_pct": 1 - w_serve_won,
    }


def _devig_two_way(p1_odds: float, p2_odds: float) -> tuple[float, float]:
    """Convert decimal odds to a normalized two-way probability (removes overround)."""
    inv1, inv2 = 1.0 / p1_odds, 1.0 / p2_odds
    s = inv1 + inv2
    if s <= 0:
        return 0.5, 0.5
    return inv1 / s, inv2 / s


def _level_bucket(level: str | None) -> str:
    """Coarse bucket for tournament 'Series'/'Tier'. Stable across ATP and WTA."""
    if not isinstance(level, str):
        return "Other"
    s = level.strip().lower()
    if "grand slam" in s:
        return "Slam"
    if "masters" in s and "1000" in s:
        return "Masters1000"
    if "premier mandatory" in s or "wta 1000" in s:
        return "WTA1000"
    if "premier 5" in s or "wta 500" in s:
        return "WTA500"
    if "international gold" in s or "atp500" in s or "atp 500" in s:
        return "ATP500"
    if "international" in s or "atp250" in s or "atp 250" in s or "wta 250" in s:
        return "Tour250"
    if "challenger" in s:
        return "Challenger"
    return "Other"


def _round_bucket(rnd: str | None) -> int:
    """Map round labels to an integer (later rounds = higher)."""
    if not isinstance(rnd, str):
        return 0
    s = rnd.strip().lower()
    table = {
        "round robin": 1, "1st round": 1, "1r": 1,
        "2nd round": 2, "2r": 2,
        "3rd round": 3, "3r": 3,
        "4th round": 4, "4r": 4,
        "quarterfinals": 5, "qf": 5,
        "semifinals": 6, "sf": 6,
        "the final": 7, "final": 7, "f": 7,
    }
    return table.get(s, 0)


def build_features(p1p2: pd.DataFrame, td_full: pd.DataFrame, cfg: FeatureConfig
                   ) -> tuple[pd.DataFrame, FeatureState]:
    """Compute features for every match in chronological order.

    Args:
        p1p2: output of data_ingestion.matches_to_p1p2 (one row per match,
              with player1/player2 randomized and label y).
        td_full: the original winner/loser frame in the same order, used
              to update Elo state from the *true* outcome.
        cfg: FeatureConfig.

    Returns:
        (features_df, final_state) where features_df has one row per match
        with all model inputs and the label, in chronological order.
    """
    state = FeatureState(cfg=cfg)
    # We iterate the canonical (winner/loser) frame to update state, but we emit
    # features keyed to the corresponding p1/p2 row.
    assert len(p1p2) == len(td_full), "p1p2 and td_full must align row-for-row"

    rows: list[dict] = []

    for i in range(len(p1p2)):
        m = p1p2.iloc[i]
        wm = td_full.iloc[i]
        date = pd.Timestamp(m["date"])
        surface = m["surface"] if isinstance(m["surface"], str) else "Hard"
        p1, p2 = m["player1"], m["player2"]

        # ----- read state BEFORE this match -----
        p1_eo, p1_es, p1_eb = state.blended_elo(p1, surface)
        p2_eo, p2_es, p2_eb = state.blended_elo(p2, surface)

        feat: dict = {
            "date": date,
            "tour": m["tour"],
            "surface": surface,
            "level_bucket": _level_bucket(m["level"]),
            "round_num": _round_bucket(m["round"]),
            "best_of": float(m["best_of"]) if pd.notna(m["best_of"]) else 3.0,

            "elo_overall_diff": p1_eo - p2_eo,
            "elo_surface_diff": p1_es - p2_es,
            "elo_blend_diff": p1_eb - p2_eb,
            "elo_blend_p1": p1_eb,
            "elo_blend_p2": p2_eb,

            "rank_p1": float(m["p1_rank"]) if pd.notna(m["p1_rank"]) else 500.0,
            "rank_p2": float(m["p2_rank"]) if pd.notna(m["p2_rank"]) else 500.0,
            "pts_p1": float(m["p1_pts"]) if pd.notna(m["p1_pts"]) else 0.0,
            "pts_p2": float(m["p2_pts"]) if pd.notna(m["p2_pts"]) else 0.0,
        }
        feat["rank_diff"] = feat["rank_p1"] - feat["rank_p2"]
        feat["log_rank_ratio"] = np.log(feat["rank_p2"] + 1) - np.log(feat["rank_p1"] + 1)
        feat["pts_diff"] = feat["pts_p1"] - feat["pts_p2"]
        feat["log_pts_ratio"] = np.log(feat["pts_p1"] + 1) - np.log(feat["pts_p2"] + 1)

        # Form
        for w in cfg.recent_form_windows:
            f1 = state.form_pct(p1, w)
            f2 = state.form_pct(p2, w)
            feat[f"form_overall_{w}_diff"] = (f1 - f2) if (f1 is not None and f2 is not None) else 0.0
            f1s = state.form_pct(p1, w, surface)
            f2s = state.form_pct(p2, w, surface)
            feat[f"form_surface_{w}_diff"] = (f1s - f2s) if (f1s is not None and f2s is not None) else 0.0

        # Career surface win%
        c1 = state.career_surface_winpct(p1, surface)
        c2 = state.career_surface_winpct(p2, surface)
        feat["career_surface_winpct_diff"] = (c1 - c2) if (c1 is not None and c2 is not None) else 0.0

        # Rest
        r1 = state.days_since_last(p1, date)
        r2 = state.days_since_last(p2, date)
        feat["rest_days_p1"] = r1
        feat["rest_days_p2"] = r2
        feat["rest_diff"] = r1 - r2

        # H2H
        h1, h2 = state.get_h2h(p1, p2)
        feat["h2h_total"] = h1 + h2
        feat["h2h_p1_winpct"] = (h1 / (h1 + h2)) if (h1 + h2) > 0 else 0.5
        sh1, sh2 = state.get_h2h(p1, p2, surface)
        feat["h2h_surface_total"] = sh1 + sh2
        feat["h2h_surface_p1_winpct"] = (sh1 / (sh1 + sh2)) if (sh1 + sh2) > 0 else 0.5

        # Sackmann serve avgs
        s1_serve, s1_ret = state.serve_avg(p1)
        s2_serve, s2_ret = state.serve_avg(p2)
        feat["serve_winpct_diff"] = (s1_serve - s2_serve) if (s1_serve is not None and s2_serve is not None) else 0.0
        feat["return_winpct_diff"] = (s1_ret - s2_ret) if (s1_ret is not None and s2_ret is not None) else 0.0

        # Static-ish player features from Sackmann (these are the same row's
        # values, taken from the *match* row - they reflect age/height/hand at
        # that match, which are causally available pre-match).
        for col, default in [("p1_age", np.nan), ("p2_age", np.nan),
                             ("p1_ht", np.nan), ("p2_ht", np.nan)]:
            feat[col] = float(m[col]) if (col in m.index and pd.notna(m[col])) else np.nan
        feat["age_diff"] = feat["p1_age"] - feat["p2_age"] if not (np.isnan(feat["p1_age"]) or np.isnan(feat["p2_age"])) else 0.0
        feat["height_diff"] = feat["p1_ht"] - feat["p2_ht"] if not (np.isnan(feat["p1_ht"]) or np.isnan(feat["p2_ht"])) else 0.0
        # Handedness matchup: 1 if same, -1 if different, 0 if unknown
        h1 = m["p1_hand"] if "p1_hand" in m.index else None
        h2 = m["p2_hand"] if "p2_hand" in m.index else None
        feat["same_hand"] = 1.0 if (isinstance(h1, str) and isinstance(h2, str) and h1 == h2) else (-1.0 if (isinstance(h1, str) and isinstance(h2, str)) else 0.0)

        # Market implied probability (devigged).
        ip1, ip2 = _devig_two_way(float(m["p1_odds"]), float(m["p2_odds"]))
        feat["market_p1"] = ip1
        feat["market_p2"] = ip2
        # Raw odds carried through for the backtest stage but NOT used as model features
        # to keep the backtest clean - we'll concat them back at the end.
        feat["p1_odds"] = float(m["p1_odds"])
        feat["p2_odds"] = float(m["p2_odds"])
        feat["odds_source"] = m["odds_source"]
        feat["player1"] = p1
        feat["player2"] = p2

        # Label
        feat["y"] = int(m["y"])

        rows.append(feat)

        # ----- update state AFTER reading -----
        winner = wm["winner"]
        loser = wm["loser"]
        serve_stats = _serve_stats_from_row(wm)
        state.update(winner, loser, surface, date, serve_stats)

        if i % 25000 == 0 and i > 0:
            log.info("Built features for %d / %d matches", i, len(p1p2))

    feat_df = pd.DataFrame(rows)
    log.info("Feature build complete: %d rows, %d columns", len(feat_df), feat_df.shape[1])
    return feat_df, state


# Columns that are *features* fed to the model (excludes label, odds carry-throughs,
# names, raw market probabilities — we don't feed market_p1 to avoid the model
# trivially learning to copy the bookmaker, which would kill any edge).
FEATURE_COLUMNS = [
    "elo_overall_diff", "elo_surface_diff", "elo_blend_diff",
    "elo_blend_p1", "elo_blend_p2",
    "rank_p1", "rank_p2", "rank_diff", "log_rank_ratio",
    "pts_p1", "pts_p2", "pts_diff", "log_pts_ratio",
    "form_overall_5_diff", "form_overall_10_diff", "form_overall_20_diff",
    "form_surface_5_diff", "form_surface_10_diff", "form_surface_20_diff",
    "career_surface_winpct_diff",
    "rest_days_p1", "rest_days_p2", "rest_diff",
    "h2h_total", "h2h_p1_winpct", "h2h_surface_total", "h2h_surface_p1_winpct",
    "serve_winpct_diff", "return_winpct_diff",
    "p1_age", "p2_age", "age_diff", "p1_ht", "p2_ht", "height_diff", "same_hand",
    "round_num", "best_of",
]
# Categorical, will be one-hot encoded in the model layer.
CATEGORICAL_COLUMNS = ["tour", "surface", "level_bucket"]
