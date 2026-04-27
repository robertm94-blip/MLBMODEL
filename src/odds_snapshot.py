"""Shared in-memory state for the Shadow Maker bot.

The bookmaker scraper writes to an OddsSnapshot every 2s; the Kalshi
quoting loop reads from it. Coordination is via per-game asyncio.Event
so the quoting loop reacts only when something changes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional


GameKey = tuple[str, str, str]  # (date_iso, away_abbr, home_abbr)


@dataclass
class MLBOdds:
    """Raw moneyline snapshot for a single MLB game from Bookmaker.eu."""
    away_team: str
    home_team: str
    away_ml: int
    home_ml: int
    start_time: str
    fetched_at: float


@dataclass
class GameState:
    """Per-game shared state — odds plus the change signal."""
    odds: MLBOdds
    away_fair_prob: float
    home_fair_prob: float
    change_event: asyncio.Event = field(default_factory=asyncio.Event)

    def update(self, odds: MLBOdds, away_fp: float, home_fp: float) -> bool:
        """Atomic update. Returns True if fair probs changed."""
        changed = (away_fp != self.away_fair_prob) or (home_fp != self.home_fair_prob)
        self.odds = odds
        self.away_fair_prob = away_fp
        self.home_fair_prob = home_fp
        if changed:
            self.change_event.set()
        return changed


@dataclass
class OddsSnapshot:
    """Full set of game states keyed by (date, away, home)."""
    games: dict[GameKey, GameState] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def get(self, key: GameKey) -> Optional[GameState]:
        return self.games.get(key)

    def ensure(self, key: GameKey, odds: MLBOdds, away_fp: float, home_fp: float) -> GameState:
        """Insert or update a game; returns the GameState."""
        state = self.games.get(key)
        if state is None:
            state = GameState(odds=odds, away_fair_prob=away_fp, home_fair_prob=home_fp)
            state.change_event.set()  # initial — trigger first quote
            self.games[key] = state
        else:
            state.update(odds, away_fp, home_fp)
        return state
