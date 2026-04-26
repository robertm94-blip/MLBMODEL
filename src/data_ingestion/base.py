"""Interface contract for sport data sources.

Every concrete source (MLB Stats API, NBA Stats API, future Statcast, etc.)
implements `DataSource` and returns the normalized record types defined here.
The storage layer and CLI consume `DataSource` only — they never touch the
upstream HTTP shape directly.

This file is the **public contract**: changing it is a breaking change for
every implementation and consumer. Per-source files are free to evolve.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


# =====================================================================
# Normalized record types
# =====================================================================

@dataclass(frozen=True)
class Team:
    """A team in a given league."""
    sport: str
    team_id: str
    team_name: str | None = None
    team_abbr: str | None = None


@dataclass(frozen=True)
class Player:
    """A player in a given league."""
    sport: str
    player_id: str
    player_name: str | None = None
    primary_position: str | None = None


@dataclass(frozen=True)
class Venue:
    """A venue (stadium/arena)."""
    sport: str
    venue_id: str
    venue_name: str | None = None


@dataclass(frozen=True)
class Game:
    """A scheduled or completed game."""
    sport: str
    game_id: str
    game_date: str                       # YYYY-MM-DD
    status: str | None
    home_team: Team
    away_team: Team
    venue: Venue | None = None
    home_score: int | None = None
    away_score: int | None = None
    raw: dict[str, Any] | None = None    # original upstream payload


@dataclass(frozen=True)
class LineupEntry:
    """One starting-lineup slot. Either `batting_order` (MLB) or
    `starter_slot` (NBA) is populated, never both."""
    sport: str
    game_id: str
    team_id: str
    player: Player
    batting_order: int | None = None     # MLB 1..9
    starter_slot: int | None = None      # NBA 1..5
    position: str | None = None          # game-specific (may differ from primary)
    source: str | None = None            # 'official' | 'projected' | 'roster'


@dataclass(frozen=True)
class BoxscoreEntry:
    """One player's per-game stat line."""
    sport: str
    game_id: str
    team_id: str
    player: Player
    stats: dict[str, Any]                # source-specific stat dict
    position: str | None = None


@dataclass(frozen=True)
class FetchResult:
    """Bundle returned from a per-game fetch.

    `lineups` and `box_scores` may be empty lists if the upstream payload
    didn't contain that data yet (e.g. pre-game). `game` is the
    (optionally) refreshed Game with final scores once the game is over.
    """
    game: Game
    lineups: list[LineupEntry] = field(default_factory=list)
    box_scores: list[BoxscoreEntry] = field(default_factory=list)


# =====================================================================
# Interface
# =====================================================================

class DataSource(ABC):
    """Abstract base for any sport data source.

    Implementations are expected to be cheap to construct and stateful
    only to the extent that they hold an HTTP client. Concurrency safety
    is the implementation's responsibility.
    """

    sport: str  # 'mlb', 'nba', etc.

    # ---- discovery ----------------------------------------------------

    @abstractmethod
    def fetch_schedule(self, game_date: str) -> list[Game]:
        """Return the list of games scheduled on `game_date` (YYYY-MM-DD).

        Empty list if there are no games. Pre-game and in-progress games
        should still be returned so downstream callers can decide whether
        to fetch lineups / box scores.
        """

    # ---- per-game data ------------------------------------------------

    @abstractmethod
    def fetch_lineups(self, game: Game) -> list[LineupEntry]:
        """Return the starting lineup entries for both sides of `game`.

        Returns an empty list if no lineup is available yet (TBD lineup,
        pre-game, etc.). MUST NOT raise on missing data — return [].
        """

    @abstractmethod
    def fetch_boxscore(self, game: Game) -> FetchResult:
        """Return per-player box-score rows + an updated Game.

        The returned `FetchResult.game` should carry final `home_score` /
        `away_score` / `status` if the game is complete; for in-progress
        games these fields may be partial. Empty `box_scores` list is
        valid (e.g. game hasn't started).
        """

    # ---- convenience helpers (default impls; override if cheaper) -----

    def fetch_day(self, game_date: str) -> Iterable[FetchResult]:
        """Yield `FetchResult` for every game on `game_date`.

        Default: schedule -> per-game boxscore. Implementations that have
        a single endpoint covering an entire day can override.
        """
        for game in self.fetch_schedule(game_date):
            result = self.fetch_boxscore(game)
            # If the boxscore fetch didn't include lineups, get them separately.
            if not result.lineups:
                lineups = self.fetch_lineups(result.game)
                result = FetchResult(
                    game=result.game,
                    lineups=lineups,
                    box_scores=result.box_scores,
                )
            yield result


__all__ = [
    "Team",
    "Player",
    "Venue",
    "Game",
    "LineupEntry",
    "BoxscoreEntry",
    "FetchResult",
    "DataSource",
]
