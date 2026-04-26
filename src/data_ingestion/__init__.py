"""Sport data ingestion package.

The interface lives in `base.py` (DataSource ABC + record types). Concrete
sources live in `mlb_stats.py`, `nba_stats.py`, etc. Persistence lives in
`storage.py` (SQLiteStore). The CLI lives in `cli.py`.
"""

from src.data_ingestion.base import (
    BoxscoreEntry,
    DataSource,
    FetchResult,
    Game,
    LineupEntry,
    Player,
    Team,
    Venue,
)

__all__ = [
    "BoxscoreEntry",
    "DataSource",
    "FetchResult",
    "Game",
    "LineupEntry",
    "Player",
    "Team",
    "Venue",
]
