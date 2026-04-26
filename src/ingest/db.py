"""SQLite schema and write helpers for the ingestion module."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

DEFAULT_DB_PATH = os.path.join("data", "ingest.db")


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS games (
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        game_date TEXT NOT NULL,
        status TEXT,
        home_team_id TEXT,
        home_team_name TEXT,
        away_team_id TEXT,
        away_team_name TEXT,
        venue TEXT,
        home_score INTEGER,
        away_score INTEGER,
        raw_json TEXT,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, game_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineups (
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        team_id TEXT NOT NULL,
        batting_order INTEGER,
        starter_slot INTEGER,
        player_id TEXT NOT NULL,
        player_name TEXT,
        position TEXT,
        source TEXT,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, game_id, team_id, player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS box_scores (
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        team_id TEXT NOT NULL,
        player_id TEXT NOT NULL,
        stats_json TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, game_id, team_id, player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sport TEXT,
        table_name TEXT,
        target_date TEXT,
        rows_inserted INTEGER,
        rows_updated INTEGER,
        started_at TEXT,
        finished_at TEXT,
        error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team_features (
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        team_id TEXT NOT NULL,
        side TEXT NOT NULL,
        game_date TEXT NOT NULL,
        venue_id TEXT,
        team_name TEXT,
        opponent_team_id TEXT,
        opponent_team_name TEXT,

        lineup_wrc_plus REAL,
        offensive_factor REAL,
        platoon_factor REAL,
        lineup_size INTEGER,
        lineup_source TEXT,

        starter_player_id TEXT,
        starter_name TEXT,
        starter_hand TEXT,
        starter_fip REAL,
        starter_era REAL,
        starter_projected_ip REAL,
        starter_systems_count INTEGER,
        bullpen_fip REAL,
        bullpen_era REAL,
        bullpen_total_ip REAL,
        starter_share REAL,
        pitching_blended_fip REAL,

        is_opener INTEGER NOT NULL,
        is_bullpen_heavy INTEGER NOT NULL,

        park_runs_factor REAL,

        raw_json TEXT,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, game_id, team_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_games_date ON games (sport, game_date)",
    "CREATE INDEX IF NOT EXISTS idx_lineups_game ON lineups (sport, game_id)",
    "CREATE INDEX IF NOT EXISTS idx_box_game ON box_scores (sport, game_id)",
    "CREATE INDEX IF NOT EXISTS idx_team_features_date ON team_features (sport, game_date)",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection and ensure the schema is applied."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    for stmt in SCHEMA:
        conn.execute(stmt)
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def upsert_game(conn: sqlite3.Connection, game: dict[str, Any]) -> int:
    """Upsert a row in `games`. Returns 1 if inserted, 0 if updated."""
    sport = game["sport"]
    game_id = str(game["game_id"])
    fetched = now_iso()
    cur = conn.execute("SELECT 1 FROM games WHERE sport=? AND game_id=?", (sport, game_id))
    existed = cur.fetchone() is not None
    conn.execute(
        """
        INSERT INTO games (
            sport, game_id, game_date, status,
            home_team_id, home_team_name, away_team_id, away_team_name,
            venue, home_score, away_score, raw_json, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sport, game_id) DO UPDATE SET
            game_date = excluded.game_date,
            status = excluded.status,
            home_team_id = excluded.home_team_id,
            home_team_name = excluded.home_team_name,
            away_team_id = excluded.away_team_id,
            away_team_name = excluded.away_team_name,
            venue = excluded.venue,
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            raw_json = excluded.raw_json,
            fetched_at = excluded.fetched_at
        """,
        (
            sport,
            game_id,
            game["game_date"],
            game.get("status"),
            _str_or_none(game.get("home_team_id")),
            game.get("home_team_name"),
            _str_or_none(game.get("away_team_id")),
            game.get("away_team_name"),
            game.get("venue"),
            game.get("home_score"),
            game.get("away_score"),
            json.dumps(game.get("raw")) if game.get("raw") is not None else None,
            fetched,
        ),
    )
    return 0 if existed else 1


def upsert_lineup_rows(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    inserted = updated = 0
    fetched = now_iso()
    for row in rows:
        key = (row["sport"], str(row["game_id"]), str(row["team_id"]), str(row["player_id"]))
        cur = conn.execute(
            "SELECT 1 FROM lineups WHERE sport=? AND game_id=? AND team_id=? AND player_id=?",
            key,
        )
        existed = cur.fetchone() is not None
        conn.execute(
            """
            INSERT INTO lineups (
                sport, game_id, team_id, batting_order, starter_slot,
                player_id, player_name, position, source, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sport, game_id, team_id, player_id) DO UPDATE SET
                batting_order = excluded.batting_order,
                starter_slot = excluded.starter_slot,
                player_name = excluded.player_name,
                position = excluded.position,
                source = excluded.source,
                fetched_at = excluded.fetched_at
            """,
            (
                row["sport"],
                str(row["game_id"]),
                str(row["team_id"]),
                row.get("batting_order"),
                row.get("starter_slot"),
                str(row["player_id"]),
                row.get("player_name"),
                row.get("position"),
                row.get("source"),
                fetched,
            ),
        )
        if existed:
            updated += 1
        else:
            inserted += 1
    return inserted, updated


def upsert_boxscore_rows(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    inserted = updated = 0
    fetched = now_iso()
    for row in rows:
        key = (row["sport"], str(row["game_id"]), str(row["team_id"]), str(row["player_id"]))
        cur = conn.execute(
            "SELECT 1 FROM box_scores WHERE sport=? AND game_id=? AND team_id=? AND player_id=?",
            key,
        )
        existed = cur.fetchone() is not None
        conn.execute(
            """
            INSERT INTO box_scores (
                sport, game_id, team_id, player_id, stats_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sport, game_id, team_id, player_id) DO UPDATE SET
                stats_json = excluded.stats_json,
                fetched_at = excluded.fetched_at
            """,
            (
                row["sport"],
                str(row["game_id"]),
                str(row["team_id"]),
                str(row["player_id"]),
                json.dumps(row["stats"]),
                fetched,
            ),
        )
        if existed:
            updated += 1
        else:
            inserted += 1
    return inserted, updated


def record_run_start(conn: sqlite3.Connection, sport: str, table_name: str, target_date: str | None) -> int:
    cur = conn.execute(
        """
        INSERT INTO ingest_runs (sport, table_name, target_date, rows_inserted, rows_updated, started_at)
        VALUES (?, ?, ?, 0, 0, ?)
        """,
        (sport, table_name, target_date, now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def record_run_finish(
    conn: sqlite3.Connection,
    run_id: int,
    rows_inserted: int,
    rows_updated: int,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE ingest_runs
        SET rows_inserted=?, rows_updated=?, finished_at=?, error=?
        WHERE id=?
        """,
        (rows_inserted, rows_updated, now_iso(), error, run_id),
    )
    conn.commit()


def date_already_ingested(
    conn: sqlite3.Connection, sport: str, table_name: str, target_date: str
) -> bool:
    cur = conn.execute(
        """
        SELECT 1 FROM ingest_runs
        WHERE sport=? AND table_name=? AND target_date=?
          AND finished_at IS NOT NULL AND error IS NULL
        LIMIT 1
        """,
        (sport, table_name, target_date),
    )
    return cur.fetchone() is not None


def upsert_team_features(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    """Upsert a row into `team_features`. Returns 1 if inserted, 0 if updated."""
    sport = row["sport"]
    game_id = str(row["game_id"])
    team_id = str(row["team_id"])
    fetched = now_iso()
    cur = conn.execute(
        "SELECT 1 FROM team_features WHERE sport=? AND game_id=? AND team_id=?",
        (sport, game_id, team_id),
    )
    existed = cur.fetchone() is not None
    conn.execute(
        """
        INSERT INTO team_features (
            sport, game_id, team_id, side, game_date, venue_id,
            team_name, opponent_team_id, opponent_team_name,
            lineup_wrc_plus, offensive_factor, platoon_factor,
            lineup_size, lineup_source,
            starter_player_id, starter_name, starter_hand,
            starter_fip, starter_era, starter_projected_ip, starter_systems_count,
            bullpen_fip, bullpen_era, bullpen_total_ip,
            starter_share, pitching_blended_fip,
            is_opener, is_bullpen_heavy,
            park_runs_factor,
            raw_json, fetched_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?
        )
        ON CONFLICT(sport, game_id, team_id) DO UPDATE SET
            side = excluded.side,
            game_date = excluded.game_date,
            venue_id = excluded.venue_id,
            team_name = excluded.team_name,
            opponent_team_id = excluded.opponent_team_id,
            opponent_team_name = excluded.opponent_team_name,
            lineup_wrc_plus = excluded.lineup_wrc_plus,
            offensive_factor = excluded.offensive_factor,
            platoon_factor = excluded.platoon_factor,
            lineup_size = excluded.lineup_size,
            lineup_source = excluded.lineup_source,
            starter_player_id = excluded.starter_player_id,
            starter_name = excluded.starter_name,
            starter_hand = excluded.starter_hand,
            starter_fip = excluded.starter_fip,
            starter_era = excluded.starter_era,
            starter_projected_ip = excluded.starter_projected_ip,
            starter_systems_count = excluded.starter_systems_count,
            bullpen_fip = excluded.bullpen_fip,
            bullpen_era = excluded.bullpen_era,
            bullpen_total_ip = excluded.bullpen_total_ip,
            starter_share = excluded.starter_share,
            pitching_blended_fip = excluded.pitching_blended_fip,
            is_opener = excluded.is_opener,
            is_bullpen_heavy = excluded.is_bullpen_heavy,
            park_runs_factor = excluded.park_runs_factor,
            raw_json = excluded.raw_json,
            fetched_at = excluded.fetched_at
        """,
        (
            sport, game_id, team_id, row["side"], row["game_date"],
            _str_or_none(row.get("venue_id")),
            row.get("team_name"),
            _str_or_none(row.get("opponent_team_id")),
            row.get("opponent_team_name"),
            row.get("lineup_wrc_plus"),
            row.get("offensive_factor"),
            row.get("platoon_factor"),
            row.get("lineup_size"),
            row.get("lineup_source"),
            _str_or_none(row.get("starter_player_id")),
            row.get("starter_name"),
            row.get("starter_hand"),
            row.get("starter_fip"),
            row.get("starter_era"),
            row.get("starter_projected_ip"),
            row.get("starter_systems_count"),
            row.get("bullpen_fip"),
            row.get("bullpen_era"),
            row.get("bullpen_total_ip"),
            row.get("starter_share"),
            row.get("pitching_blended_fip"),
            1 if row.get("is_opener") else 0,
            1 if row.get("is_bullpen_heavy") else 0,
            row.get("park_runs_factor"),
            json.dumps(row.get("raw")) if row.get("raw") is not None else None,
            fetched,
        ),
    )
    return 0 if existed else 1


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
