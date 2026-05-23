"""SQLite persistence for ingested sport data.

Schema is normalized: `teams`, `players`, `venues` are reference tables;
`games`, `lineups`, `box_scores` reference them by FK. The CLI and feature
pipeline talk to `SQLiteStore` only — they don't open raw connections.

The `team_features` table is included here for convenience even though it
holds derived data (the feature pipeline writes it). Keeping it in the
same DB simplifies cross-table joins.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from src.data_ingestion.base import (
    BoxscoreEntry,
    Game,
    LineupEntry,
    Player,
    Team,
    Venue,
)

DEFAULT_DB_PATH = os.path.join("data", "data_ingestion.db")


SCHEMA: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS teams (
        sport TEXT NOT NULL,
        team_id TEXT NOT NULL,
        team_name TEXT,
        team_abbr TEXT,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, team_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS players (
        sport TEXT NOT NULL,
        player_id TEXT NOT NULL,
        player_name TEXT,
        primary_position TEXT,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS venues (
        sport TEXT NOT NULL,
        venue_id TEXT NOT NULL,
        venue_name TEXT,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, venue_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS games (
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        game_date TEXT NOT NULL,
        status TEXT,
        home_team_id TEXT,
        away_team_id TEXT,
        venue_id TEXT,
        home_score INTEGER,
        away_score INTEGER,
        raw_json TEXT,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, game_id),
        FOREIGN KEY (sport, home_team_id) REFERENCES teams(sport, team_id),
        FOREIGN KEY (sport, away_team_id) REFERENCES teams(sport, team_id),
        FOREIGN KEY (sport, venue_id)     REFERENCES venues(sport, venue_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineups (
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        team_id TEXT NOT NULL,
        player_id TEXT NOT NULL,
        batting_order INTEGER,
        starter_slot INTEGER,
        position TEXT,
        source TEXT,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, game_id, team_id, player_id),
        FOREIGN KEY (sport, game_id)   REFERENCES games(sport, game_id),
        FOREIGN KEY (sport, team_id)   REFERENCES teams(sport, team_id),
        FOREIGN KEY (sport, player_id) REFERENCES players(sport, player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS box_scores (
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        team_id TEXT NOT NULL,
        player_id TEXT NOT NULL,
        position TEXT,
        stats_json TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (sport, game_id, team_id, player_id),
        FOREIGN KEY (sport, game_id)   REFERENCES games(sport, game_id),
        FOREIGN KEY (sport, team_id)   REFERENCES teams(sport, team_id),
        FOREIGN KEY (sport, player_id) REFERENCES players(sport, player_id)
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
    """
    CREATE TABLE IF NOT EXISTS odds_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        market TEXT NOT NULL,                -- 'moneyline' | 'total' | 'runline' | 'spread'
        side TEXT NOT NULL,                  -- 'home'|'away' (moneyline/runline) | 'over'|'under' (total)
        american_odds INTEGER,
        line REAL,                           -- run line / spread / total line; NULL for ML
        sportsbook TEXT,                     -- 'pinnacle', 'draftkings', etc. NULL = aggregate/no-vig
        captured_at TEXT NOT NULL            -- ISO timestamp the snapshot was taken
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_log (
        sport TEXT NOT NULL,
        game_id TEXT NOT NULL,
        game_date TEXT NOT NULL,
        model_version TEXT NOT NULL,
        snapshot_at TEXT NOT NULL,           -- when this prediction was last (re)made

        away_team TEXT,
        home_team TEXT,

        -- full-game predictions
        away_win_prob REAL,                  -- 0..1
        home_win_prob REAL,
        away_fair_line INTEGER,
        home_fair_line INTEGER,
        away_lambda REAL,
        home_lambda REAL,
        projected_total REAL,
        ou_line REAL,

        -- F5 predictions
        f5_away_win_prob REAL,
        f5_home_win_prob REAL,
        f5_total REAL,

        -- prediction-time context
        lineup_state TEXT,                   -- 'official_both'|'official_partial'|'none'
        weather_factor REAL,
        park_runs_factor REAL,

        -- closing line slot (nullable; filled when odds are wired in -> enables CLV)
        close_away_ml INTEGER,
        close_home_ml INTEGER,
        close_total REAL,

        -- grading (filled by grade_predictions once the game is Final)
        graded_at TEXT,
        actual_away_score INTEGER,
        actual_home_score INTEGER,
        actual_winner TEXT,                  -- 'home'|'away'
        side_correct INTEGER,                -- 1 if model's favorite (>=50%) won
        brier REAL,                          -- (home_win_prob - home_won)^2
        total_error REAL,                    -- projected_total - (away+home actual)
        total_abs_error REAL,

        PRIMARY KEY (sport, game_id, model_version)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_games_date          ON games (sport, game_date)",
    "CREATE INDEX IF NOT EXISTS idx_lineups_game        ON lineups (sport, game_id)",
    "CREATE INDEX IF NOT EXISTS idx_box_game            ON box_scores (sport, game_id)",
    "CREATE INDEX IF NOT EXISTS idx_team_features_date  ON team_features (sport, game_date)",
    "CREATE INDEX IF NOT EXISTS idx_odds_snapshots_game ON odds_snapshots (sport, game_id, captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_prediction_log_date ON prediction_log (sport, game_date)",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _s(value: Any) -> str | None:
    return None if value is None else str(value)


class SQLiteStore:
    """Connection + schema + idempotent upserts for one SQLite database."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    # ---- lifecycle ----------------------------------------------------

    def init_schema(self) -> None:
        for stmt in SCHEMA:
            self.conn.execute(stmt)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---- reference table upserts -------------------------------------

    def upsert_team(self, team: Team) -> None:
        if team.team_id is None:
            return
        self.conn.execute(
            """
            INSERT INTO teams (sport, team_id, team_name, team_abbr, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(sport, team_id) DO UPDATE SET
                team_name = COALESCE(excluded.team_name, teams.team_name),
                team_abbr = COALESCE(excluded.team_abbr, teams.team_abbr),
                fetched_at = excluded.fetched_at
            """,
            (team.sport, _s(team.team_id), team.team_name, team.team_abbr, _now_iso()),
        )

    def upsert_player(self, player: Player) -> None:
        if player.player_id is None:
            return
        self.conn.execute(
            """
            INSERT INTO players (sport, player_id, player_name, primary_position, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(sport, player_id) DO UPDATE SET
                player_name = COALESCE(excluded.player_name, players.player_name),
                primary_position = COALESCE(excluded.primary_position, players.primary_position),
                fetched_at = excluded.fetched_at
            """,
            (player.sport, _s(player.player_id), player.player_name,
             player.primary_position, _now_iso()),
        )

    def upsert_venue(self, venue: Venue) -> None:
        if venue.venue_id is None:
            return
        self.conn.execute(
            """
            INSERT INTO venues (sport, venue_id, venue_name, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sport, venue_id) DO UPDATE SET
                venue_name = COALESCE(excluded.venue_name, venues.venue_name),
                fetched_at = excluded.fetched_at
            """,
            (venue.sport, _s(venue.venue_id), venue.venue_name, _now_iso()),
        )

    # ---- main entity upserts -----------------------------------------

    def upsert_game(self, game: Game) -> bool:
        """Upsert a game and its referenced teams/venue. Returns True if newly inserted."""
        self.upsert_team(game.home_team)
        self.upsert_team(game.away_team)
        if game.venue is not None:
            self.upsert_venue(game.venue)

        cur = self.conn.execute(
            "SELECT 1 FROM games WHERE sport=? AND game_id=?",
            (game.sport, _s(game.game_id)),
        )
        existed = cur.fetchone() is not None
        self.conn.execute(
            """
            INSERT INTO games (
                sport, game_id, game_date, status,
                home_team_id, away_team_id, venue_id,
                home_score, away_score, raw_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sport, game_id) DO UPDATE SET
                game_date = excluded.game_date,
                status = excluded.status,
                home_team_id = excluded.home_team_id,
                away_team_id = excluded.away_team_id,
                venue_id = excluded.venue_id,
                home_score = COALESCE(excluded.home_score, games.home_score),
                away_score = COALESCE(excluded.away_score, games.away_score),
                raw_json = excluded.raw_json,
                fetched_at = excluded.fetched_at
            """,
            (
                game.sport, _s(game.game_id), game.game_date, game.status,
                _s(game.home_team.team_id), _s(game.away_team.team_id),
                _s(game.venue.venue_id) if game.venue else None,
                game.home_score, game.away_score,
                json.dumps(game.raw) if game.raw is not None else None,
                _now_iso(),
            ),
        )
        return not existed

    def upsert_lineups(self, entries: Iterable[LineupEntry]) -> tuple[int, int]:
        inserted = updated = 0
        fetched = _now_iso()
        for entry in entries:
            self.upsert_player(entry.player)
            key = (entry.sport, _s(entry.game_id), _s(entry.team_id), _s(entry.player.player_id))
            existed = self.conn.execute(
                "SELECT 1 FROM lineups WHERE sport=? AND game_id=? AND team_id=? AND player_id=?",
                key,
            ).fetchone() is not None
            self.conn.execute(
                """
                INSERT INTO lineups (
                    sport, game_id, team_id, player_id,
                    batting_order, starter_slot, position, source, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sport, game_id, team_id, player_id) DO UPDATE SET
                    batting_order = excluded.batting_order,
                    starter_slot = excluded.starter_slot,
                    position = excluded.position,
                    source = excluded.source,
                    fetched_at = excluded.fetched_at
                """,
                (
                    entry.sport, _s(entry.game_id), _s(entry.team_id),
                    _s(entry.player.player_id),
                    entry.batting_order, entry.starter_slot,
                    entry.position, entry.source, fetched,
                ),
            )
            if existed:
                updated += 1
            else:
                inserted += 1
        return inserted, updated

    def upsert_boxscores(self, entries: Iterable[BoxscoreEntry]) -> tuple[int, int]:
        inserted = updated = 0
        fetched = _now_iso()
        for entry in entries:
            self.upsert_player(entry.player)
            key = (entry.sport, _s(entry.game_id), _s(entry.team_id), _s(entry.player.player_id))
            existed = self.conn.execute(
                "SELECT 1 FROM box_scores WHERE sport=? AND game_id=? AND team_id=? AND player_id=?",
                key,
            ).fetchone() is not None
            self.conn.execute(
                """
                INSERT INTO box_scores (
                    sport, game_id, team_id, player_id, position, stats_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sport, game_id, team_id, player_id) DO UPDATE SET
                    position = excluded.position,
                    stats_json = excluded.stats_json,
                    fetched_at = excluded.fetched_at
                """,
                (
                    entry.sport, _s(entry.game_id), _s(entry.team_id),
                    _s(entry.player.player_id), entry.position,
                    json.dumps(entry.stats), fetched,
                ),
            )
            if existed:
                updated += 1
            else:
                inserted += 1
        return inserted, updated

    # ---- run bookkeeping ---------------------------------------------

    def record_run_start(self, sport: str, table_name: str, target_date: str | None) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO ingest_runs (sport, table_name, target_date, rows_inserted,
                                     rows_updated, started_at)
            VALUES (?, ?, ?, 0, 0, ?)
            """,
            (sport, table_name, target_date, _now_iso()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_run_finish(
        self, run_id: int, rows_inserted: int, rows_updated: int, error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE ingest_runs
            SET rows_inserted=?, rows_updated=?, finished_at=?, error=?
            WHERE id=?
            """,
            (rows_inserted, rows_updated, _now_iso(), error, run_id),
        )
        self.conn.commit()

    def date_already_ingested(self, sport: str, table_name: str, target_date: str) -> bool:
        cur = self.conn.execute(
            """
            SELECT 1 FROM ingest_runs
            WHERE sport=? AND table_name=? AND target_date=?
              AND finished_at IS NOT NULL AND error IS NULL
            LIMIT 1
            """,
            (sport, table_name, target_date),
        )
        return cur.fetchone() is not None

    # ---- queries used by feature pipeline / projection engine --------

    def query_games_for_date(self, sport: str, game_date: str) -> list[dict[str, Any]]:
        """Return enriched game rows joined to teams + venues."""
        rows = self.conn.execute(
            """
            SELECT
                g.sport, g.game_id, g.game_date, g.status,
                g.home_team_id, ht.team_name AS home_team_name,
                g.away_team_id, at.team_name AS away_team_name,
                g.venue_id, v.venue_name,
                g.home_score, g.away_score, g.raw_json
            FROM games g
            LEFT JOIN teams  ht ON ht.sport = g.sport AND ht.team_id  = g.home_team_id
            LEFT JOIN teams  at ON at.sport = g.sport AND at.team_id  = g.away_team_id
            LEFT JOIN venues v  ON v.sport  = g.sport AND v.venue_id  = g.venue_id
            WHERE g.sport=? AND g.game_date=?
            ORDER BY g.game_id
            """,
            (sport, game_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_lineups_for_date(
        self, sport: str, game_date: str
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Return {game_id: {team_id: [lineup row dicts...]}}."""
        rows = self.conn.execute(
            """
            SELECT l.game_id, l.team_id, l.batting_order, l.starter_slot,
                   l.player_id, p.player_name, l.position, l.source
            FROM lineups l
            JOIN games g    ON g.sport=l.sport AND g.game_id=l.game_id
            LEFT JOIN players p ON p.sport=l.sport AND p.player_id=l.player_id
            WHERE l.sport=? AND g.game_date=?
            ORDER BY l.game_id, l.team_id, COALESCE(l.batting_order, l.starter_slot, 999)
            """,
            (sport, game_date),
        ).fetchall()
        out: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for r in rows:
            out.setdefault(r["game_id"], {}).setdefault(r["team_id"], []).append({
                "batting_order": r["batting_order"],
                "starter_slot": r["starter_slot"],
                "id": int(r["player_id"]) if r["player_id"] is not None else None,
                "name": r["player_name"],
                "position": r["position"],
                "source": r["source"],
            })
        return out

    # ---- team_features (derived; written by feature pipeline) --------

    def upsert_team_features(self, row: dict[str, Any]) -> bool:
        sport = row["sport"]
        game_id = _s(row["game_id"])
        team_id = _s(row["team_id"])
        existed = self.conn.execute(
            "SELECT 1 FROM team_features WHERE sport=? AND game_id=? AND team_id=?",
            (sport, game_id, team_id),
        ).fetchone() is not None
        self.conn.execute(
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
                is_opener, is_bullpen_heavy, park_runs_factor,
                raw_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?,
                      ?, ?,
                      ?, ?, ?,
                      ?, ?, ?, ?,
                      ?, ?, ?,
                      ?, ?,
                      ?, ?, ?,
                      ?, ?)
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
                _s(row.get("venue_id")),
                row.get("team_name"),
                _s(row.get("opponent_team_id")), row.get("opponent_team_name"),
                row.get("lineup_wrc_plus"), row.get("offensive_factor"),
                row.get("platoon_factor"),
                row.get("lineup_size"), row.get("lineup_source"),
                _s(row.get("starter_player_id")), row.get("starter_name"),
                row.get("starter_hand"),
                row.get("starter_fip"), row.get("starter_era"),
                row.get("starter_projected_ip"), row.get("starter_systems_count"),
                row.get("bullpen_fip"), row.get("bullpen_era"),
                row.get("bullpen_total_ip"),
                row.get("starter_share"), row.get("pitching_blended_fip"),
                1 if row.get("is_opener") else 0,
                1 if row.get("is_bullpen_heavy") else 0,
                row.get("park_runs_factor"),
                json.dumps(row.get("raw")) if row.get("raw") is not None else None,
                _now_iso(),
            ),
        )
        return not existed


    # ---- odds snapshots (line-movement history) ---------------------

    def add_odds_snapshot(
        self,
        *,
        sport: str,
        game_id: str,
        market: str,
        side: str,
        american_odds: int | None,
        line: float | None = None,
        sportsbook: str | None = None,
        captured_at: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO odds_snapshots
                (sport, game_id, market, side, american_odds, line, sportsbook, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sport, _s(game_id), market, side, american_odds, line, sportsbook,
                captured_at or _now_iso(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def query_odds_snapshots(
        self, sport: str, game_id: str, market: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return snapshots for a game ordered by capture time."""
        if market:
            rows = self.conn.execute(
                """
                SELECT * FROM odds_snapshots
                WHERE sport=? AND game_id=? AND market=?
                ORDER BY captured_at
                """,
                (sport, _s(game_id), market),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM odds_snapshots
                WHERE sport=? AND game_id=?
                ORDER BY captured_at, market, side
                """,
                (sport, _s(game_id)),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- prediction log (forward testing / track record) -------------

    def log_prediction(self, row: dict[str, Any]) -> bool:
        """Upsert one game's prediction of record (latest run wins).

        Keyed by (sport, game_id, model_version) so re-running the same day
        refreshes the prediction without spawning duplicates. Grading columns
        are preserved on conflict (re-logging a prediction doesn't wipe a
        grade, but the grader recomputes from current values anyway).
        Returns True if newly inserted.
        """
        sport = row["sport"]
        game_id = _s(row["game_id"])
        mv = row.get("model_version", "unknown")
        existed = self.conn.execute(
            "SELECT 1 FROM prediction_log WHERE sport=? AND game_id=? AND model_version=?",
            (sport, game_id, mv),
        ).fetchone() is not None
        self.conn.execute(
            """
            INSERT INTO prediction_log (
                sport, game_id, game_date, model_version, snapshot_at,
                away_team, home_team,
                away_win_prob, home_win_prob, away_fair_line, home_fair_line,
                away_lambda, home_lambda, projected_total, ou_line,
                f5_away_win_prob, f5_home_win_prob, f5_total,
                lineup_state, weather_factor, park_runs_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sport, game_id, model_version) DO UPDATE SET
                game_date = excluded.game_date,
                snapshot_at = excluded.snapshot_at,
                away_team = excluded.away_team,
                home_team = excluded.home_team,
                away_win_prob = excluded.away_win_prob,
                home_win_prob = excluded.home_win_prob,
                away_fair_line = excluded.away_fair_line,
                home_fair_line = excluded.home_fair_line,
                away_lambda = excluded.away_lambda,
                home_lambda = excluded.home_lambda,
                projected_total = excluded.projected_total,
                ou_line = excluded.ou_line,
                f5_away_win_prob = excluded.f5_away_win_prob,
                f5_home_win_prob = excluded.f5_home_win_prob,
                f5_total = excluded.f5_total,
                lineup_state = excluded.lineup_state,
                weather_factor = excluded.weather_factor,
                park_runs_factor = excluded.park_runs_factor
            """,
            (
                sport, game_id, row["game_date"], mv, _now_iso(),
                row.get("away_team"), row.get("home_team"),
                row.get("away_win_prob"), row.get("home_win_prob"),
                row.get("away_fair_line"), row.get("home_fair_line"),
                row.get("away_lambda"), row.get("home_lambda"),
                row.get("projected_total"), row.get("ou_line"),
                row.get("f5_away_win_prob"), row.get("f5_home_win_prob"), row.get("f5_total"),
                row.get("lineup_state"), row.get("weather_factor"), row.get("park_runs_factor"),
            ),
        )
        return not existed

    def grade_prediction(
        self, sport: str, game_id: str, away_score: int, home_score: int,
    ) -> bool:
        """Fill grading columns for one logged prediction from a final score.
        Returns True if a row was graded."""
        rows = self.conn.execute(
            "SELECT * FROM prediction_log WHERE sport=? AND game_id=?",
            (sport, _s(game_id)),
        ).fetchall()
        if not rows:
            return False
        graded_any = False
        for r in rows:
            if away_score == home_score:
                continue  # ties not graded
            home_won = 1 if home_score > away_score else 0
            winner = "home" if home_won else "away"
            hp = r["home_win_prob"]
            side_correct = None
            brier = None
            if hp is not None:
                side_correct = 1 if ((hp >= 0.5) == (home_won == 1)) else 0
                brier = (hp - home_won) ** 2
            total_err = total_abs = None
            if r["projected_total"] is not None:
                total_err = r["projected_total"] - (away_score + home_score)
                total_abs = abs(total_err)
            self.conn.execute(
                """
                UPDATE prediction_log SET
                    graded_at=?, actual_away_score=?, actual_home_score=?,
                    actual_winner=?, side_correct=?, brier=?,
                    total_error=?, total_abs_error=?
                WHERE sport=? AND game_id=? AND model_version=?
                """,
                (_now_iso(), away_score, home_score, winner, side_correct, brier,
                 total_err, total_abs, sport, _s(game_id), r["model_version"]),
            )
            graded_any = True
        return graded_any

    def query_prediction_log(
        self, sport: str = "mlb", graded_only: bool = False,
        start_date: str | None = None, end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM prediction_log WHERE sport=?"
        params: list[Any] = [sport]
        if graded_only:
            q += " AND graded_at IS NOT NULL"
        if start_date:
            q += " AND game_date >= ?"
            params.append(start_date)
        if end_date:
            q += " AND game_date <= ?"
            params.append(end_date)
        q += " ORDER BY game_date, game_id"
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]


__all__ = ["SQLiteStore", "DEFAULT_DB_PATH", "SCHEMA"]
