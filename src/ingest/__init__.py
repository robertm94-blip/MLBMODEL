"""Data ingestion package: pulls MLB and NBA lineups + box scores into SQLite."""

from src.ingest.db import connect, init_schema
from src.ingest.http import HttpClient

__all__ = ["connect", "init_schema", "HttpClient"]
