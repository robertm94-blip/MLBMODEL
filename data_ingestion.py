#!/usr/bin/env python3
"""Top-level entry script for the data_ingestion package.

Examples:
    python data_ingestion.py mlb --date 2026-04-26
    python data_ingestion.py mlb --backfill 2025-04-01:2025-10-01
    python data_ingestion.py all --date 2026-04-26
    python data_ingestion.py migrate
"""

from src.data_ingestion.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
