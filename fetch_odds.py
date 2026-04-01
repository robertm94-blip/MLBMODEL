#!/usr/bin/env python3
"""Fetch and cache historical closing odds for MLB backtesting.

Supports multiple data sources:
1. The Odds API (requires API key + paid historical plan)
2. CSV import (any source — see expected format below)
3. Bulk fetch for entire seasons

Usage:
    # Fetch from The Odds API for a date range
    python fetch_odds.py --source api --api-key YOUR_KEY --start 2025-03-27 --end 2025-09-28

    # Import from CSV file
    python fetch_odds.py --source csv --file odds_2025.csv --season 2025

    # Check coverage for cached odds
    python fetch_odds.py --check --season 2025

    # Match cached odds to results and show stats
    python fetch_odds.py --match --season 2025

CSV Format:
    date,home_team,away_team,home_odds,away_odds
    2025-03-27,San Diego Padres,Atlanta Braves,-145,+125
    ...

    Column names are flexible — see src/odds.py import_odds_csv() for details.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

from src.odds import (
    fetch_odds_api_historical,
    import_odds_csv,
    load_cached_odds,
    save_odds_cache,
    merge_odds,
    match_odds_to_games,
    get_odds_cache_path,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch_season_api(season: int, api_key: str, start: str | None = None, end: str | None = None) -> None:
    """Fetch odds for an entire season from The Odds API."""
    if start is None:
        start = f"{season}-03-20"
    if end is None:
        end = f"{season}-10-01"

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    existing = load_cached_odds(season)
    existing_dates = {r["date"] for r in existing}
    print(f"\n⚾ Fetching odds for {season} season")
    print(f"   Range: {start} → {end}")
    print(f"   Existing cached: {len(existing)} records ({len(existing_dates)} dates)")

    all_new = []
    current = start_dt
    api_calls = 0

    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")

        if date_str in existing_dates:
            current += timedelta(days=1)
            continue

        try:
            day_odds = fetch_odds_api_historical(date_str, api_key)
            all_new.extend(day_odds)
            api_calls += 1

            if day_odds:
                print(f"   {date_str}: {len(day_odds)} games")
            else:
                print(f"   {date_str}: no games")

            # Rate limit: The Odds API has per-second limits
            time.sleep(1.0)

        except Exception as e:
            print(f"   {date_str}: ERROR — {e}")
            time.sleep(2.0)

        current += timedelta(days=1)

    if all_new:
        merged = merge_odds(existing, all_new)
        save_odds_cache(season, merged)
        print(f"\n   ✓ Saved {len(merged)} total records ({len(all_new)} new)")
    else:
        print(f"\n   No new records to save")

    print(f"   API calls made: {api_calls}")


def import_csv(filepath: str, season: int) -> None:
    """Import odds from a CSV file."""
    if not os.path.exists(filepath):
        print(f"\n❌ File not found: {filepath}")
        sys.exit(1)

    print(f"\n⚾ Importing odds from {filepath}")
    records = import_odds_csv(filepath)
    print(f"   Parsed: {len(records)} records")

    if not records:
        print("   ⚠ No valid records found. Check CSV format.")
        return

    # Show sample
    print(f"\n   Sample records:")
    for r in records[:3]:
        print(f"   {r['date']}  {r['away_team']:25s} @ {r['home_team']:25s}  "
              f"{r['away_odds']:+d} / {r['home_odds']:+d}")

    existing = load_cached_odds(season)
    merged = merge_odds(existing, records)
    save_odds_cache(season, merged)
    print(f"\n   ✓ Saved {len(merged)} total records (was {len(existing)})")


def check_coverage(season: int) -> None:
    """Report on odds data coverage for a season."""
    odds = load_cached_odds(season)
    results_path = os.path.join(DATA_DIR, f"results_{season}.json")

    print(f"\n⚾ Odds Coverage Report — {season}")
    print(f"{'─' * 55}")

    if not odds:
        print(f"   No odds data cached for {season}")
        print(f"   Cache file: {get_odds_cache_path(season)}")
        print(f"\n   To add odds data:")
        print(f"   1. API:  python fetch_odds.py --source api --api-key YOUR_KEY --season {season}")
        print(f"   2. CSV:  python fetch_odds.py --source csv --file odds_{season}.csv --season {season}")
        return

    dates = sorted({r["date"] for r in odds})
    sources = {}
    for r in odds:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"   Total odds records: {len(odds)}")
    print(f"   Date range: {dates[0]} → {dates[-1]}")
    print(f"   Unique dates: {len(dates)}")
    print(f"   Sources: {', '.join(f'{s} ({n})' for s, n in sources.items())}")

    # Check against results
    if os.path.exists(results_path):
        with open(results_path) as f:
            games = json.load(f)

        matched = match_odds_to_games(games, odds)
        with_odds = sum(1 for g in matched if g.get("home_odds") is not None)
        pct = with_odds / len(games) * 100 if games else 0

        print(f"\n   Game results: {len(games)}")
        print(f"   Matched with odds: {with_odds} ({pct:.1f}%)")
        print(f"   Missing odds: {len(games) - with_odds}")

        # Show odds distribution
        odds_vals = [g["home_odds"] for g in matched if g.get("home_odds") is not None]
        if odds_vals:
            fav_count = sum(1 for o in odds_vals if o < -110)
            dog_count = sum(1 for o in odds_vals if o > 110)
            even_count = len(odds_vals) - fav_count - dog_count
            print(f"\n   Home favorites: {fav_count} ({fav_count/len(odds_vals)*100:.0f}%)")
            print(f"   Home dogs: {dog_count} ({dog_count/len(odds_vals)*100:.0f}%)")
            print(f"   Pick'em: {even_count} ({even_count/len(odds_vals)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Fetch/import historical MLB closing odds")

    parser.add_argument("--source", choices=["api", "csv"],
                        help="Data source: 'api' (The Odds API) or 'csv' (file import)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("ODDS_API_KEY"),
                        help="The Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--file", type=str, help="CSV file to import")
    parser.add_argument("--season", type=int, default=2025, help="Season year")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--check", action="store_true", help="Check odds coverage for a season")
    parser.add_argument("--match", action="store_true", help="Match odds to results and show stats")

    args = parser.parse_args()

    if args.check or args.match:
        check_coverage(args.season)
        return

    if args.source == "api":
        if not args.api_key:
            print("\n❌ API key required. Use --api-key or set ODDS_API_KEY env var.")
            print("   Get a key at https://the-odds-api.com")
            sys.exit(1)
        fetch_season_api(args.season, args.api_key, args.start, args.end)

    elif args.source == "csv":
        if not args.file:
            print("\n❌ CSV file required. Use --file path/to/odds.csv")
            sys.exit(1)
        import_csv(args.file, args.season)

    else:
        parser.print_help()
        print("\n\nExamples:")
        print(f"  python fetch_odds.py --source csv --file odds_2025.csv --season 2025")
        print(f"  python fetch_odds.py --source api --api-key KEY --season 2025")
        print(f"  python fetch_odds.py --check --season 2025")


if __name__ == "__main__":
    main()
