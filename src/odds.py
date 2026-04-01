"""Historical odds data management for backtesting.

Fetches closing moneyline odds from The Odds API, caches them locally,
and matches them to historical game results for backtest evaluation.

Supports:
- The Odds API (https://the-odds-api.com) — requires API key, historical endpoint
- Manual CSV import (any source with date, teams, closing odds)
- Fallback: derive market-implied odds from consensus lines

Data is cached per-season in data/odds_{year}.json.
"""

import csv
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any

import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# The Odds API configuration
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"

# Team name normalization — maps various formats to MLB Stats API full names
TEAM_ALIASES: dict[str, str] = {
    # Common abbreviations
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres", "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
    # The Odds API uses these names
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "Atlanta Braves": "Atlanta Braves",
    "Baltimore Orioles": "Baltimore Orioles",
    "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs",
    "Chicago White Sox": "Chicago White Sox",
    "Cincinnati Reds": "Cincinnati Reds",
    "Cleveland Guardians": "Cleveland Guardians",
    "Cleveland Indians": "Cleveland Guardians",
    "Colorado Rockies": "Colorado Rockies",
    "Detroit Tigers": "Detroit Tigers",
    "Houston Astros": "Houston Astros",
    "Kansas City Royals": "Kansas City Royals",
    "Los Angeles Angels": "Los Angeles Angels",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "Miami Marlins": "Miami Marlins",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "Minnesota Twins": "Minnesota Twins",
    "New York Mets": "New York Mets",
    "New York Yankees": "New York Yankees",
    "Oakland Athletics": "Athletics",
    "Athletics": "Athletics",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "San Diego Padres": "San Diego Padres",
    "San Francisco Giants": "San Francisco Giants",
    "Seattle Mariners": "Seattle Mariners",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "St Louis Cardinals": "St. Louis Cardinals",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Texas Rangers": "Texas Rangers",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "Washington Nationals": "Washington Nationals",
}


def normalize_team(name: str) -> str:
    """Normalize a team name to MLB Stats API full name."""
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    # Try case-insensitive match
    for alias, full in TEAM_ALIASES.items():
        if alias.lower() == name.lower():
            return full
    return name


# ─── THE ODDS API ──────────────────────────────────────────────────────────


def fetch_odds_api_historical(
    date: str,
    api_key: str,
    bookmakers: str = "fanduel,draftkings,betmgm,pointsbet",
) -> list[dict]:
    """Fetch historical closing odds from The Odds API.

    Args:
        date: Date string YYYY-MM-DD
        api_key: The Odds API key
        bookmakers: Comma-separated bookmaker keys

    Returns list of game odds records.
    Requires a paid plan with historical endpoint access.
    """
    # The historical endpoint uses ISO timestamps
    dt = datetime.strptime(date, "%Y-%m-%d")
    # Use end-of-day to get closing lines
    iso_date = dt.strftime("%Y-%m-%dT23:59:59Z")

    url = f"{ODDS_API_BASE}/historical/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": bookmakers,
        "date": iso_date,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for event in data.get("data", []):
        game_date = event.get("commence_time", "")[:10]
        home = normalize_team(event.get("home_team", ""))
        away = normalize_team(event.get("away_team", ""))

        # Average closing odds across available bookmakers
        home_prices = []
        away_prices = []
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team = normalize_team(outcome.get("name", ""))
                    price = outcome.get("price", 0)
                    if team == home:
                        home_prices.append(price)
                    elif team == away:
                        away_prices.append(price)

        if home_prices and away_prices:
            games.append({
                "date": game_date,
                "home_team": home,
                "away_team": away,
                "home_odds": _consensus_odds(home_prices),
                "away_odds": _consensus_odds(away_prices),
                "n_books": len(home_prices),
                "source": "the-odds-api",
            })

    return games


def _consensus_odds(prices: list[int]) -> int:
    """Compute consensus (median) American odds from multiple bookmakers."""
    if not prices:
        return -110
    sorted_prices = sorted(prices)
    mid = len(sorted_prices) // 2
    if len(sorted_prices) % 2 == 0:
        return int(round((sorted_prices[mid - 1] + sorted_prices[mid]) / 2))
    return sorted_prices[mid]


# ─── CSV IMPORT ────────────────────────────────────────────────────────────


def import_odds_csv(filepath: str) -> list[dict]:
    """Import odds from a CSV file.

    Expected columns (flexible naming):
        date, home_team, away_team, home_odds (or home_ml), away_odds (or away_ml)

    Accepts American odds format (e.g., -150, +130).
    """
    records = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        # Normalize column names
        for row in reader:
            # Find the right column names
            date = row.get("date") or row.get("Date") or row.get("game_date") or ""
            home = row.get("home_team") or row.get("Home") or row.get("home") or ""
            away = row.get("away_team") or row.get("Away") or row.get("away") or ""
            home_ml = row.get("home_odds") or row.get("home_ml") or row.get("Home ML") or row.get("close_home") or ""
            away_ml = row.get("away_odds") or row.get("away_ml") or row.get("Away ML") or row.get("close_away") or ""

            if not all([date, home, away, home_ml, away_ml]):
                continue

            try:
                records.append({
                    "date": date.strip(),
                    "home_team": normalize_team(home.strip()),
                    "away_team": normalize_team(away.strip()),
                    "home_odds": int(float(home_ml)),
                    "away_odds": int(float(away_ml)),
                    "source": "csv",
                })
            except (ValueError, TypeError):
                continue

    return records


# ─── CACHE MANAGEMENT ──────────────────────────────────────────────────────


def get_odds_cache_path(season: int) -> str:
    """Get path to cached odds file for a season."""
    return os.path.join(DATA_DIR, f"odds_{season}.json")


def load_cached_odds(season: int) -> list[dict]:
    """Load cached odds for a season."""
    path = get_odds_cache_path(season)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_odds_cache(season: int, odds: list[dict]) -> None:
    """Save odds data to cache."""
    path = get_odds_cache_path(season)
    with open(path, "w") as f:
        json.dump(odds, f, indent=2)


def merge_odds(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge new odds into existing, avoiding duplicates.

    Uses (date, home_team, away_team) as the unique key.
    New records overwrite existing for the same game.
    """
    by_key = {}
    for rec in existing:
        key = (rec["date"], rec["home_team"], rec["away_team"])
        by_key[key] = rec
    for rec in new:
        key = (rec["date"], rec["home_team"], rec["away_team"])
        by_key[key] = rec
    return sorted(by_key.values(), key=lambda r: r["date"])


# ─── GAME MATCHING ─────────────────────────────────────────────────────────


def match_odds_to_games(
    games: list[dict], odds: list[dict]
) -> list[dict]:
    """Match odds records to historical game results.

    Enriches each game dict with 'home_odds' and 'away_odds' fields.
    Returns only games that have matching odds.

    Games without matching odds are returned with home_odds=None, away_odds=None.
    """
    # Build lookup: (date, home_team) -> odds record
    # Also try (date, away_team) for flexibility
    odds_lookup: dict[tuple[str, str], dict] = {}
    for rec in odds:
        odds_lookup[(rec["date"], rec["home_team"])] = rec
        # Also index by away team in case team names differ slightly
        odds_lookup[(rec["date"], rec["away_team"])] = rec

    matched = 0
    results = []
    for game in games:
        game_date = game.get("date", "")
        home = game.get("home_team_name", "") or game.get("home_team", "")
        away = game.get("away_team_name", "") or game.get("away_team", "")

        enriched = dict(game)

        # Try exact match on home team
        odds_rec = odds_lookup.get((game_date, home))

        # If not found, try normalizing
        if not odds_rec:
            odds_rec = odds_lookup.get((game_date, normalize_team(home)))

        if odds_rec:
            enriched["home_odds"] = odds_rec["home_odds"]
            enriched["away_odds"] = odds_rec["away_odds"]
            enriched["odds_source"] = odds_rec.get("source", "unknown")
            matched += 1
        else:
            enriched["home_odds"] = None
            enriched["away_odds"] = None
            enriched["odds_source"] = None

        results.append(enriched)

    return results


# ─── MARKET-IMPLIED ODDS GENERATION ───────────────────────────────────────


def generate_market_odds_from_probabilities(
    predictions: list[dict],
    vig: float = 0.045,
) -> list[dict]:
    """Generate synthetic market odds from model probabilities.

    Useful as a baseline when real closing odds aren't available.
    Applies standard vig (~4.5%) to model probabilities to simulate
    what a bookmaker would post.

    This is NOT a substitute for real odds — it tests whether the model
    can beat its own vig-adjusted lines (sanity check).
    """
    odds_records = []
    for pred in predictions:
        home_prob = pred.get("home_win_prob", 0.5)
        away_prob = 1 - home_prob

        # Apply vig (inflate both sides proportionally)
        total = 1.0 + vig
        home_implied = home_prob * total / (home_prob + away_prob)
        away_implied = away_prob * total / (home_prob + away_prob)

        home_odds = _prob_to_american(home_implied)
        away_odds = _prob_to_american(away_implied)

        odds_records.append({
            "date": pred.get("date", ""),
            "home_team": pred.get("home_team", ""),
            "away_team": pred.get("away_team", ""),
            "home_odds": home_odds,
            "away_odds": away_odds,
            "source": "model-derived",
        })

    return odds_records


def _prob_to_american(prob: float) -> int:
    """Convert implied probability to American odds."""
    if prob <= 0 or prob >= 1:
        return -110
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    else:
        return int(round(100 * (1 - prob) / prob))
