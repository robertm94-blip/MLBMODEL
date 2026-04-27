"""Bookmaker.eu MLB moneyline scraper.

Fetches the public MLB sportsbook page, parses each game's away/home
American moneyline, and computes no-vig "fair" probabilities using
src.edge.remove_vig (the same utility used by predict_full.py).

Note: bookmaker.eu's HTML markup changes from time to time. Two parser
strategies are tried (a structured selector and a fallback regex sweep)
so a single layout tweak doesn't take the bot down silently.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from src.edge import remove_vig
from src.odds_snapshot import GameKey, MLBOdds, OddsSnapshot

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://www.bookmaker.eu/sportsbook/baseball/mlb"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Map bookmaker display names → MLB standard 2-3 letter abbreviation.
TEAM_CODE_ALIASES = {
    "arizona diamondbacks": "ARI", "diamondbacks": "ARI", "arizona": "ARI",
    "atlanta braves": "ATL", "braves": "ATL", "atlanta": "ATL",
    "baltimore orioles": "BAL", "orioles": "BAL", "baltimore": "BAL",
    "boston red sox": "BOS", "red sox": "BOS", "boston": "BOS",
    "chicago cubs": "CHC", "cubs": "CHC",
    "chicago white sox": "CWS", "white sox": "CWS",
    "cincinnati reds": "CIN", "reds": "CIN", "cincinnati": "CIN",
    "cleveland guardians": "CLE", "guardians": "CLE", "cleveland": "CLE",
    "colorado rockies": "COL", "rockies": "COL", "colorado": "COL",
    "detroit tigers": "DET", "tigers": "DET", "detroit": "DET",
    "houston astros": "HOU", "astros": "HOU", "houston": "HOU",
    "kansas city royals": "KC", "royals": "KC", "kansas city": "KC",
    "los angeles angels": "LAA", "angels": "LAA",
    "los angeles dodgers": "LAD", "dodgers": "LAD",
    "miami marlins": "MIA", "marlins": "MIA", "miami": "MIA",
    "milwaukee brewers": "MIL", "brewers": "MIL", "milwaukee": "MIL",
    "minnesota twins": "MIN", "twins": "MIN", "minnesota": "MIN",
    "new york mets": "NYM", "mets": "NYM",
    "new york yankees": "NYY", "yankees": "NYY",
    "athletics": "OAK", "oakland athletics": "OAK", "oakland": "OAK", "as": "OAK",
    "philadelphia phillies": "PHI", "phillies": "PHI", "philadelphia": "PHI",
    "pittsburgh pirates": "PIT", "pirates": "PIT", "pittsburgh": "PIT",
    "san diego padres": "SD", "padres": "SD", "san diego": "SD",
    "san francisco giants": "SF", "giants": "SF", "san francisco": "SF",
    "seattle mariners": "SEA", "mariners": "SEA", "seattle": "SEA",
    "st. louis cardinals": "STL", "st louis cardinals": "STL", "cardinals": "STL",
    "tampa bay rays": "TB", "rays": "TB", "tampa bay": "TB",
    "texas rangers": "TEX", "rangers": "TEX", "texas": "TEX",
    "toronto blue jays": "TOR", "blue jays": "TOR", "toronto": "TOR",
    "washington nationals": "WSH", "nationals": "WSH", "washington": "WSH",
}

_AMERICAN_ODDS_RE = re.compile(r"^[+-]?\d{2,4}$")


def normalize_team(name: str) -> Optional[str]:
    """Map a bookmaker display name to a standard MLB code."""
    if not name:
        return None
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = cleaned.replace(".", "").strip()
    if cleaned in TEAM_CODE_ALIASES:
        return TEAM_CODE_ALIASES[cleaned]
    # Match by uppercase abbrev directly (already correct)
    upper = name.strip().upper()
    if upper in TEAM_CODE_ALIASES.values():
        return upper
    return None


def parse_american(text: str) -> Optional[int]:
    """Parse an American moneyline string (e.g. '-115', '+105', 'EV' → 100)."""
    if text is None:
        return None
    s = text.strip().replace("−", "-")  # unicode minus
    if s.lower() in ("ev", "even", "pk"):
        return 100
    if not _AMERICAN_ODDS_RE.match(s.replace("+", "")):
        # allow leading + sign
        if s.startswith("+") and _AMERICAN_ODDS_RE.match(s[1:]):
            try:
                return int(s)
            except ValueError:
                return None
        return None
    try:
        return int(s)
    except ValueError:
        return None


def compute_fair_probs(odds: MLBOdds) -> tuple[float, float]:
    """No-vig (away_prob, home_prob) using src.edge.remove_vig."""
    return remove_vig(odds.away_ml, odds.home_ml)


@dataclass
class _ParsedGame:
    away_team: str
    home_team: str
    away_ml: int
    home_ml: int
    start_time: str = ""


class BookmakerClient:
    """Async HTML scraper for Bookmaker.eu MLB moneylines."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 10.0):
        self.url = url
        self._client = httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        self._last_good: dict[GameKey, MLBOdds] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BookmakerClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def fetch_html(self) -> Optional[str]:
        try:
            resp = await self._client.get(self.url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as e:
            logger.warning("Bookmaker fetch failed: %s", e)
            return None

    async def fetch_mlb_odds(self) -> dict[GameKey, MLBOdds]:
        """One-shot scrape. Falls back to last good snapshot on failure."""
        html = await self.fetch_html()
        if html is None:
            return dict(self._last_good)
        try:
            parsed = self._parse_html(html)
        except Exception:
            logger.exception("Bookmaker parse failed; returning last-good snapshot")
            return dict(self._last_good)

        now = time.time()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out: dict[GameKey, MLBOdds] = {}
        for g in parsed:
            away_code = normalize_team(g.away_team)
            home_code = normalize_team(g.home_team)
            if not (away_code and home_code):
                logger.debug("Skipping unmappable teams: %s vs %s", g.away_team, g.home_team)
                continue
            key: GameKey = (today, away_code, home_code)
            out[key] = MLBOdds(
                away_team=away_code,
                home_team=home_code,
                away_ml=g.away_ml,
                home_ml=g.home_ml,
                start_time=g.start_time,
                fetched_at=now,
            )
        if out:
            self._last_good = out
        elif self._last_good:
            logger.warning("Empty parse result; serving last-good snapshot (%d games)", len(self._last_good))
            return dict(self._last_good)
        return out

    # ── Parsing ──────────────────────────────────────────────────────

    def _parse_html(self, html: str) -> list[_ParsedGame]:
        """Try the structured parser first; fall back to a regex sweep."""
        soup = BeautifulSoup(html, "lxml")
        games = self._parse_structured(soup)
        if games:
            return games
        logger.debug("Structured parse found no games; trying fallback regex parser")
        return self._parse_fallback(soup, html)

    def _parse_structured(self, soup: BeautifulSoup) -> list[_ParsedGame]:
        """Parse using common bookmaker.eu selectors. Returns [] if layout doesn't match."""
        games: list[_ParsedGame] = []

        # Bookmaker typically wraps each matchup in a container. We look for
        # any element that contains exactly two team-name children plus two
        # odds children. The exact class names change, so we iterate broadly
        # and rely on the shape of the data, not specific class names.
        candidates = soup.select(
            "[class*='event'], [class*='matchup'], [class*='game'], "
            "[class*='line'], div[data-event], tr"
        )
        seen_pairs: set[tuple[str, str]] = set()
        for el in candidates:
            text_nodes = [t.strip() for t in el.stripped_strings]
            if len(text_nodes) < 4:
                continue
            # Look for two team names and two odds in close sequence
            team_idxs = [i for i, t in enumerate(text_nodes) if normalize_team(t)]
            odds_idxs = [i for i, t in enumerate(text_nodes) if parse_american(t) is not None]
            if len(team_idxs) < 2 or len(odds_idxs) < 2:
                continue
            t1, t2 = team_idxs[0], team_idxs[1]
            # Find the two odds that come AFTER each team token (ML column)
            o1 = next((i for i in odds_idxs if i > t1 and i < t2), None)
            o2 = next((i for i in odds_idxs if i > t2), None)
            if o1 is None or o2 is None:
                continue
            away = text_nodes[t1]
            home = text_nodes[t2]
            pair = (normalize_team(away), normalize_team(home))
            if pair[0] is None or pair[1] is None or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            away_ml = parse_american(text_nodes[o1])
            home_ml = parse_american(text_nodes[o2])
            if away_ml is None or home_ml is None:
                continue
            # Sanity: a moneyline of 0 or ±50 is almost certainly a parse error
            if abs(away_ml) < 100 and away_ml != 100:
                continue
            if abs(home_ml) < 100 and home_ml != 100:
                continue
            games.append(_ParsedGame(
                away_team=away, home_team=home,
                away_ml=away_ml, home_ml=home_ml,
            ))
        return games

    def _parse_fallback(self, soup: BeautifulSoup, html: str) -> list[_ParsedGame]:
        """Last-resort regex parser over visible text.

        Looks for the pattern:  <TeamA> ... <oddsA> ... <TeamB> ... <oddsB>
        within sliding windows of stripped text tokens.
        """
        text = " ".join(soup.stripped_strings)
        tokens = text.split()
        games: list[_ParsedGame] = []
        seen: set[tuple[str, str]] = set()

        # Walk the token stream looking for two-team sequences. Try 1-3 word names.
        i = 0
        n = len(tokens)
        while i < n - 6:
            for span_a in (3, 2, 1):
                if i + span_a >= n:
                    continue
                team_a_str = " ".join(tokens[i : i + span_a])
                code_a = normalize_team(team_a_str)
                if not code_a:
                    continue
                # Next 1-3 tokens may be odds
                j = i + span_a
                ml_a = None
                while j < min(i + span_a + 4, n):
                    ml_a = parse_american(tokens[j])
                    if ml_a is not None and abs(ml_a) >= 100:
                        break
                    ml_a = None
                    j += 1
                if ml_a is None:
                    continue
                k = j + 1
                # Look ahead for second team within next 8 tokens
                team_b_found = False
                while k < min(j + 9, n):
                    for span_b in (3, 2, 1):
                        if k + span_b > n:
                            continue
                        team_b_str = " ".join(tokens[k : k + span_b])
                        code_b = normalize_team(team_b_str)
                        if not code_b or code_b == code_a:
                            continue
                        # Second odds within next 4 tokens
                        m = k + span_b
                        ml_b = None
                        while m < min(k + span_b + 4, n):
                            ml_b = parse_american(tokens[m])
                            if ml_b is not None and abs(ml_b) >= 100:
                                break
                            ml_b = None
                            m += 1
                        if ml_b is None:
                            continue
                        pair = (code_a, code_b)
                        if pair in seen:
                            i = m
                            team_b_found = True
                            break
                        seen.add(pair)
                        games.append(_ParsedGame(
                            away_team=team_a_str, home_team=team_b_str,
                            away_ml=ml_a, home_ml=ml_b,
                        ))
                        i = m
                        team_b_found = True
                        break
                    if team_b_found:
                        break
                    k += 1
                if team_b_found:
                    break
            i += 1
        return games

    # ── Polling loop ─────────────────────────────────────────────────

    async def run(
        self,
        snapshot: OddsSnapshot,
        interval: float = 2.0,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Poll loop: fetch every `interval` seconds, write to snapshot."""
        logger.info("BookmakerClient started — polling %s every %.1fs", self.url, interval)
        while not (stop_event and stop_event.is_set()):
            t0 = time.time()
            odds_map = await self.fetch_mlb_odds()
            if odds_map:
                async with snapshot.lock:
                    for key, odds in odds_map.items():
                        away_fp, home_fp = compute_fair_probs(odds)
                        snapshot.ensure(key, odds, away_fp, home_fp)
                logger.debug("Bookmaker tick: %d games", len(odds_map))
            else:
                logger.warning("Bookmaker tick: no games parsed")
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.1, interval - elapsed))
        logger.info("BookmakerClient stopped")
