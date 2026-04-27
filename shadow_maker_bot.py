#!/usr/bin/env python3
"""Shadow Maker — Kalshi MLB moneyline bot anchored to Bookmaker.eu.

Two asyncio coroutines run concurrently:
  - BookmakerClient.run()  scrapes bookmaker.eu every 2s, writes fair
                           probabilities into the shared OddsSnapshot.
  - ShadowMaker.run()      reacts to per-game change events; for each
                           mapped Kalshi market, posts post-only quotes
                           at P_f ± 1¢. Cancels and re-quotes when the
                           anchor moves more than 2¢.

Usage:
    cp .env.example .env  # then fill in KALSHI_API_KEY + KALSHI_PRIVATE_KEY_PATH
    python shadow_maker_bot.py --dry-run --verbose       # smoke test
    python shadow_maker_bot.py                           # demo (sandbox)
    python shadow_maker_bot.py --prod --size 5           # live trading
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.bookmaker_client import BookmakerClient
from src.kalshi_v2_client import KalshiV2Client
from src.odds_snapshot import GameKey, OddsSnapshot
from src.shadow_maker import MarketMapping, ShadowMaker


logger = logging.getLogger("shadow_maker")

# ── Kalshi ticker parsing ───────────────────────────────────────────

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Best-effort match for tickers like  KXMLBGAME-26APR27NYYBOS-NYY
_TICKER_RE = re.compile(
    r"^KXMLB(?:GAME)?-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})"
    r"(?P<away>[A-Z]{2,4})(?P<home>[A-Z]{2,4})-(?P<side>[A-Z]{2,4})$"
)


def parse_kalshi_ticker(ticker: str) -> Optional[tuple[GameKey, str]]:
    """Parse a Kalshi MLB game ticker into (game_key, side).

    Returns None if the ticker doesn't match the expected format.
    Side is "yes_home" or "yes_away" depending on which team the
    YES contract resolves on.
    """
    m = _TICKER_RE.match(ticker)
    if not m:
        return None
    mon = _MONTHS.get(m.group("mon"))
    if not mon:
        return None
    try:
        year = 2000 + int(m.group("yy"))
        date_iso = f"{year:04d}-{mon:02d}-{int(m.group('dd')):02d}"
    except ValueError:
        return None
    away = m.group("away")
    home = m.group("home")
    side_team = m.group("side")
    if side_team == away:
        side = "yes_away"
    elif side_team == home:
        side = "yes_home"
    else:
        return None
    return (date_iso, away, home), side


# ── Mapping discovery ───────────────────────────────────────────────

async def auto_discover(client: KalshiV2Client) -> list[MarketMapping]:
    """List today's open Kalshi MLB markets and parse each ticker."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mappings: list[MarketMapping] = []
    cursor: Optional[str] = None
    while True:
        params = {"limit": 200, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        resp = await client.get_markets(**params)
        for m in resp.get("markets", []):
            ticker = m.get("ticker") or ""
            if not ticker.startswith("KXMLB"):
                continue
            parsed = parse_kalshi_ticker(ticker)
            if not parsed:
                logger.debug("Skipping unparseable ticker: %s", ticker)
                continue
            game_key, side = parsed
            if game_key[0] != today:
                continue
            mappings.append(MarketMapping(game_key=game_key, ticker=ticker, side=side))
        cursor = resp.get("cursor")
        if not cursor:
            break
    return mappings


def apply_config_overrides(
    mappings: list[MarketMapping],
    config_path: Path,
) -> list[MarketMapping]:
    """Apply overrides + exclusions from config/shadow_markets.json."""
    if not config_path.exists():
        return mappings
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        logger.exception("Failed to read config %s; ignoring", config_path)
        return mappings

    excluded = set(cfg.get("exclude") or [])
    by_ticker: dict[str, MarketMapping] = {m.ticker: m for m in mappings}

    for entry in cfg.get("overrides") or []:
        try:
            key: GameKey = (entry["date"], entry["away"], entry["home"])
            ticker = entry["ticker"]
            side = entry["side"]
        except KeyError:
            logger.warning("Skipping malformed override: %s", entry)
            continue
        if side not in ("yes_home", "yes_away"):
            logger.warning("Bad side %r in override %s", side, ticker)
            continue
        by_ticker[ticker] = MarketMapping(game_key=key, ticker=ticker, side=side)

    out = []
    for m in by_ticker.values():
        m.excluded = m.ticker in excluded
        out.append(m)
    return out


# ── Main ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shadow Maker — Kalshi MLB bot anchored on Bookmaker.eu."
    )
    p.add_argument("--config", type=Path, default=Path("config/shadow_markets.json"),
                   help="JSON file with ticker overrides/exclusions.")
    p.add_argument("--size", type=int, default=1,
                   help="Contracts per side per market (default: 1).")
    p.add_argument("--interval", type=float, default=2.0,
                   help="Seconds between bookmaker fetches (default: 2).")
    p.add_argument("--prod", action="store_true",
                   help="Hit Kalshi production. Default is demo/sandbox.")
    p.add_argument("--dry-run", action="store_true",
                   help="Scrape and decide quotes, but don't send orders.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug logging.")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    api_key = os.environ.get("KALSHI_API_KEY")
    pk_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if not api_key or not pk_path:
        logger.error("KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH must be set in .env")
        return 1
    if not Path(pk_path).exists():
        logger.error("Private key file not found: %s", pk_path)
        return 1

    demo = not args.prod
    env_label = "DEMO/SANDBOX" if demo else "PRODUCTION"
    logger.info("Starting Shadow Maker [%s]", env_label)
    if args.dry_run:
        logger.info("*** DRY-RUN — no orders will be placed ***")

    kalshi = KalshiV2Client(api_key, pk_path, demo=demo)
    try:
        bal = await kalshi.get_balance()
        logger.info("Kalshi connected. Balance: %s", bal.get("balance", bal))
    except Exception as e:
        logger.error("Kalshi connectivity check failed: %s", e)
        return 1

    logger.info("Discovering MLB markets for today...")
    auto_mappings = await auto_discover(kalshi)
    logger.info("Auto-discovered %d markets", len(auto_mappings))
    mappings = apply_config_overrides(auto_mappings, args.config)
    active = [m for m in mappings if not m.excluded]
    logger.info("Quoting %d markets (after config): %s", len(active),
                ", ".join(sorted(m.ticker for m in active)) or "<none>")

    if not active:
        logger.error("No markets to quote. Add overrides to %s or wait for markets to open.",
                     args.config)
        return 1

    snapshot = OddsSnapshot()
    bookmaker = BookmakerClient()
    shadow = ShadowMaker(
        client=kalshi,
        snapshot=snapshot,
        mappings=mappings,
        order_size=args.size,
        dry_run=args.dry_run,
    )

    stop_event = asyncio.Event()

    def _on_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()
        shadow.stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            # Windows fallback
            signal.signal(sig, lambda *_: _on_signal())

    bookmaker_task = asyncio.create_task(
        bookmaker.run(snapshot, interval=args.interval, stop_event=stop_event),
        name="bookmaker_loop",
    )
    shadow_task = asyncio.create_task(shadow.run(), name="shadow_loop")
    status_task = asyncio.create_task(_status_logger(shadow, stop_event), name="status_loop")

    await stop_event.wait()

    logger.info("Cancelling tasks and resting orders...")
    for t in (bookmaker_task, shadow_task, status_task):
        t.cancel()
    await asyncio.gather(bookmaker_task, shadow_task, status_task, return_exceptions=True)
    await bookmaker.aclose()
    await shadow.cancel_all()
    logger.info("Shadow Maker stopped.")
    return 0


async def _status_logger(shadow: ShadowMaker, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.info(shadow.status_summary())


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    load_dotenv()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
