#!/usr/bin/env python3
"""
Kalshi Market-Making Bot
========================
Maintains a fixed 2-cent spread on every configured market.
Continuously refreshes orders so you always have a bid and ask
exactly 2 cents apart, centered on the market midpoint.

Usage:
    # Using environment variables (recommended)
    export KALSHI_API_KEY="your-api-key-uuid"
    export KALSHI_PRIVATE_KEY_PATH="/path/to/your/private_key.pem"

    # Quote specific market tickers
    python kalshi_mm_bot.py --tickers TICKER1 TICKER2 TICKER3

    # Quote all open markets under an event
    python kalshi_mm_bot.py --event MLB-FAVTEAM-2026

    # Adjust refresh interval and order size
    python kalshi_mm_bot.py --tickers TICKER1 --interval 5 --size 10

    # Use demo/sandbox environment
    python kalshi_mm_bot.py --tickers TICKER1 --demo

    # Dry-run mode (no real orders)
    python kalshi_mm_bot.py --tickers TICKER1 --dry-run
"""

import argparse
import logging
import os
import signal
import sys
import time

from src.kalshi_client import KalshiClient
from src.market_maker import MarketMaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kalshi_mm")

# ── Globals for graceful shutdown ────────────────────────────────────
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received — cancelling all quotes...")
    _shutdown = True


def parse_args():
    p = argparse.ArgumentParser(
        description="Kalshi market-making bot — maintains a 2-cent spread."
    )
    p.add_argument(
        "--tickers", nargs="+", default=[],
        help="Market tickers to quote (e.g. MLB-NYY-YES-2026-04-02)",
    )
    p.add_argument(
        "--event", type=str, default=None,
        help="Event ticker — bot will quote all open markets under this event.",
    )
    p.add_argument(
        "--interval", type=float, default=3.0,
        help="Seconds between refresh cycles (default: 3).",
    )
    p.add_argument(
        "--size", type=int, default=1,
        help="Number of contracts per side per market (default: 1).",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Use Kalshi demo/sandbox environment.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without placing orders.",
    )
    p.add_argument(
        "--api-key", type=str, default=None,
        help="Kalshi API key (or set KALSHI_API_KEY env var).",
    )
    p.add_argument(
        "--private-key", type=str, default=None,
        help="Path to RSA private key PEM (or set KALSHI_PRIVATE_KEY_PATH env var).",
    )
    p.add_argument(
        "--max-exposure", type=int, default=None,
        help="Maximum total contracts across all markets. Bot pauses quoting when exceeded.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def resolve_tickers(client: KalshiClient, args) -> list[str]:
    """Build the ticker list from CLI args."""
    tickers = list(args.tickers)

    if args.event:
        logger.info("Fetching markets for event %s...", args.event)
        markets = client.get_event_markets(args.event)
        for m in markets:
            if m.get("status") in ("open", "active"):
                t = m.get("ticker")
                if t and t not in tickers:
                    tickers.append(t)
        logger.info("Found %d open markets under event %s", len(tickers), args.event)

    return tickers


def check_exposure(mm: MarketMaker, max_exposure: int | None) -> bool:
    """Return True if within exposure limit (or no limit set)."""
    if max_exposure is None:
        return True
    total = len(mm.quotes) * mm.order_size * 2  # 2 sides
    if total >= max_exposure:
        logger.warning(
            "Exposure limit reached (%d contracts >= %d max). Pausing new quotes.",
            total, max_exposure,
        )
        return False
    return True


def main():
    global _shutdown
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    api_key = args.api_key or os.environ.get("KALSHI_API_KEY")
    pk_path = args.private_key or os.environ.get("KALSHI_PRIVATE_KEY_PATH")

    if not api_key:
        logger.error("No API key. Set KALSHI_API_KEY or use --api-key.")
        sys.exit(1)
    if not pk_path:
        logger.error("No private key path. Set KALSHI_PRIVATE_KEY_PATH or use --private-key.")
        sys.exit(1)

    env_label = "DEMO" if args.demo else "PRODUCTION"
    logger.info("Starting Kalshi MM bot [%s]", env_label)

    if args.dry_run:
        logger.info("*** DRY-RUN MODE — no orders will be placed ***")

    client = KalshiClient(api_key, pk_path, demo=args.demo)

    # Verify connectivity
    try:
        bal = client.get_balance()
        balance = bal.get("balance", bal.get("available_balance", "?"))
        logger.info("Connected. Balance: %s", balance)
    except Exception as e:
        logger.error("Failed to connect to Kalshi: %s", e)
        sys.exit(1)

    tickers = resolve_tickers(client, args)
    if not tickers:
        logger.error("No tickers to quote. Use --tickers or --event.")
        sys.exit(1)

    logger.info("Quoting %d markets with %d contracts/side, %ds refresh",
                len(tickers), args.size, args.interval)
    for t in tickers:
        logger.info("  -> %s", t)

    mm = MarketMaker(
        client=client,
        tickers=tickers,
        order_size=args.size,
    )

    # Graceful shutdown on Ctrl+C / SIGTERM
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Main loop ────────────────────────────────────────────────────
    cycle = 0
    while not _shutdown:
        cycle += 1
        t0 = time.time()

        try:
            if not check_exposure(mm, args.max_exposure):
                logger.info("Cycle %d: skipped (exposure limit)", cycle)
            elif args.dry_run:
                _dry_run_cycle(client, tickers)
            else:
                mm.refresh_all()
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("Cycle %d: unexpected error", cycle)

        if cycle % 10 == 0:
            logger.info("Cycle %d | %s", cycle, mm.status_summary())

        elapsed = time.time() - t0
        sleep_time = max(0, args.interval - elapsed)
        if sleep_time > 0 and not _shutdown:
            time.sleep(sleep_time)

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down — cancelling all resting orders...")
    if not args.dry_run:
        mm.cancel_all_quotes()
    logger.info("All quotes cancelled. Bot stopped.")


def _dry_run_cycle(client: KalshiClient, tickers: list[str]):
    """Simulate a refresh cycle without placing orders."""
    for ticker in tickers:
        try:
            market = client.get_market(ticker).get("market", {})
            bid_d = market.get("yes_bid_dollars", "?")
            ask_d = market.get("yes_ask_dollars", "?")
            last_d = market.get("last_price_dollars", "?")
            logger.info(
                "[DRY-RUN] %s: bid=%s ask=%s last=%s",
                ticker, bid_d, ask_d, last_d,
            )
        except Exception:
            logger.exception("[DRY-RUN] %s: failed to fetch", ticker)


if __name__ == "__main__":
    main()
