"""Market-making engine that maintains a fixed 2-cent spread on Kalshi markets.

For each market the bot is quoting, it keeps exactly two resting limit orders:
  - YES BUY  at  midpoint - 1  (the bid)
  - YES SELL at  midpoint + 1  (the ask)

This guarantees a 2-cent spread. Orders are post-only so we never cross
the book and always earn the maker rebate (if any).

The midpoint is derived from the market's current best bid/ask. If no
book exists, the bot uses the last trade price. If neither is available
the market is skipped.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field

from src.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)

SPREAD = 2  # cents — fixed, non-negotiable per user requirement


@dataclass
class Quote:
    """Represents a two-sided quote on a single market."""
    ticker: str
    bid_price: int  # YES buy price (cents 1-99)
    ask_price: int  # YES sell price (cents 1-99)
    bid_order_id: str | None = None
    ask_order_id: str | None = None


@dataclass
class MarketMaker:
    """Maintains 2-cent spread quotes across a set of Kalshi markets."""

    client: KalshiClient
    tickers: list[str]
    order_size: int = 1          # contracts per side
    quotes: dict[str, Quote] = field(default_factory=dict)
    _active_order_ids: set = field(default_factory=set)

    # ── Public interface ─────────────────────────────────────────────

    def refresh_all(self):
        """Full refresh cycle: pull market data, reconcile orders."""
        for ticker in self.tickers:
            try:
                self._refresh_market(ticker)
            except Exception:
                logger.exception("Error refreshing %s", ticker)

    def cancel_all_quotes(self):
        """Cancel every resting order the bot owns."""
        ids = list(self._active_order_ids)
        if not ids:
            return
        # batch cancel in chunks of 20
        for i in range(0, len(ids), 20):
            chunk = ids[i : i + 20]
            try:
                self.client.batch_cancel_orders(chunk)
                for oid in chunk:
                    self._active_order_ids.discard(oid)
                logger.info("Cancelled %d orders", len(chunk))
            except Exception:
                logger.exception("Batch cancel failed for chunk starting at %d", i)
        self.quotes.clear()

    # ── Internal logic ───────────────────────────────────────────────

    def _refresh_market(self, ticker: str):
        """Refresh the quote for a single market."""
        market = self.client.get_market(ticker).get("market", {})

        if market.get("status") not in ("open", "active"):
            self._cancel_quote(ticker)
            return

        mid = self._calc_midpoint(market)
        if mid is None:
            logger.warning("%s: no midpoint available, skipping", ticker)
            return

        bid = mid - (SPREAD // 2)   # mid - 1
        ask = mid + (SPREAD // 2)   # mid + 1

        # Clamp to valid range
        bid = max(1, min(bid, 98))
        ask = max(2, min(ask, 99))

        # Enforce exactly 2-cent spread after clamping
        if ask - bid != SPREAD:
            if bid <= 1:
                ask = bid + SPREAD
            elif ask >= 99:
                bid = ask - SPREAD
            # Final sanity
            if bid < 1 or ask > 99 or ask - bid != SPREAD:
                logger.warning("%s: can't place valid 2c spread at mid=%d", ticker, mid)
                return

        existing = self.quotes.get(ticker)
        if existing and existing.bid_price == bid and existing.ask_price == ask:
            # Quote is already correct — verify orders still resting
            if self._orders_still_resting(existing):
                logger.debug("%s: quote unchanged at %d/%d", ticker, bid, ask)
                return

        # Need to update — cancel old, place new
        self._cancel_quote(ticker)
        self._place_quote(ticker, bid, ask)

    def _calc_midpoint(self, market: dict) -> int | None:
        """Derive integer midpoint from market data."""
        yes_bid = self._parse_price(market.get("yes_bid"))
        yes_ask = self._parse_price(market.get("yes_ask"))

        if yes_bid is not None and yes_ask is not None:
            return (yes_bid + yes_ask) // 2

        # Fallback: dollar-denominated fields
        bid_d = market.get("yes_bid_dollars")
        ask_d = market.get("yes_ask_dollars")
        if bid_d and ask_d:
            try:
                mid_dollars = (float(bid_d) + float(ask_d)) / 2
                return max(1, min(99, round(mid_dollars * 100)))
            except (ValueError, TypeError):
                pass

        # Fallback: last trade price
        last = market.get("last_price")
        if last is not None:
            return max(1, min(99, int(last)))

        last_d = market.get("last_price_dollars")
        if last_d:
            try:
                return max(1, min(99, round(float(last_d) * 100)))
            except (ValueError, TypeError):
                pass

        return None

    def _parse_price(self, val) -> int | None:
        if val is None:
            return None
        try:
            v = int(val)
            return v if 1 <= v <= 99 else None
        except (ValueError, TypeError):
            return None

    def _place_quote(self, ticker: str, bid: int, ask: int):
        """Place bid + ask as a batch."""
        bid_cid = str(uuid.uuid4())
        ask_cid = str(uuid.uuid4())

        orders = [
            {
                "ticker": ticker,
                "side": "yes",
                "action": "buy",
                "count": self.order_size,
                "yes_price": bid,
                "type": "limit",
                "time_in_force": "good_till_canceled",
                "post_only": True,
                "client_order_id": bid_cid,
            },
            {
                "ticker": ticker,
                "side": "yes",
                "action": "sell",
                "count": self.order_size,
                "yes_price": ask,
                "type": "limit",
                "time_in_force": "good_till_canceled",
                "post_only": True,
                "client_order_id": ask_cid,
            },
        ]

        try:
            resp = self.client.batch_create_orders(orders)
        except Exception:
            logger.exception("%s: failed to place quote %d/%d", ticker, bid, ask)
            return

        bid_oid = None
        ask_oid = None
        for entry in resp.get("orders", []):
            if entry.get("error"):
                logger.error("%s: order error: %s", ticker, entry["error"])
                continue
            order = entry.get("order", {})
            oid = order.get("order_id")
            cid = order.get("client_order_id") or entry.get("client_order_id")
            if cid == bid_cid:
                bid_oid = oid
            elif cid == ask_cid:
                ask_oid = oid
            if oid:
                self._active_order_ids.add(oid)

        self.quotes[ticker] = Quote(
            ticker=ticker,
            bid_price=bid,
            ask_price=ask,
            bid_order_id=bid_oid,
            ask_order_id=ask_oid,
        )
        logger.info("%s: quoted %d/%d (spread=%d)", ticker, bid, ask, ask - bid)

    def _cancel_quote(self, ticker: str):
        """Cancel both sides of an existing quote."""
        existing = self.quotes.pop(ticker, None)
        if not existing:
            return
        ids = [oid for oid in (existing.bid_order_id, existing.ask_order_id) if oid]
        if ids:
            try:
                self.client.batch_cancel_orders(ids)
                for oid in ids:
                    self._active_order_ids.discard(oid)
                logger.info("%s: cancelled old quote", ticker)
            except Exception:
                logger.exception("%s: failed to cancel old quote", ticker)

    def _orders_still_resting(self, quote: Quote) -> bool:
        """Check that both legs of a quote are still resting on the book."""
        for oid in (quote.bid_order_id, quote.ask_order_id):
            if not oid:
                return False
            try:
                resp = self.client._request("GET", f"/portfolio/orders/{oid}")
                order = resp.get("order", {})
                if order.get("status") != "resting":
                    return False
            except Exception:
                return False
        return True

    def status_summary(self) -> str:
        """Human-readable status of all quotes."""
        lines = []
        for ticker, q in sorted(self.quotes.items()):
            lines.append(f"  {ticker}: BID {q.bid_price}¢ / ASK {q.ask_price}¢  (spread {q.ask_price - q.bid_price}¢)")
        if not lines:
            return "No active quotes."
        return "Active quotes:\n" + "\n".join(lines)
