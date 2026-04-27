"""Shadow Maker quoting engine.

For each (bookmaker game → Kalshi market) mapping, maintains a pair of
post-only YES orders centered on bookmaker's no-vig fair probability:

    target_cents = round(P_f * 100)
    bid_cents    = target_cents - 1   (buy YES)
    ask_cents    = target_cents + 1   (sell YES)

When bookmaker moves the line by more than 2 cents (per the user spec),
all resting orders for that market are cancelled and replaced. Smaller
moves trigger an in-place re-quote so we always sit one tick from fair.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.kalshi_v2_client import KalshiV2Client
from src.odds_snapshot import GameKey, OddsSnapshot

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────

MOVE_TRIGGER_CENTS = 2  # spec: re-quote when |Δ| > 2
PRICE_FLOOR = 2          # need room for bid = target - 1 ≥ 1
PRICE_CEIL = 98          # need room for ask = target + 1 ≤ 99


@dataclass
class MarketMapping:
    """Which Kalshi market we're shadowing for which bookmaker game."""
    game_key: GameKey                  # (date, away_abbr, home_abbr)
    ticker: str                        # Kalshi market ticker
    side: str                          # "yes_home" or "yes_away" — which side YES contract pays out
    excluded: bool = False


@dataclass
class MarketQuote:
    """Live state of a market we're quoting."""
    ticker: str
    target_cents: int = 0
    bid_cents: int = 0
    ask_cents: int = 0
    bid_order_id: Optional[str] = None
    ask_order_id: Optional[str] = None
    last_pf: Optional[float] = None
    quoted: bool = False


# ── Math helpers ────────────────────────────────────────────────────

def pf_to_cents(pf: float) -> int:
    """Map a 0–1 fair probability to a Kalshi price in cents (clamped)."""
    cents = round(pf * 100)
    return max(PRICE_FLOOR, min(PRICE_CEIL, cents))


def derive_quote(pf: float) -> tuple[int, int, int]:
    """Return (target_cents, bid_cents, ask_cents) from fair prob."""
    t = pf_to_cents(pf)
    return t, t - 1, t + 1


# ── Engine ──────────────────────────────────────────────────────────

@dataclass
class ShadowMaker:
    client: KalshiV2Client
    snapshot: OddsSnapshot
    mappings: list[MarketMapping]
    order_size: int = 1
    dry_run: bool = False
    quotes: dict[str, MarketQuote] = field(default_factory=dict)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    # ── Public ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Spawn one reconciliation task per mapped market."""
        active = [m for m in self.mappings if not m.excluded]
        if not active:
            logger.warning("ShadowMaker: no active mappings to quote")
            return
        logger.info("ShadowMaker started for %d markets", len(active))
        tasks = [asyncio.create_task(self._market_loop(m)) for m in active]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise

    def stop(self) -> None:
        self._stop.set()

    async def cancel_all(self) -> None:
        """Cancel every resting order this bot owns. Called on shutdown."""
        ids: list[str] = []
        for q in self.quotes.values():
            if q.bid_order_id:
                ids.append(q.bid_order_id)
            if q.ask_order_id:
                ids.append(q.ask_order_id)
        if not ids:
            logger.info("Shutdown: no resting orders to cancel")
            return
        if self.dry_run:
            logger.info("[DRY-RUN] Would cancel %d orders", len(ids))
            return
        for i in range(0, len(ids), 20):
            chunk = ids[i : i + 20]
            try:
                await self.client.batch_cancel_orders(chunk)
                logger.info("Cancelled %d orders", len(chunk))
            except Exception:
                logger.exception("Cancel failed for chunk starting at %d", i)
        self.quotes.clear()

    # ── Per-market loop ─────────────────────────────────────────────

    async def _market_loop(self, mapping: MarketMapping) -> None:
        """Wait on this game's change_event, reconcile orders on each tick."""
        ticker = mapping.ticker
        self.quotes.setdefault(ticker, MarketQuote(ticker=ticker))
        while not self._stop.is_set():
            state = self.snapshot.get(mapping.game_key)
            if state is None:
                # Bookmaker hasn't seen this game yet — wait briefly and retry
                await asyncio.sleep(1.0)
                continue
            try:
                await asyncio.wait_for(state.change_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            # Snapshot the value, then clear so the next change re-fires the event
            pf = state.away_fair_prob if mapping.side == "yes_away" else state.home_fair_prob
            state.change_event.clear()
            try:
                await self._reconcile(mapping, pf)
            except Exception:
                logger.exception("%s: reconcile failed", ticker)

    # ── Reconciliation ──────────────────────────────────────────────

    async def _reconcile(self, mapping: MarketMapping, pf: float) -> None:
        ticker = mapping.ticker
        quote = self.quotes[ticker]

        new_target, new_bid, new_ask = derive_quote(pf)

        if quote.quoted and quote.target_cents == new_target:
            logger.debug("%s: target unchanged at %d¢", ticker, new_target)
            return

        # Per spec: cancel all orders if line moves > 2 cents
        large_move = (
            quote.quoted
            and abs(new_target - quote.target_cents) > MOVE_TRIGGER_CENTS
        )
        if large_move:
            logger.info(
                "%s: bookmaker moved %d¢→%d¢ (Δ=%d > %d), cancelling and re-quoting",
                ticker, quote.target_cents, new_target,
                abs(new_target - quote.target_cents), MOVE_TRIGGER_CENTS,
            )

        if quote.quoted:
            await self._cancel_quote(quote)

        await self._place_quote(quote, mapping, pf, new_target, new_bid, new_ask)

    # ── Order placement ─────────────────────────────────────────────

    async def _place_quote(
        self,
        quote: MarketQuote,
        mapping: MarketMapping,
        pf: float,
        target: int,
        bid: int,
        ask: int,
    ) -> None:
        ticker = mapping.ticker

        if not (1 <= bid < ask <= 99):
            logger.warning("%s: invalid bid/ask %d/%d (target=%d), skipping", ticker, bid, ask, target)
            return

        if self.dry_run:
            logger.info(
                "[DRY-RUN] %s: would post BID %d¢ / ASK %d¢ (P_f=%.4f, target=%d)",
                ticker, bid, ask, pf, target,
            )
            quote.target_cents = target
            quote.bid_cents = bid
            quote.ask_cents = ask
            quote.last_pf = pf
            quote.quoted = True
            return

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
            resp = await self.client.batch_create_orders(orders)
        except Exception:
            logger.exception("%s: batch_create_orders failed (target=%d)", ticker, target)
            return

        bid_oid: Optional[str] = None
        ask_oid: Optional[str] = None
        for entry in resp.get("orders", []):
            if entry.get("error"):
                # Post-only rejection happens here when the price would cross.
                logger.warning("%s: order error: %s", ticker, entry["error"])
                continue
            order = entry.get("order", {})
            oid = order.get("order_id")
            cid = order.get("client_order_id") or entry.get("client_order_id")
            if cid == bid_cid:
                bid_oid = oid
            elif cid == ask_cid:
                ask_oid = oid

        quote.target_cents = target
        quote.bid_cents = bid
        quote.ask_cents = ask
        quote.bid_order_id = bid_oid
        quote.ask_order_id = ask_oid
        quote.last_pf = pf
        quote.quoted = bool(bid_oid or ask_oid)
        if quote.quoted:
            logger.info(
                "%s: quoted BID %d¢ / ASK %d¢ (P_f=%.4f)",
                ticker, bid, ask, pf,
            )
        else:
            logger.warning("%s: no orders accepted (likely post-only rejection)", ticker)

    async def _cancel_quote(self, quote: MarketQuote) -> None:
        ids = [oid for oid in (quote.bid_order_id, quote.ask_order_id) if oid]
        quote.bid_order_id = None
        quote.ask_order_id = None
        quote.quoted = False
        if not ids or self.dry_run:
            return
        try:
            await self.client.batch_cancel_orders(ids)
        except Exception:
            logger.exception("%s: cancel failed for ids %s", quote.ticker, ids)

    # ── Status ──────────────────────────────────────────────────────

    def status_summary(self) -> str:
        if not self.quotes:
            return "No quotes yet."
        lines = []
        for ticker, q in sorted(self.quotes.items()):
            if not q.quoted:
                lines.append(f"  {ticker}: (no quote)")
                continue
            lines.append(
                f"  {ticker}: BID {q.bid_cents}¢ / ASK {q.ask_cents}¢  (target {q.target_cents}¢, P_f={q.last_pf:.4f})"
            )
        return "Active quotes:\n" + "\n".join(lines)
