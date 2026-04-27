"""Async wrapper around src.kalshi_client.KalshiClient.

Reuses the existing sync client (RSA-PSS auth, batch order endpoints)
and bridges every method into asyncio via to_thread. The 2-second
quoting cadence makes a thread-pool bridge more than adequate; this
keeps a single source of truth for Kalshi API calls instead of
forking an aiohttp duplicate.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)


class KalshiV2Client:
    """Async-friendly view over a sync KalshiClient.

    All public methods are coroutines that run their sync counterpart
    in a worker thread, leaving the event loop free for the bookmaker
    fetch loop.
    """

    def __init__(self, api_key: str, private_key_path: str, demo: bool = True):
        self._sync = KalshiClient(api_key, private_key_path, demo=demo)
        self.demo = demo

    @property
    def base_url(self) -> str:
        return self._sync.base_url

    # ── Market data ──────────────────────────────────────────────────

    async def get_market(self, ticker: str) -> dict:
        return await asyncio.to_thread(self._sync.get_market, ticker)

    async def get_markets(self, **params) -> dict:
        return await asyncio.to_thread(self._sync._request, "GET", "/markets", params=params)

    async def get_event_markets(self, event_ticker: str) -> list:
        return await asyncio.to_thread(self._sync.get_event_markets, event_ticker)

    # ── Portfolio ────────────────────────────────────────────────────

    async def get_balance(self) -> dict:
        return await asyncio.to_thread(self._sync.get_balance)

    async def get_orders(self, **params) -> dict:
        return await asyncio.to_thread(self._sync.get_orders, **params)

    async def get_positions(self, **params) -> dict:
        return await asyncio.to_thread(self._sync.get_positions, **params)

    # ── Order management ────────────────────────────────────────────

    async def batch_create_orders(self, orders: list) -> dict:
        return await asyncio.to_thread(self._sync.batch_create_orders, orders)

    async def batch_cancel_orders(self, order_ids: list) -> dict:
        return await asyncio.to_thread(self._sync.batch_cancel_orders, order_ids)

    async def cancel_order(self, order_id: str) -> dict:
        return await asyncio.to_thread(self._sync.cancel_order, order_id)

    async def create_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int = 1,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        post_only: bool = True,
        client_order_id: Optional[str] = None,
    ) -> dict:
        return await asyncio.to_thread(
            self._sync.create_order,
            ticker, side, action, count,
            yes_price, no_price, post_only, client_order_id,
        )
