"""Kalshi Exchange API client with RSA-PSS authentication."""

import base64
import time
import uuid
import logging
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

PROD_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiClient:
    """HTTP client for Kalshi's trading API with RSA-PSS signing."""

    def __init__(self, api_key: str, private_key_path: str, demo: bool = False):
        self.api_key = api_key
        self.base_url = DEMO_BASE_URL if demo else PROD_BASE_URL
        self.session = requests.Session()
        self._load_private_key(private_key_path)

    def _load_private_key(self, path: str):
        with open(path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, timestamp_ms: int, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method}{path}".encode()
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, path: str) -> dict:
        ts = int(time.time() * 1000)
        sig = self._sign(ts, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        path = f"/trade-api/v2{endpoint}"
        url = f"{self.base_url}{endpoint}"
        headers = self._auth_headers(method, path)
        resp = self.session.request(method, url, headers=headers, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── Market data ──────────────────────────────────────────────────

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}")

    def get_markets(self, **params) -> dict:
        return self._request("GET", "/markets", params=params)

    def get_event_markets(self, event_ticker: str) -> list:
        """Fetch all markets for an event, handling pagination."""
        markets = []
        cursor = None
        while True:
            params = {"event_ticker": event_ticker, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            resp = self._request("GET", "/markets", params=params)
            markets.extend(resp.get("markets", []))
            cursor = resp.get("cursor")
            if not cursor:
                break
        return markets

    # ── Portfolio ─────────────────────────────────────────────────────

    def get_balance(self) -> dict:
        return self._request("GET", "/portfolio/balance")

    def get_positions(self, **params) -> dict:
        return self._request("GET", "/portfolio/positions", params=params)

    def get_orders(self, **params) -> dict:
        return self._request("GET", "/portfolio/orders", params=params)

    # ── Order management ─────────────────────────────────────────────

    def create_order(
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
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
            "time_in_force": "good_till_canceled",
            "post_only": post_only,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        else:
            body["client_order_id"] = str(uuid.uuid4())
        return self._request("POST", "/portfolio/orders", json=body)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def batch_create_orders(self, orders: list) -> dict:
        return self._request("POST", "/portfolio/orders/batched", json={"orders": orders})

    def batch_cancel_orders(self, order_ids: list) -> dict:
        orders = [{"order_id": oid} for oid in order_ids]
        return self._request("DELETE", "/portfolio/orders/batched", json={"orders": orders})

    def amend_order(self, order_id: str, **updates) -> dict:
        return self._request("PATCH", f"/portfolio/orders/{order_id}", json=updates)
