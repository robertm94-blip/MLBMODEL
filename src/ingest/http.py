"""Shared HTTP client with per-host rate limiting and retry-with-backoff."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)


@dataclass
class HostConfig:
    rate_per_sec: float
    burst: int = 1
    timeout: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)


# Conservative defaults. NBA throttles aggressively and requires browser-like headers.
DEFAULT_HOSTS: dict[str, HostConfig] = {
    "mlb": HostConfig(rate_per_sec=5.0, burst=5, timeout=30.0),
    "nba": HostConfig(
        rate_per_sec=0.6,
        burst=1,
        timeout=30.0,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true",
        },
    ),
}


class _TokenBucket:
    """Simple thread-safe token bucket."""

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self.rate = rate_per_sec
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait)


class HttpClient:
    """Per-host `requests.Session` with rate limiting and retry."""

    def __init__(
        self,
        hosts: dict[str, HostConfig] | None = None,
        max_attempts: int = 4,
        backoff_base: float = 1.0,
    ) -> None:
        self.hosts = dict(hosts or DEFAULT_HOSTS)
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._sessions: dict[str, requests.Session] = {}
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def _config(self, host_key: str) -> HostConfig:
        if host_key not in self.hosts:
            raise KeyError(f"Unknown host_key: {host_key!r}. Known: {sorted(self.hosts)}")
        return self.hosts[host_key]

    def _session(self, host_key: str) -> requests.Session:
        with self._lock:
            sess = self._sessions.get(host_key)
            if sess is None:
                sess = requests.Session()
                sess.headers.update(self._config(host_key).headers)
                self._sessions[host_key] = sess
            return sess

    def _bucket(self, host_key: str) -> _TokenBucket:
        with self._lock:
            bucket = self._buckets.get(host_key)
            if bucket is None:
                cfg = self._config(host_key)
                bucket = _TokenBucket(cfg.rate_per_sec, cfg.burst)
                self._buckets[host_key] = bucket
            return bucket

    def get_json(
        self,
        url: str,
        *,
        host_key: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        cfg = self._config(host_key)
        bucket = self._bucket(host_key)
        session = self._session(host_key)
        attempt = 0
        last_exc: Exception | None = None
        while attempt < self.max_attempts:
            attempt += 1
            bucket.acquire()
            try:
                resp = session.get(url, params=params, timeout=timeout or cfg.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep_backoff(attempt, retry_after=None)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    last_exc = exc
                    self._sleep_backoff(attempt, retry_after=None)
                    continue

            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                log.warning(
                    "HTTP %s on %s (attempt %d/%d); backing off",
                    resp.status_code, url, attempt, self.max_attempts,
                )
                last_exc = requests.HTTPError(f"{resp.status_code} for {url}", response=resp)
                self._sleep_backoff(attempt, retry_after=retry_after)
                continue

            resp.raise_for_status()
            return resp.json()

        assert last_exc is not None
        raise last_exc

    def _sleep_backoff(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            time.sleep(max(0.0, retry_after))
            return
        delay = self.backoff_base * (2 ** (attempt - 1))
        time.sleep(delay)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
