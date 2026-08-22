"""Shared HTTPS helpers for MLB API clients."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

import certifi

DEFAULT_USER_AGENT = "mlb-analysis-toolkit/1.0"


def default_ssl_context() -> ssl.SSLContext:
    """Build a TLS context from certifi's current trusted CA bundle."""
    return ssl.create_default_context(cafile=certifi.where())


def fetch_page_with_retry(
    url: str,
    params: Optional[dict[str, Any]] = None,
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 60.0,
    max_retries: int = 4,
    backoff_base: float = 2.0,
    backoff_max: float = 8.0,
    ssl_context: Optional[ssl.SSLContext] = None,
) -> Any:
    """
    Perform a single HTTP GET and return the parsed JSON body.

    Query parameters are URL-encoded into ``url`` when ``params`` is given.
    Transient failures (HTTP errors, network errors, timeouts, bad JSON) are
    retried with exponential backoff up to ``max_retries`` attempts. A
    RuntimeError is raised if every attempt fails.
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req_headers = {"User-Agent": DEFAULT_USER_AGENT,
                   "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    ctx = ssl_context if ssl_context is not None else default_ssl_context()
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.load(resp)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as err:
            last_err = err
            if attempt < max_retries:
                time.sleep(min(backoff_base ** attempt, backoff_max))
    raise RuntimeError(
        f"GET failed after {max_retries} tries: {url}") from last_err
