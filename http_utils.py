"""Shared HTTPS helpers for MLB API clients."""

from __future__ import annotations

import ssl

import certifi


def default_ssl_context() -> ssl.SSLContext:
    """Build a TLS context from certifi's current trusted CA bundle."""
    return ssl.create_default_context(cafile=certifi.where())
