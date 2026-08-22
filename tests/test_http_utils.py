"""Tests for http_utils.fetch_page_with_retry."""

import urllib.error
from unittest.mock import patch

import pytest

from http_utils import fetch_page_with_retry


class _FakeResp:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_fetch_page_with_retry_success():
    with patch("http_utils.urllib.request.urlopen") as mock_open:
        mock_open.return_value = _FakeResp(b'{"a": 1}')
        assert fetch_page_with_retry("https://example.test/x") == {"a": 1}
    mock_open.assert_called_once()


class _Flaky:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise urllib.error.HTTPError("url", 500, "boom", {}, None)
        return _FakeResp(b'{"ok": true}')


def test_fetch_page_with_retry_retries_then_succeeds():
    flaky = _Flaky()
    with patch("http_utils.urllib.request.urlopen", flaky):
        assert fetch_page_with_retry(
            "https://example.test/x", max_retries=4, backoff_base=0.001
        ) == {"ok": True}
    assert flaky.calls == 3


def test_fetch_page_with_retry_gives_up():
    def always_fail(*args, **kwargs):
        raise urllib.error.URLError("nope")

    with patch("http_utils.urllib.request.urlopen", side_effect=always_fail):
        with pytest.raises(RuntimeError, match="GET failed after 2 tries"):
            fetch_page_with_retry(
                "https://example.test/x", max_retries=2, backoff_base=0.001
            )
