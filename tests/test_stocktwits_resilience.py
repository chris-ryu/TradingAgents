"""StockTwits fetch: transport-error resilience (#1024), crypto symbol
mapping (#1113), and per-status HTTP handling.

StockTwits lists crypto under ``<BASE>.X`` (Yahoo's ``BTC-USD`` 404s), and any
transport error must degrade to a placeholder rather than raise. HTTP statuses
are not interchangeable: a 404 means the symbol is not listed (benign), a 429
is a retryable throttle, and anything else is an outage worth naming.
"""

from __future__ import annotations

import http.client
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import stocktwits


def _raise(exc):
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            raise exc
    return _Resp()


def _json_resp(payload):
    """A urlopen-shaped context manager returning ``payload`` as JSON bytes."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            return json.dumps(payload).encode()
    return _Resp()


@pytest.mark.unit
class TestStockTwitsResilience:
    @pytest.mark.parametrize(
        "exc",
        [
            http.client.IncompleteRead(b""),
            HTTPError("url", 503, "down", {}, None),
            TimeoutError("slow"),
        ],
    )
    def test_transport_errors_return_placeholder(self, exc):
        with patch.object(stocktwits, "urlopen", return_value=_raise(exc)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "unavailable" in out.lower()
        assert out.startswith("<stocktwits unavailable")


@pytest.mark.unit
class TestStockTwitsCryptoSymbols:
    @pytest.mark.parametrize(
        ("ticker", "expected"),
        [
            ("BTC-USD", "BTC.X"),
            ("eth-usd", "ETH.X"),
            ("SOL-USD", "SOL.X"),
            ("BTCUSD", "BTC.X"),      # undashed broker form
            ("BTC-USDT", "BTC.X"),    # stablecoin quote
            ("AMD", "AMD"),
            ("BRK-B", "BRK-B"),       # dashed class share: untouched
            ("GOLD", "GOLD"),         # real equity (aliases elsewhere): untouched here
            ("XYZ-USD", "XYZ-USD"),   # unknown base: not treated as crypto
        ],
    )
    def test_symbol_mapping(self, ticker, expected):
        assert stocktwits._stocktwits_symbol(ticker) == expected

    def test_crypto_pair_requests_dot_x_endpoint(self):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            raise TimeoutError("stop after capturing the URL")

        with patch.object(stocktwits, "urlopen", side_effect=fake_urlopen):
            stocktwits.fetch_stocktwits_messages("BTC-USD")
        assert "/symbol/BTC.X.json" in seen["url"]


@pytest.mark.unit
class TestSymbolNotFoundIsNotAnOutage:
    """A 404 is StockTwits saying "no such symbol", not "service down" — the
    analyst must see the same "no messages" signal as an empty stream, or it
    hedges its report over an infrastructure failure that never happened.
    """

    def test_404_reports_no_messages(self):
        err = HTTPError("url", 404, "Not Found", {}, None)
        with patch.object(stocktwits, "urlopen", side_effect=err):
            out = stocktwits.fetch_stocktwits_messages("ZZZZ")
        assert "unavailable" not in out.lower()
        assert "no StockTwits messages" in out
        assert "$ZZZZ" in out


@pytest.mark.unit
class TestCloudflareChallengeIsRetried:
    """Cloudflare fronts the API and answers a fraction of requests with a 403
    "Just a moment..." HTML challenge — observed on the *same* URL between two
    successful 404s, so it is transient and unrelated to symbol validity.
    Treating it as terminal blanks sentiment for a perfectly valid ticker.
    """

    def test_403_then_success_retries_once(self):
        err = HTTPError("url", 403, "Forbidden", {"Content-Type": "text/html"}, None)
        with patch.object(stocktwits, "urlopen",
                          side_effect=[err, _json_resp({"messages": []})]) as op, \
             patch.object(stocktwits.time, "sleep"):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert op.call_count == 2
        assert "unavailable" not in out.lower()   # recovered, not degraded

    def test_403_twice_gives_up_and_names_the_status(self):
        err = HTTPError("url", 403, "Forbidden", {"Content-Type": "text/html"}, None)
        with patch.object(stocktwits, "urlopen", side_effect=[err, err]) as op, \
             patch.object(stocktwits.time, "sleep"):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert op.call_count == 2
        assert "403" in out


@pytest.mark.unit
class TestRateLimitBackoff:
    """429 is retryable; reddit.py already backs off once for the same reason."""

    def test_429_then_success_retries_once(self):
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(stocktwits, "urlopen",
                          side_effect=[err, _json_resp({"messages": []})]) as op, \
             patch.object(stocktwits.time, "sleep") as slept:
            stocktwits.fetch_stocktwits_messages("NVDA")
        assert op.call_count == 2          # original + exactly one retry
        slept.assert_called_once()         # backed off before retrying

    def test_429_twice_gives_up_and_names_the_status(self):
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(stocktwits, "urlopen", side_effect=[err, err]) as op, \
             patch.object(stocktwits.time, "sleep"):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert op.call_count == 2          # one retry, then gives up cleanly
        assert "429" in out

    def test_headerless_backoff_uses_the_measured_floor(self):
        """Without Retry-After we must wait out the same measured floor reddit
        uses: retries at 5s, 8s, 10s and 30s all still 429, only 60s succeeds,
        so a shorter wait spends the one retry on a request that cannot work."""
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(stocktwits, "urlopen",
                          side_effect=[err, _json_resp({"messages": []})]), \
             patch.object(stocktwits.time, "sleep") as slept:
            stocktwits.fetch_stocktwits_messages("NVDA")
        waited = slept.call_args[0][0]
        assert 48.0 <= waited <= 72.0, f"backed off {waited}s, expected ~60s (+/-20% jitter)"

    def test_retry_after_header_is_honoured(self):
        err = HTTPError("url", 429, "Too Many Requests", {"Retry-After": "12"}, None)
        with patch.object(stocktwits, "urlopen",
                          side_effect=[err, _json_resp({"messages": []})]), \
             patch.object(stocktwits.time, "sleep") as slept:
            stocktwits.fetch_stocktwits_messages("NVDA")
        slept.assert_called_once_with(12.0)


@pytest.mark.unit
class TestPlaceholderNamesTheStatus:
    """``<stocktwits unavailable: HTTPError>`` is undiagnosable — the operator
    cannot tell a throttle from an outage without the status code."""

    def test_503_placeholder_carries_the_code(self):
        err = HTTPError("url", 503, "Service Unavailable", {}, None)
        with patch.object(stocktwits, "urlopen", side_effect=err):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "503" in out
