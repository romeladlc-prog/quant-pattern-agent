"""Credentials never appear in error messages or tracebacks."""
import traceback

import pytest
import requests

from src.context import liquidity
from src.data import alpaca_client
from src.security import redact_secrets

FRED = "fake-fred-key-for-tests"
ALPACA_KEY, ALPACA_SECRET = "fake-alpaca-key-for-tests", "fake-alpaca-secret-for-tests"


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", FRED)
    monkeypatch.setenv("ALPACA_API_KEY", ALPACA_KEY)
    monkeypatch.setenv("ALPACA_SECRET_KEY", ALPACA_SECRET)
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def rendered(info) -> str:
    return str(info.value) + "".join(traceback.format_exception(info.value))


def assert_clean(text):
    for secret in (FRED, ALPACA_KEY, ALPACA_SECRET):
        assert secret not in text


@pytest.mark.parametrize("error", [requests.HTTPError, requests.ConnectionError, requests.Timeout])
def test_fred_errors_are_redacted(monkeypatch, error):
    def failing_get(url, params=None, timeout=None):
        query = "&".join(f"{k}={v}" for k, v in params.items())
        raise error(f"500 Server Error for url: {url}?{query}")
    monkeypatch.setattr(liquidity.requests, "get", failing_get)
    with pytest.raises(RuntimeError) as info:
        liquidity.fetch_fred_series("fed_assets", "2025-01-01")
    assert "[REDACTED]" in str(info.value)
    assert_clean(rendered(info))


def test_fred_explicit_key_is_redacted(monkeypatch):
    explicit = "fake-explicit-fred-key"
    def failing_get(url, params=None, timeout=None):
        raise requests.HTTPError(f"bad request {params['api_key']}")
    monkeypatch.setattr(liquidity.requests, "get", failing_get)
    with pytest.raises(RuntimeError) as info:
        liquidity.fetch_fred_series("fed_assets", "2025-01-01", api_key=explicit)
    assert explicit not in rendered(info)


def test_alpaca_errors_are_redacted(monkeypatch):
    class FailingClient:
        def __init__(self, key, secret):
            self.key, self.secret = key, secret
        def get_stock_bars(self, request):
            raise ValueError(f"auth failed key={self.key} secret={self.secret}")
    monkeypatch.setattr(alpaca_client, "StockHistoricalDataClient", FailingClient)
    with pytest.raises(RuntimeError) as info:
        alpaca_client.download_raw_bars("ARM", "2025-01-01", "2025-01-10")
    assert_clean(rendered(info))


def test_redact_query_parameters_without_env(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY")
    text = redact_secrets("https://x/obs?series_id=WALCL&api_key=abc123&file_type=json")
    assert "abc123" not in text and "series_id=WALCL" in text
