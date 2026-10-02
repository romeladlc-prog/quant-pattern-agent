"""One ALPACA_DATA_FEED setting; IEX by default, SIP never assumed."""
import pytest

from src.data import alpaca_client


def test_default_feed_is_iex(monkeypatch):
    monkeypatch.setattr(alpaca_client, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    assert alpaca_client.data_feed() == "iex"


def test_env_and_explicit_override(monkeypatch):
    monkeypatch.setattr(alpaca_client, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ALPACA_DATA_FEED", "SIP")
    assert alpaca_client.data_feed() == "sip"
    assert alpaca_client.data_feed("iex") == "iex"
    monkeypatch.setenv("ALPACA_DATA_FEED", "nasdaq")
    with pytest.raises(ValueError):
        alpaca_client.data_feed()


def test_requests_carry_configured_feed(monkeypatch):
    seen = {}
    class Client:
        def __init__(self, key, secret): pass
        def get_stock_bars(self, request):
            seen["feed"] = request.feed
            class Response: df = __import__("pandas").DataFrame()
            return Response()
    monkeypatch.setattr(alpaca_client, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(alpaca_client, "StockHistoricalDataClient", Client)
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    raw = alpaca_client.download_raw_bars("ARM", "2025-01-01", "2025-01-10")
    assert str(getattr(seen["feed"], "value", seen["feed"])) == "iex" and raw.attrs["feed"] == "iex"
