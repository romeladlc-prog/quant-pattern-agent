from src.data.alpaca_client import download_historical_bars


if __name__ == "__main__":
    bars = download_historical_bars(
        symbol="ARM",
        start="2025-01-01",
        end="2025-01-31",
        timeframe="1Day",
    )
    print(bars[["timestamp", "open", "high", "low", "close", "volume"]].tail(10).to_string(index=False))
