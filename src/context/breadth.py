"""Breadth for a fixed, disclosed ten-stock Nasdaq-heavy basket."""
from __future__ import annotations

import numpy as np
import pandas as pd

BREADTH_BASKET = ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "AMD")


def calculate_breadth(bars_by_ticker: dict[str, pd.DataFrame], min_coverage: int = 8) -> pd.DataFrame:
    """Equal-member breadth; fixed 2026 basket entails survivorship bias historically."""
    missing = set(BREADTH_BASKET) - set(bars_by_ticker)
    if missing:
        raise ValueError(f"Missing basket members: {sorted(missing)}")
    closes = pd.concat({ticker: bars_by_ticker[ticker].set_index("timestamp").close.astype(float)
                        for ticker in BREADTH_BASKET}, axis=1).sort_index()
    highs = pd.concat({ticker: bars_by_ticker[ticker].set_index("timestamp").high.astype(float)
                       for ticker in BREADTH_BASKET}, axis=1).reindex(closes.index)
    lows = pd.concat({ticker: bars_by_ticker[ticker].set_index("timestamp").low.astype(float)
                      for ticker in BREADTH_BASKET}, axis=1).reindex(closes.index)
    out = pd.DataFrame(index=closes.index)
    out["members_available"] = closes.notna().sum(axis=1)
    for window in (20, 50, 200):
        sma = closes.rolling(window, min_periods=window).mean()
        eligible = sma.notna() & closes.notna()
        denominator = eligible.sum(axis=1)
        out[f"breadth_sma{window}"] = ((closes > sma) & eligible).sum(axis=1) / denominator.replace(0, np.nan)
        out.loc[denominator < min_coverage, f"breadth_sma{window}"] = np.nan
    previous = closes.shift(1)
    comparable = closes.notna() & previous.notna()
    out["advances"] = ((closes > previous) & comparable).sum(axis=1)
    out["declines"] = ((closes < previous) & comparable).sum(axis=1)
    out["advance_decline_ratio"] = out.advances / out.declines.replace(0, np.nan)
    prior_high = highs.shift(1).rolling(252, min_periods=252).max()
    prior_low = lows.shift(1).rolling(252, min_periods=252).min()
    out["new_highs_252d"] = (closes > prior_high).sum(axis=1).where(prior_high.notna().sum(axis=1) >= min_coverage)
    out["new_lows_252d"] = (closes < prior_low).sum(axis=1).where(prior_low.notna().sum(axis=1) >= min_coverage)
    out.loc[out.members_available < min_coverage,
            ["advances", "declines", "advance_decline_ratio"]] = np.nan
    out["available_at"] = (out.index.tz_convert("America/New_York").normalize() +
                           pd.Timedelta(hours=16, minutes=30)).tz_convert("UTC")
    return out
