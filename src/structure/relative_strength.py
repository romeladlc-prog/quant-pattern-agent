"""Point-in-time relative price strength against a configurable benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd


def relative_strength(bars: pd.DataFrame, benchmark: pd.DataFrame,
                      benchmark_symbol: str = "QQQ", window: int = 20) -> pd.DataFrame:
    left = bars[["timestamp", "close"]].rename(columns={"close":"asset_close"})
    right = benchmark[["timestamp", "close"]].rename(columns={"close":"benchmark_close"})
    frame = left.merge(right, on="timestamp", how="left", validate="one_to_one")
    # Exact timestamps only: no future benchmark close is ever joined.
    ratio = frame.asset_close/frame.benchmark_close
    for period in (5,20,60):
        frame[f"relative_return_{period}"] = frame.asset_close.pct_change(period) - \
            frame.benchmark_close.pct_change(period)
    frame["price_benchmark_ratio"] = ratio
    frame["ratio_slope"] = (np.log(ratio)-np.log(ratio.shift(window)))/window
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    frame["ratio_zscore"] = (ratio-mean)/std.replace(0,np.nan)
    frame["benchmark"] = benchmark_symbol
    return frame
