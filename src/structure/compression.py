"""Rolling, price-scale-free compression and expansion descriptors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .swings import atr


def compression_features(bars: pd.DataFrame, window: int = 20,
                         history: int = 60, low_quantile: float = .2,
                         high_quantile: float = .8) -> pd.DataFrame:
    close = bars.close.astype(float)
    relative_atr = atr(bars, window) / close
    std = close.rolling(window, min_periods=window).std()
    bandwidth = 4*std/close.rolling(window, min_periods=window).mean()
    rv = np.log(close).diff().rolling(window, min_periods=window).std()
    mean_range = ((bars.high-bars.low)/close).rolling(window, min_periods=window).mean()
    components = pd.DataFrame({"relative_atr": relative_atr, "bollinger_bandwidth": bandwidth,
                               "realized_volatility": rv, "mean_range": mean_range})
    # Prior distribution only: today's observation does not set its own threshold.
    low = components.shift().rolling(history, min_periods=max(20,history//2)).quantile(low_quantile)
    high = components.shift().rolling(history, min_periods=max(20,history//2)).quantile(high_quantile)
    votes_low = components.le(low).sum(axis=1)
    votes_high = components.ge(high).sum(axis=1)
    valid = low.notna().sum(axis=1).ge(3)
    compressed = valid & votes_low.ge(3)
    expanded = valid & votes_high.ge(3)
    duration = []; run = 0
    for value in compressed:
        run = run+1 if value else 0; duration.append(run)
    return components.assign(timestamp=bars.timestamp.to_numpy(),
        volatility_compression=compressed.to_numpy(), volatility_expansion=expanded.to_numpy(),
        compression_duration=duration)
