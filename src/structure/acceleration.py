"""Causal momentum geometry and multimetric exhaustion candidate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .swings import atr


def acceleration_features(bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    logp = np.log(bars.close.astype(float))
    slope = (logp-logp.shift(window))/window
    slope_change = slope-slope.shift(5)
    mean = bars.close.rolling(window, min_periods=window).mean()
    std = bars.close.rolling(window, min_periods=window).std()
    z = (bars.close-mean)/std.replace(0, np.nan)
    cumulative = logp-logp.shift(window)
    relative_volume = bars.volume/bars.volume.shift().rolling(window, min_periods=window).mean()
    current_atr = atr(bars, 14)
    atr_expansion = current_atr/current_atr.shift().rolling(window, min_periods=window).median()
    state = pd.Series("neutral", index=bars.index, dtype=object)
    state.loc[(slope>0)&(slope_change>0)] = "accelerating_up"
    state.loc[(slope<0)&(slope_change<0)] = "accelerating_down"
    state.loc[(slope>0)&(slope_change<0)] = "decelerating_up"
    state.loc[(slope<0)&(slope_change>0)] = "decelerating_down"
    # At least three independent conditions; no RSI-only trigger.
    votes = z.abs().ge(2).astype(int) + relative_volume.ge(1.5).astype(int) + \
            atr_expansion.ge(1.25).astype(int) + (slope.abs().lt(slope.shift().abs())).astype(int)
    exhaustion = votes.ge(3) & cumulative.abs().gt(cumulative.abs().shift().rolling(
        window, min_periods=window).median())
    return pd.DataFrame({"timestamp": bars.timestamp, "slope": slope,
        "slope_change": slope_change, "acceleration": slope_change/5,
        "distance_to_mean": bars.close/mean-1, "price_zscore": z,
        "cumulative_log_return": cumulative, "relative_volume": relative_volume,
        "atr_expansion": atr_expansion, "acceleration_state": state,
        "exhaustion_candidate": exhaustion})
