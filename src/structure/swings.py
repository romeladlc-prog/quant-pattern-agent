"""Confirmed pivots and a causal ATR reversal alternative."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    prev = bars.close.shift()
    tr = pd.concat([(bars.high - bars.low).abs(), (bars.high - prev).abs(),
                    (bars.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def confirmed_swings(bars: pd.DataFrame, left_bars: int = 3,
                     right_bars: int = 2) -> pd.DataFrame:
    """A pivot at i is published only at i+right_bars; strict ties are excluded."""
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be positive")
    records = []
    for i in range(left_bars, len(bars) - right_bars):
        left = bars.iloc[i-left_bars:i]
        right = bars.iloc[i+1:i+right_bars+1]
        for kind, column, cmp in (("swing_high", "high", np.greater),
                                  ("swing_low", "low", np.less)):
            value = float(bars.iloc[i][column])
            neighbors = pd.concat([left[column], right[column]])
            if np.isfinite(value) and cmp(value, neighbors.to_numpy()).all():
                records.append({"timestamp": bars.iloc[i].timestamp,
                                "confirmed_at": bars.iloc[i+right_bars].timestamp,
                                "bar_index": i, "confirmation_index": i+right_bars,
                                "type": kind, "price_level": value,
                                "confirmation_lag_bars": right_bars,
                                "method": "fractal"})
    return pd.DataFrame(records, columns=["timestamp", "confirmed_at", "bar_index",
        "confirmation_index", "type", "price_level", "confirmation_lag_bars", "method"])


def atr_reversal_swings(bars: pd.DataFrame, atr_multiple: float = 1.5,
                        atr_window: int = 14, reversal_pct: float = 0.0) -> pd.DataFrame:
    """Causal zigzag: a running extreme is confirmed after an ATR/percent reversal.

    Prior extrema are revised only while unconfirmed; each emitted row has a
    confirmation timestamp and the actual, variable bar latency.
    """
    if atr_multiple < 0 or reversal_pct < 0 or (atr_multiple == reversal_pct == 0):
        raise ValueError("positive ATR multiple or percentage required")
    if len(bars) < 2:
        return pd.DataFrame(columns=["timestamp", "confirmed_at", "bar_index",
            "confirmation_index", "type", "price_level", "confirmation_lag_bars", "method"])
    scale = atr(bars, atr_window)
    direction = 0
    high_i = low_i = 0
    records = []
    for i in range(1, len(bars)):
        h, l = float(bars.high.iloc[i]), float(bars.low.iloc[i])
        if direction >= 0:
            if h > float(bars.high.iloc[high_i]): high_i = i
            peak = float(bars.high.iloc[high_i])
            threshold = max(float(scale.iloc[i]) * atr_multiple if pd.notna(scale.iloc[i]) else np.inf,
                            peak * reversal_pct)
            if peak - l >= threshold and i > high_i:
                records.append((high_i, i, "swing_high", peak)); direction = -1; low_i = i
        if direction <= 0:
            if l < float(bars.low.iloc[low_i]): low_i = i
            trough = float(bars.low.iloc[low_i])
            threshold = max(float(scale.iloc[i]) * atr_multiple if pd.notna(scale.iloc[i]) else np.inf,
                            trough * reversal_pct)
            if h - trough >= threshold and i > low_i:
                records.append((low_i, i, "swing_low", trough)); direction = 1; high_i = i
    return pd.DataFrame([{"timestamp": bars.timestamp.iloc[j], "confirmed_at": bars.timestamp.iloc[k],
        "bar_index": j, "confirmation_index": k, "type": kind, "price_level": price,
        "confirmation_lag_bars": k-j, "method": "atr_reversal"} for j,k,kind,price in records],
        columns=["timestamp", "confirmed_at", "bar_index", "confirmation_index",
                 "type", "price_level", "confirmation_lag_bars", "method"])
