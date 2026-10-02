"""Causal close-based breakout events, including observed failures.

A crossing at bar i is tested only against levels known at the close of bar
i-1: as-of levels rebuilt from swings confirmed by then. Follow-up events
(confirmation, failure) are dated at the later bar where they are observed.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from .levels import levels_history
from .swings import atr, confirmed_swings

EVENT_COLUMNS = ["timestamp", "bar_index", "level", "level_type", "level_touch_count",
                 "level_asof_timestamp", "direction", "state", "distance_atr",
                 "relative_volume", "outside_bars", "retest", "failure_bars",
                 "breakout_amplitude", "confidence"]


def _scan(bars: pd.DataFrame, levels_before: Callable[[int], Iterable[dict]],
          min_atr_distance: float, min_relative_volume: float, confirm_bars: int,
          fail_within: int) -> pd.DataFrame:
    if bars.empty: return pd.DataFrame(columns=EVENT_COLUMNS)
    av = atr(bars).to_numpy()
    rv = (bars.volume/bars.volume.shift().rolling(20, min_periods=5).mean()).to_numpy()
    closes, highs, lows = (bars[c].to_numpy(dtype=float) for c in ("close", "high", "low"))
    stamps = bars.timestamp
    events = []
    for i in range(1, len(bars)):
        if pd.isna(av[i]) or av[i] <= 0: continue
        prev, close = closes[i-1], closes[i]
        for level in levels_before(i):
            price = float(level["price_level"])
            direction = 1 if prev <= price < close else -1 if prev >= price > close else 0
            if not direction: continue
            distance = float(abs(close-price)/av[i])
            if distance < min_atr_distance: continue
            relative = float(rv[i]) if pd.notna(rv[i]) else None
            volume_ok = relative is not None and relative >= min_relative_volume
            base = {"level":price, "level_type":level.get("type"),
                    "level_touch_count":level.get("touch_count"),
                    "level_asof_timestamp":level.get("asof_timestamp"),
                    "direction":"up" if direction==1 else "down",
                    "distance_atr":distance, "relative_volume":relative,
                    "breakout_amplitude":float(abs(close-price))}
            events.append({**base, "timestamp":stamps.iloc[i], "bar_index":i,
                "state":"breakout_candidate", "outside_bars":1, "retest":False, "failure_bars":None,
                "confidence":round(min(1., .25+.25*min(distance,1)+.25*volume_ok),3)})
            retest = False
            for j in range(i+1, min(len(bars),i+max(confirm_bars,fail_within)+1)):
                c = closes[j]
                if direction*(c-price) <= 0:
                    if j-i <= fail_within:
                        events.append({**base, "timestamp":stamps.iloc[j], "bar_index":j,
                            "state":"fake_breakout_up" if direction==1 else "fake_breakout_down",
                            "outside_bars":j-i, "retest":retest, "failure_bars":j-i,
                            "confidence":round(min(1.,.5+.25*min(distance,1)+.25*(j-i<=2)),3)})
                    break
                if (direction==1 and lows[j]<=price or
                    direction==-1 and highs[j]>=price): retest=True
                if j-i+1 == confirm_bars and volume_ok:
                    events.append({**base, "timestamp":stamps.iloc[j], "bar_index":j,
                        "state":"breakout_confirmed", "outside_bars":confirm_bars,
                        "retest":retest, "failure_bars":None,
                        "confidence":round(min(1.,.5+.25*min(distance,1)+.25*retest),3)})
    return pd.DataFrame(events, columns=EVENT_COLUMNS).sort_values(
        ["timestamp", "bar_index"], kind="mergesort").reset_index(drop=True)


def detect_breakouts(bars: pd.DataFrame, swings: pd.DataFrame | None = None,
                     left_bars: int = 3, right_bars: int = 2,
                     min_atr_distance: float = .1, min_relative_volume: float = 1.,
                     confirm_bars: int = 2, fail_within: int = 5,
                     level_kwargs: dict | None = None,
                     history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Historical breakouts against as-of levels; the default for every caller.

    ``history`` may be a precomputed ``levels_history`` frame for these bars.
    """
    if bars.empty: return pd.DataFrame(columns=EVENT_COLUMNS)
    if history is None:
        if swings is None:
            swings = confirmed_swings(bars, left_bars, right_bars)
        history = levels_history(bars, swings, **(level_kwargs or {}))
    by_bar = {int(k): g.to_dict("records") for k, g in history.groupby("asof_index")}
    return _scan(bars, lambda i: by_bar.get(i-1, []), min_atr_distance,
                 min_relative_volume, confirm_bars, fail_within)


def detect_breakouts_fixed_levels(bars: pd.DataFrame, levels: pd.DataFrame,
                                  min_atr_distance: float = .1, min_relative_volume: float = 1.,
                                  confirm_bars: int = 2, fail_within: int = 5) -> pd.DataFrame:
    """Breakouts of externally defined levels whose price is known at ``available_at``.

    Never pass ``cluster_levels`` output from a full sample here: its prices can
    include later swings. A level is usable from the first bar after
    ``available_at``.
    """
    if "available_at" not in levels:
        raise ValueError("fixed levels need available_at")
    records = levels.to_dict("records")
    stamps = bars.timestamp
    return _scan(bars, lambda i: [r for r in records if r["available_at"] < stamps.iloc[i]],
                 min_atr_distance, min_relative_volume, confirm_bars, fail_within)
