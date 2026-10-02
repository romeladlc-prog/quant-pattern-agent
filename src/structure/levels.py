"""ATR/percent clustered swing levels; descriptive strength only.

A level set is always built *as of* one bar: only swings whose
``confirmation_index`` is at or before that bar, and the ATR, close and
volume median known at its close. Adding later bars cannot change it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .swings import atr, confirmed_swings

LEVEL_COLUMNS = ["price_level", "type", "touch_count", "first_touch", "last_touch",
                 "distance_from_current_price", "strength_score", "available_at",
                 "asof_index", "asof_timestamp"]


def _cluster(close: np.ndarray, volume: np.ndarray, timestamps: pd.Series, known: list,
             end: int, current_atr: float, volume_base: float, atr_tolerance: float,
             pct_tolerance: float, min_touches: int) -> list[dict]:
    """Cluster swings already known at bar ``end``; inputs carry no later data.

    ``known`` holds swing records sorted by price, all with confirmation_index <= end.
    """
    current = float(close[end])
    tolerance = max(current_atr*atr_tolerance if pd.notna(current_atr) else 0,
                    current*pct_tolerance)
    groups = []
    for row in known:
        eligible = [g for g in groups if abs(row["price_level"]-g["center"]) <= tolerance]
        if eligible:
            g = min(eligible, key=lambda item: abs(row["price_level"]-item["center"]))
            g["rows"].append(row)
            g["center"] = np.mean([r["price_level"] for r in g["rows"]])
        else: groups.append({"center": row["price_level"], "rows": [row]})
    result = []
    for group in groups:
        rows = group["rows"]
        if len(rows) < min_touches: continue
        kinds = {r["type"] for r in rows}
        kind = "mixed" if len(kinds)>1 else ("resistance" if "swing_high" in kinds else "support")
        last_index = max(r["confirmation_index"] for r in rows)
        recency = 1/(1+(end-last_index)/20)
        mean_volume = np.mean(volume[[r["bar_index"] for r in rows]])
        vol_factor = min(float(mean_volume/volume_base), 2) if volume_base > 0 else 1
        price = float(group["center"])
        result.append({"price_level": price, "type": kind, "touch_count": len(rows),
          "first_touch": min(r["timestamp"] for r in rows), "last_touch": max(r["timestamp"] for r in rows),
          "distance_from_current_price": (price/current)-1, "strength_score":
          round(np.log1p(len(rows))*recency*vol_factor, 4),
          "available_at": max(r["confirmed_at"] for r in rows),
          "asof_index": end, "asof_timestamp": timestamps.iloc[end]})
    return result


def _arrays(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return bars.close.to_numpy(dtype=float), bars.volume.to_numpy(dtype=float)


def _frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=LEVEL_COLUMNS).sort_values(
        ["asof_index", "price_level"]).reset_index(drop=True)


def cluster_levels(bars: pd.DataFrame, swings: pd.DataFrame, asof_index: int | None = None,
                   atr_tolerance: float = 0.5, pct_tolerance: float = 0.005,
                   min_touches: int = 2) -> pd.DataFrame:
    """Levels known at the close of ``asof_index`` (default: the last bar)."""
    if bars.empty or swings.empty: return pd.DataFrame(columns=LEVEL_COLUMNS)
    end = len(bars)-1 if asof_index is None else asof_index
    known = swings.loc[swings.confirmation_index.le(end)].sort_values("price_level", kind="mergesort")
    if known.empty: return pd.DataFrame(columns=LEVEL_COLUMNS)
    prefix = bars.iloc[:end+1]
    close, volume = _arrays(prefix)
    return _frame(_cluster(close, volume, prefix.timestamp, known.to_dict("records"), end,
                           atr(prefix).iloc[-1], prefix.volume.median(),
                           atr_tolerance, pct_tolerance, min_touches))


def build_levels_asof(bars: pd.DataFrame, timestamp, swings: pd.DataFrame | None = None,
                      left_bars: int = 3, right_bars: int = 2, **kwargs) -> pd.DataFrame:
    """Levels available at ``timestamp``: built from the last bar labelled <= it.

    Swings are recomputed on that prefix unless supplied; supplied swings are
    still filtered by ``confirmation_index``, so a later pivot never enters.
    """
    stamp = pd.Timestamp(timestamp)
    position = int(bars.timestamp.searchsorted(stamp, side="right")) - 1
    if position < 0: return pd.DataFrame(columns=LEVEL_COLUMNS)
    if swings is None:
        swings = confirmed_swings(bars.iloc[:position+1], left_bars, right_bars)
    return cluster_levels(bars, swings, position, **kwargs)


def levels_history(bars: pd.DataFrame, swings: pd.DataFrame, atr_tolerance: float = 0.5,
                   pct_tolerance: float = 0.005, min_touches: int = 2) -> pd.DataFrame:
    """Level set as of every bar, one row per (asof_index, level).

    Equal to calling ``cluster_levels(bars, swings, i)`` for each i, but the
    causal ATR and expanding volume median are computed once.
    """
    if bars.empty or swings.empty: return pd.DataFrame(columns=LEVEL_COLUMNS)
    current_atr = atr(bars).to_numpy()
    volume_base = bars.volume.expanding().median().to_numpy()
    close, volume = _arrays(bars)
    ordered = swings.sort_values("price_level", kind="mergesort").to_dict("records")
    records = []
    for end in range(len(bars)):
        known = [row for row in ordered if row["confirmation_index"] <= end]
        if not known: continue
        records += _cluster(close, volume, bars.timestamp, known, end, current_atr[end],
                            volume_base[end], atr_tolerance, pct_tolerance, min_touches)
    return _frame(records)
