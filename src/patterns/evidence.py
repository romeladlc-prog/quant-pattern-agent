"""Causal per-bar evidence frame for the Pattern Engine.

Row i holds only what is known at the close of bar i:
* features and structure descriptors use trailing windows (Phases 3 and 6);
* swings count from ``confirmation_index`` (their ``confirmed_at``);
* support/resistance come from Phase 6.1 as-of levels for bar i;
* breakout events come from Phase 6.1 ``detect_breakouts`` (each tested
  against levels known at bar i-1) and are attached to the bar where observed;
* quant and external context are supplied already aligned and causal;
* other timeframes join by ``close_time`` (only bars closed by this bar's close;
  incomplete intraday blocks and weeks are dropped first).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time

import numpy as np
import pandas as pd

from src.features.statistical_features import calculate_statistical_features
from src.structure.acceleration import acceleration_features
from src.structure.breakouts import detect_breakouts
from src.structure.compression import compression_features
from src.structure.gaps import detect_gaps
from src.structure.levels import levels_history
from src.structure.relative_strength import relative_strength
from src.structure.swings import atr, confirmed_swings
from src.structure.trend_structure import classify_swings

from .config import BREAKOUT_MAX_AGE_BARS, GAP_FILTER, THRESHOLDS, TIMEFRAME_PREFIX

NY = "America/New_York"
MTF_COLUMNS = ["structural_state", "slope", "zscore_20", "acceleration_state",
               "breakout_state", "volatility_expansion"]


@dataclass
class EvidenceBundle:
    timeframe: str
    frame: pd.DataFrame
    events: pd.DataFrame
    meta: dict = field(default_factory=dict)


def bar_close_times(bars: pd.DataFrame, timeframe: str) -> pd.Series:
    """When each bar's OHLCV is final. Patterns at bar i are evaluated then."""
    if "close_time" in bars:
        return pd.to_datetime(bars.close_time, utc=True).reset_index(drop=True)
    stamps = pd.to_datetime(bars.timestamp, utc=True).reset_index(drop=True)
    if timeframe == "1Week" and "source_last_session_close" in bars:
        return pd.to_datetime(bars.source_last_session_close, utc=True).reset_index(drop=True)
    if timeframe in ("1Day", "1Week"):
        # Label + 12h lands on the session date for NY-midnight and UTC-midnight labels.
        days = (stamps + pd.Timedelta(hours=12)).dt.tz_convert(NY).dt.date
        offset = 4 if timeframe == "1Week" else 0
        return pd.Series([pd.Timestamp(datetime.combine(d, time(16)), tz=NY).tz_convert("UTC")
                          + pd.Timedelta(days=offset) for d in days])
    hours = {"4Hour": 4, "1Hour": 1}.get(timeframe)
    if hours is None:
        raise ValueError(f"unsupported timeframe {timeframe}")
    ends = stamps + pd.Timedelta(hours=hours)
    session_end = [pd.Timestamp(datetime.combine(d, time(16)), tz=NY).tz_convert("UTC")
                   for d in stamps.dt.tz_convert(NY).dt.date]
    return pd.Series([min(a, b) for a, b in zip(ends, session_end)])


def complete_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Drop partial intraday blocks and partial weeks before any analysis."""
    if "is_complete" in bars:
        bars = bars.loc[bars.is_complete.astype(bool)]
    return bars.reset_index(drop=True)


def _state(high_label, low_label) -> str:
    if high_label is None or low_label is None:
        return "insufficient_swings"
    if high_label == "HH" and low_label == "HL": return "uptrend"
    if high_label == "LH" and low_label == "LL": return "downtrend"
    if high_label == "EQ" or low_label == "EQ" or (high_label == "LH" and low_label == "HL"):
        return "range"
    return "transition"


def structure_asof(n: int, swings: pd.DataFrame) -> pd.DataFrame:
    """Per-bar structural state and last pivots from swings confirmed by each bar."""
    ordered = classify_swings(swings).sort_values(["confirmation_index", "bar_index"],
                                                   kind="mergesort")
    by_bar: dict[int, list[dict]] = {}
    for row in ordered.to_dict("records"):
        by_bar.setdefault(int(row["confirmation_index"]), []).append(row)
    last = {"swing_high": [], "swing_low": []}
    records = []
    for i in range(n):
        new_high = new_low = None
        for row in by_bar.get(i, []):
            last[row["type"]].append(row)
            if row["type"] == "swing_high": new_high = row
            else: new_low = row
        highs, lows = last["swing_high"], last["swing_low"]
        high_label = highs[-1]["structure_label"] if highs else None
        low_label = lows[-1]["structure_label"] if lows else None
        record = {"structural_state": _state(high_label, low_label),
                  "last_high_label": high_label, "last_low_label": low_label}
        for key, items in (("high", highs), ("low", lows)):
            for depth, tag in ((1, "last"), (2, "prev")):
                item = items[-depth] if len(items) >= depth else None
                record[f"{tag}_{key}"] = item["price_level"] if item else np.nan
                record[f"{tag}_{key}_index"] = item["bar_index"] if item else np.nan
        record["new_swing_high"] = new_high is not None
        record["new_swing_low"] = new_low is not None
        records.append(record)
    frame = pd.DataFrame(records)
    if n:
        lag = THRESHOLDS["structure_change_bars"]
        previous = frame.structural_state.shift(lag)
        frame["structure_changed_recent"] = (frame.structural_state != previous).where(previous.notna())
    return frame


def nearest_levels(history: pd.DataFrame, close: np.ndarray, current_atr: np.ndarray) -> pd.DataFrame:
    n = len(close)
    out = pd.DataFrame({"nearest_support": np.nan, "nearest_resistance": np.nan,
                        "support_touches": np.nan, "resistance_touches": np.nan}, index=range(n))
    if history.empty:
        return out.assign(support_dist_atr=np.nan, resistance_dist_atr=np.nan)
    frame = history.assign(_close=close[history.asof_index.astype(int).to_numpy()])
    for side, below, kinds, pick in (("support", True, ["support", "mixed"], "idxmax"),
                                     ("resistance", False, ["resistance", "mixed"], "idxmin")):
        mask = (frame.price_level < frame._close) if below else (frame.price_level > frame._close)
        eligible = frame.loc[mask & frame.type.isin(kinds)]
        if eligible.empty:
            continue
        chosen = eligible.loc[getattr(eligible.groupby("asof_index").price_level, pick)()]
        positions = chosen.asof_index.astype(int).to_numpy()
        out.loc[positions, f"nearest_{side}"] = chosen.price_level.to_numpy()
        out.loc[positions, f"{side}_touches"] = chosen.touch_count.to_numpy()
    safe_atr = np.where(current_atr > 0, current_atr, np.nan)
    out["support_dist_atr"] = (close - out.nearest_support.to_numpy())/safe_atr
    out["resistance_dist_atr"] = (out.nearest_resistance.to_numpy() - close)/safe_atr
    return out


def breakout_recency(n: int, events: pd.DataFrame, max_age: int = BREAKOUT_MAX_AGE_BARS) -> pd.DataFrame:
    """Phase 6.1 recency per bar: latest breakout/fake event at most max_age bars old."""
    out = pd.DataFrame({"breakout_state": "inactive", "breakout_direction": None,
                        "breakout_age": np.nan, "fake_breakout_state": "inactive",
                        "fake_breakout_age": np.nan}, index=range(n))
    if events.empty:
        return out
    latest = {"breakout": None, "fake": None}
    grouped = {int(k): g.to_dict("records") for k, g in events.groupby("bar_index")}
    for i in range(n):
        for event in grouped.get(i, []):
            kind = "fake" if str(event["state"]).startswith("fake") else "breakout"
            latest[kind] = event
        for kind, prefix in (("breakout", "breakout"), ("fake", "fake_breakout")):
            event = latest[kind]
            if event is not None and i - int(event["bar_index"]) <= max_age:
                out.at[i, f"{prefix}_state"] = event["state"]
                out.at[i, f"{prefix}_age"] = i - int(event["bar_index"])
                if kind == "breakout":
                    out.at[i, "breakout_direction"] = event["direction"]
    return out


def filtered_gaps(bars: pd.DataFrame, current_atr: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Phase 6 gaps that pass both size filters, per bar; counts what is discarded."""
    n = len(bars)
    out = pd.DataFrame({"gap_up_recent": False, "gap_down_recent": False}, index=range(n))
    gaps = detect_gaps(bars)
    opened = gaps.loc[gaps.state.eq("unfilled_gap")] if not gaps.empty else gaps
    total = len(opened)
    if total == 0:
        return out, {"gaps_detected": 0, "gaps_kept": 0, "gaps_discarded": 0}
    keep = opened.size_pct.ge(GAP_FILTER["min_pct"]) & \
        pd.to_numeric(opened.size_atr, errors="coerce").ge(GAP_FILTER["min_atr"])
    kept = opened.loc[keep]
    position = {stamp: k for k, stamp in enumerate(bars.timestamp)}
    recent = GAP_FILTER["recent_bars"]
    for row in kept.itertuples():
        k = position[row.gap_timestamp]
        column = "gap_up_recent" if row.type == "gap_up" else "gap_down_recent"
        out.loc[k:min(n-1, k+recent-1), column] = True
    return out, {"gaps_detected": total, "gaps_kept": int(keep.sum()),
                 "gaps_discarded": int(total - keep.sum())}


def _percentile_of_last(values: np.ndarray) -> float:
    return float((values[:-1] < values[-1]).mean())


def build_evidence(bars: pd.DataFrame, timeframe: str, *, benchmark: pd.DataFrame | None = None,
                   quant: pd.DataFrame | None = None, external: pd.DataFrame | None = None,
                   light: bool = False) -> EvidenceBundle:
    """Assemble the causal evidence frame. ``light`` skips levels and breakouts
    (used for context timeframes that only provide descriptive state)."""
    bars = complete_bars(bars)
    n = len(bars)
    if n == 0 or not bars.timestamp.is_monotonic_increasing or bars.timestamp.duplicated().any():
        raise ValueError("bars must be nonempty, unique and sorted")
    close = bars.close.astype(float).to_numpy()
    frame = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    frame["close_time"] = pd.to_datetime(bar_close_times(bars, timeframe), utc=True).set_axis(frame.index)
    current_atr = atr(bars, 14)
    frame["atr"] = current_atr.to_numpy()
    features = calculate_statistical_features(bars).reset_index(drop=True)
    for column in ("log_return", "zscore_20", "momentum_5", "momentum_20",
                   "realized_volatility_20", "rolling_slope_20", "rolling_mean_20"):
        frame[column] = features[column].to_numpy()
    frame["slope_20_pct"] = frame.rolling_slope_20/frame.close
    frame["rv20_percentile"] = frame.realized_volatility_20.rolling(252, min_periods=60).apply(
        _percentile_of_last, raw=True)
    acceleration = acceleration_features(bars).reset_index(drop=True)
    for column in ("slope", "slope_change", "price_zscore", "distance_to_mean",
                   "cumulative_log_return", "relative_volume", "atr_expansion",
                   "acceleration_state", "exhaustion_candidate"):
        frame[column] = acceleration[column].to_numpy()
    recent = THRESHOLDS["recent_bars"]
    for state, column in (("accelerating_up", "accel_up_recent"),
                          ("accelerating_down", "accel_down_recent")):
        frame[column] = frame.acceleration_state.eq(state).astype(int).rolling(
            recent, min_periods=1).max().astype(bool)
    compression = compression_features(bars).reset_index(drop=True)
    for column in ("volatility_compression", "volatility_expansion", "compression_duration"):
        frame[column] = compression[column].to_numpy()
    # Compression in the previous 20 bars, excluding the current bar.
    frame["compression_recent"] = frame.volatility_compression.astype(int).shift(1).rolling(
        20, min_periods=1).max().fillna(0).astype(bool)
    swings = confirmed_swings(bars)
    frame = pd.concat([frame, structure_asof(n, swings)], axis=1)
    meta = {"bars": n, "first": bars.timestamp.iloc[0], "last": bars.timestamp.iloc[-1]}
    events = pd.DataFrame()
    if not light:
        history = levels_history(bars, swings)
        frame = pd.concat([frame, nearest_levels(history, close, frame.atr.to_numpy())], axis=1)
        frame["levels_asof_timestamp"] = frame.timestamp
        events = detect_breakouts(bars, history=history)
        frame = pd.concat([frame, breakout_recency(n, events)], axis=1)
        meta["breakout_events"] = len(events)
    else:
        frame["breakout_state"] = "inactive"
    if timeframe == "1Day" and not light:
        gaps, gap_meta = filtered_gaps(bars, current_atr)
        frame = pd.concat([frame, gaps], axis=1)
        meta.update(gap_meta)
    else:
        frame["gap_up_recent"] = np.nan
        frame["gap_down_recent"] = np.nan
    if benchmark is not None and not benchmark.empty:
        rs = relative_strength(bars, complete_bars(benchmark))
        for column in ("relative_return_5", "relative_return_20", "price_benchmark_ratio",
                       "ratio_slope", "ratio_zscore", "benchmark_available"):
            frame[column] = rs[column].to_numpy()
    else:
        for column in ("relative_return_5", "relative_return_20", "price_benchmark_ratio",
                       "ratio_slope", "ratio_zscore"):
            frame[column] = np.nan
        frame["benchmark_available"] = False
    for source, label in ((quant, "quant"), (external, "external")):
        if source is not None:
            if len(source) != n:
                raise ValueError(f"{label} context must align with bars ({len(source)} != {n})")
            frame = pd.concat([frame, source.reset_index(drop=True)], axis=1)
            meta[f"{label}_attrs"] = dict(source.attrs)
    return EvidenceBundle(timeframe, frame, events, meta)


def attach_timeframe(frame: pd.DataFrame, other: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Join another timeframe's descriptive state: latest bar with close_time <= this close."""
    prefix = TIMEFRAME_PREFIX[timeframe]
    right = other[["close_time"] + MTF_COLUMNS].rename(
        columns={c: f"{prefix}_{c}" for c in MTF_COLUMNS})
    right = right.rename(columns={"close_time": f"{prefix}_close_time"})
    left = frame.reset_index().rename(columns={"index": "_pos"})
    merged = pd.merge_asof(left.sort_values("close_time"), right.sort_values(f"{prefix}_close_time"),
                           left_on="close_time", right_on=f"{prefix}_close_time",
                           direction="backward", allow_exact_matches=True)
    return merged.sort_values("_pos").drop(columns="_pos").reset_index(drop=True)
