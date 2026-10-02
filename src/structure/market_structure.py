"""Composable descriptive snapshot; each timeframe remains independent."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import pandas as pd

from .acceleration import acceleration_features
from .breakouts import detect_breakouts
from .compression import compression_features
from .gaps import detect_gaps
from .levels import cluster_levels
from .relative_strength import relative_strength
from .swings import confirmed_swings
from .trend_structure import classify_swings, structural_state


@dataclass
class MarketStructureSnapshot:
    ticker: str
    timestamp: pd.Timestamp
    timeframe: str
    structural_state: str
    last_swing_high: dict | None
    last_swing_low: dict | None
    recent_structure: list[str]
    nearest_support: dict | None
    nearest_resistance: dict | None
    breakout_state: str
    fake_breakout_state: str
    compression_state: str | None
    expansion_state: str | None
    compression_duration: int
    acceleration_state: str | None
    exhaustion_candidate: bool
    latest_gap: dict | None
    relative_strength_metrics: dict
    latest_breakout_event: dict | None = None
    latest_fake_breakout_event: dict | None = None
    breakout_max_age_bars: int = 5

    def as_dict(self) -> dict:
        return asdict(self)


def _last(frame: pd.DataFrame) -> dict | None:
    if frame.empty: return None
    result = frame.iloc[-1].to_dict()
    return {key: (None if pd.isna(value) else value) for key,value in result.items()}


def _recent(events: pd.DataFrame, last_index: int, max_age_bars: int) -> tuple[str, dict | None]:
    """State of the latest event if at most max_age_bars old, else 'inactive'.

    Age is last_index - bar_index of the bar where the event was observed, so
    an event on the snapshot bar has age 0.
    """
    latest = _last(events)
    if latest is None: return "inactive", None
    latest["age_bars"] = last_index-int(latest["bar_index"])
    return (latest["state"] if latest["age_bars"] <= max_age_bars else "inactive"), latest


def make_structure_snapshot(ticker: str, bars: pd.DataFrame, timeframe: str = "1Day",
                            benchmark: pd.DataFrame | None = None,
                            left_bars: int = 3, right_bars: int = 2,
                            breakout_max_age_bars: int = 5) -> MarketStructureSnapshot:
    if timeframe not in {"1Hour","4Hour","1Day","1Week"}:
        raise ValueError("unsupported timeframe")
    if bars.empty or bars.timestamp.duplicated().any() or not bars.timestamp.is_monotonic_increasing:
        raise ValueError("bars must be nonempty, unique and sorted")
    if timeframe in {"1Hour","4Hour"} and "is_complete" in bars and not bars.is_complete.all():
        raise ValueError("partial intraday blocks cannot enter structure analysis")
    if breakout_max_age_bars < 0:
        raise ValueError("breakout_max_age_bars must be nonnegative")
    swings = classify_swings(confirmed_swings(bars,left_bars,right_bars))
    levels = cluster_levels(bars,swings)  # as of the last bar: known at its close
    events = detect_breakouts(bars,swings)
    compression = compression_features(bars).iloc[-1]
    acceleration = acceleration_features(bars).iloc[-1]
    gaps = detect_gaps(bars) if timeframe == "1Day" else pd.DataFrame()
    rs = relative_strength(bars,benchmark).iloc[-1] if benchmark is not None else None
    price = float(bars.close.iloc[-1])
    supports = levels.loc[levels.price_level.lt(price) & levels.type.isin(["support","mixed"])]
    resistances = levels.loc[levels.price_level.gt(price) & levels.type.isin(["resistance","mixed"])]
    breakout = events.loc[events.state.isin(["breakout_candidate","breakout_confirmed"])]
    fake = events.loc[events.state.str.startswith("fake_breakout")]
    last_index = len(bars)-1
    breakout_state, latest_breakout = _recent(breakout, last_index, breakout_max_age_bars)
    fake_state, latest_fake = _recent(fake, last_index, breakout_max_age_bars)
    rs_columns = ["relative_return_5","relative_return_20","relative_return_60",
                  "price_benchmark_ratio","ratio_slope","ratio_zscore","benchmark"]
    return MarketStructureSnapshot(ticker.upper(),bars.timestamp.iloc[-1],timeframe,
        structural_state(swings),
        _last(swings.loc[swings.type.eq("swing_high")]),
        _last(swings.loc[swings.type.eq("swing_low")]),
        swings.structure_label.dropna().tail(4).tolist(),
        _last(supports.sort_values("price_level")),
        _last(resistances.sort_values("price_level",ascending=False)),
        breakout_state, fake_state,
        "volatility_compression" if compression.volatility_compression else None,
        "volatility_expansion" if compression.volatility_expansion else None,
        int(compression.compression_duration),
        acceleration.acceleration_state,
        bool(acceleration.exhaustion_candidate), _last(gaps),
        {key: None if pd.isna(rs[key]) else rs[key] for key in rs_columns} if rs is not None else {},
        latest_breakout, latest_fake, breakout_max_age_bars)
