"""Hand-built evidence rows for deterministic detector scenarios.

Keys left out of a row are "not evaluable" (None), exactly as when an input
family is missing in the real evidence frame.
"""
import pandas as pd

from src.patterns.engine import bar_contexts

BASE = dict(open=100., high=101., low=99., close=100., volume=1000., atr=1.0, zscore_20=0.0,
            slope_change=0.0, rolling_slope_20=0.0, slope=0.0, accel_up_recent=False,
            accel_down_recent=False, structural_state="range", relative_volume=1.0,
            atr_expansion=1.0, volatility_expansion=False, volatility_compression=False,
            compression_recent=False, breakout_state="inactive", breakout_direction=None,
            exhaustion_candidate=False, acceleration_state="neutral", new_swing_high=False,
            new_swing_low=False)


def make_rows(n, overrides=None, **constant):
    stamps = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for i in range(n):
        row = {**BASE, **constant, "timestamp": stamps[i], "close_time": stamps[i], "events_here": []}
        row.update((overrides or {}).get(i, {}))
        rows.append(row)
    return rows


def span(rows_overrides, start, stop, **values):
    for i in range(start, stop):
        rows_overrides.setdefault(i, {}).update(values)
    return rows_overrides


def run(detector_cls, direction, rows):
    contexts = [bar_contexts(row, []) for row in rows]
    return detector_cls(direction, "TEST", "1Day").run(rows, contexts)


def states(results):
    return [r.state for r in results]


def event(rows, i, state, direction, level=100.0, touches=2, failure_bars=None, asof_offset=1):
    rows[i]["events_here"].append({
        "timestamp": rows[i]["timestamp"], "bar_index": i, "level": level, "level_type": "resistance",
        "level_touch_count": touches, "level_asof_timestamp": rows[i-asof_offset]["timestamp"],
        "direction": direction, "state": state, "distance_atr": 0.6, "relative_volume": 1.4,
        "failure_bars": failure_bars})
