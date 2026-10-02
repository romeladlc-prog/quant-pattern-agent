"""Deterministic synthetic OHLCV shared by the offline tests."""
import numpy as np
import pandas as pd


def bars_from_close(values, volumes=None, opens=None, start="2025-01-01", freq="D"):
    close = np.asarray(values, dtype=float)
    op = np.asarray(opens if opens is not None else close, dtype=float)
    volume = np.asarray(volumes if volumes is not None else np.full(len(close), 1000.), dtype=float)
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=len(close), freq=freq, tz="UTC"),
        "open": op, "high": np.maximum(op, close)+.2, "low": np.minimum(op, close)-.2,
        "close": close, "volume": volume})


def random_walk(n=160, seed=3):
    rng = np.random.default_rng(seed)
    return bars_from_close(100+np.cumsum(rng.normal(0, 1, n)),
                           volumes=rng.uniform(500, 1500, n))


def anchored(anchors: dict[int, float]):
    """Piecewise-linear closes through {bar: close} anchors: strict pivots at anchors."""
    points = sorted(anchors)
    return bars_from_close(np.interp(np.arange(points[-1]+1), points, [anchors[p] for p in points]))
