"""Rolling complexity descriptors; these are features, not decisions."""
from __future__ import annotations

from collections import Counter
from math import factorial

import numpy as np
import pandas as pd
from scipy.stats import entropy


def hurst_exponent(values: np.ndarray) -> float:
    """Estimate H by log-log slope of rescaled ranges over several block sizes."""
    values = np.asarray(values, dtype=float)
    sizes = np.unique(np.geomspace(8, max(9, len(values) // 2), 8).astype(int))
    pairs = []
    for size in sizes:
        ratios = []
        for block in np.array_split(values[:len(values) // size * size], len(values) // size):
            centered = block - block.mean()
            scale = centered.std(ddof=1)
            if scale > 0:
                ratios.append(np.ptp(np.cumsum(centered)) / scale)
        if ratios:
            pairs.append((np.log(size), np.log(np.mean(ratios))))
    return float(np.polyfit(*np.array(pairs).T, deg=1)[0]) if len(pairs) >= 3 else np.nan


def shannon_entropy(values: np.ndarray, bins: int = 10) -> float:
    """Histogram entropy in nats; bin edges depend only on this window."""
    counts = np.histogram(values, bins=bins)[0]
    return float(entropy(counts[counts > 0]))


def permutation_entropy(values: np.ndarray, order: int = 3) -> float:
    """Normalized ordinal-pattern entropy; stable sorting resolves ties."""
    patterns = Counter(tuple(np.argsort(values[i:i + order], kind="stable"))
                       for i in range(len(values) - order + 1))
    counts = np.fromiter(patterns.values(), dtype=float)
    return float(entropy(counts) / np.log(factorial(order)))


def rolling_complexity(log_returns: pd.Series, window: int = 100,
                       bins: int = 10, order: int = 3) -> pd.DataFrame:
    """Trailing-window Hurst, Shannon and permutation entropy with initial NaNs."""
    if window < 32 or bins < 2 or order < 2 or order > window:
        raise ValueError("Parámetros de complejidad inválidos.")
    rolling = log_returns.rolling(window, min_periods=window)
    return pd.DataFrame({
        "hurst": rolling.apply(hurst_exponent, raw=True),
        "shannon_entropy": rolling.apply(lambda x: shannon_entropy(x, bins), raw=True),
        "permutation_entropy": rolling.apply(lambda x: permutation_entropy(x, order), raw=True),
    }, index=log_returns.index)
