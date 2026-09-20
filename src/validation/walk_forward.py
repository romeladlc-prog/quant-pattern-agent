"""Reusable, timestamp-preserving walk-forward splits and score summaries."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train: int = 252
    step: int = 21
    horizon: int = 1
    window: int | None = None  # None means expanding; otherwise rolling.

    def __post_init__(self):
        if self.min_train < 60 or self.step < 1 or self.horizon < 1:
            raise ValueError("min_train>=60, step>=1 and horizon>=1 are required")
        if self.window is not None and self.window < self.min_train:
            raise ValueError("rolling window must be at least min_train")


def walk_forward_splits(index: pd.DatetimeIndex, config: WalkForwardConfig):
    """Yield train positions and one test position; train ends before test."""
    if not isinstance(index, pd.DatetimeIndex) or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("index must be a sorted unique DatetimeIndex")
    for test_pos in range(config.min_train + config.horizon - 1, len(index), config.step):
        train_end = test_pos - config.horizon + 1  # exclusive
        train_start = max(0, train_end - config.window) if config.window else 0
        yield slice(train_start, train_end), test_pos


def score_rows(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Scores from recorded OOS forecasts; zero forecasts have no direction."""
    output = []
    for (asset, model), frame in rows.dropna(subset=["forecast", "actual"]).groupby(["asset", "model"]):
        error = frame.forecast - frame.actual
        record = {"asset": asset, "model": model, "n": len(frame),
                  "mae": error.abs().mean(), "rmse": np.sqrt(np.mean(error**2)),
                  "error_std": error.std(ddof=1)}
        if kind == "return":
            nonzero = frame.forecast.ne(0)
            record["directional_accuracy"] = (np.sign(frame.loc[nonzero, "forecast"]) ==
                                                np.sign(frame.loc[nonzero, "actual"])).mean() if nonzero.any() else np.nan
            record["correlation"] = frame.forecast.corr(frame.actual) if frame.forecast.nunique() > 1 else np.nan
        else:
            predicted_var = frame.forecast.clip(lower=1e-8) ** 2
            record["qlike"] = np.mean(np.log(predicted_var) + frame.actual**2 / predicted_var)
        output.append(record)
    return pd.DataFrame(output)


def subperiod_scores(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Split recorded test dates into early and late halves per asset."""
    pieces = []
    for asset, frame in rows.groupby("asset"):
        dates = sorted(frame.test_date.unique())
        midpoint = dates[len(dates) // 2]
        for label, part in (("early", frame[frame.test_date < midpoint]),
                            ("late", frame[frame.test_date >= midpoint])):
            if not part.empty:
                scored = score_rows(part, kind)
                scored["subperiod"] = label
                pieces.append(scored)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
