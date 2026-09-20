"""Causal statistical features on NORMALIZED OHLCV bars.

The output contains only features and retains the input's UTC timestamp index.
All windows include the current bar and earlier bars, never later bars. Windows
require their full number of observations. Undefined early values remain NaN.

Columns:
  simple_return: close / previous close - 1.
  log_return: log(close / previous close).
  rolling_mean_20: mean of the last 20 closes.
  rolling_std_20: sample standard deviation of the last 20 closes (ddof=1).
  zscore_20: (close - rolling_mean_20) / rolling_std_20.
  momentum_5, momentum_20: close / close 5 or 20 bars ago - 1.
  realized_volatility_20: square root of the sum of 20 squared log returns;
      unannualized, in log-return units.
  rolling_skew_20, rolling_kurtosis_20: pandas rolling sample skew and excess
      kurtosis of 20 log returns.
  drawdown: close / highest close observed so far - 1.
  rolling_max_drawdown_60: minimum drawdown from a running peak within the
      trailing 60 closes; zero if that window has no decline.
  autocorr_lag1, autocorr_lag5: Pearson autocorrelation of log returns within
      the trailing 20 returns, at lag 1 or 5. Constant windows yield NaN.
  rolling_slope_20: least-squares slope of the trailing 20 closes against
      bar positions 0..19, in price units per bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def _autocorr(values: np.ndarray, lag: int) -> float:
    left, right = values[:-lag], values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _max_drawdown(values: np.ndarray) -> float:
    return float(np.min(values / np.maximum.accumulate(values) - 1))


def calculate_statistical_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return causal features indexed by NORMALIZED UTC timestamps.

    Accepts the client's timestamp column or an already indexed DatetimeIndex.
    Close prices must be finite and strictly positive. Missing observations are
    rejected; undefined feature values are left as NaN, including zero-variance
    z-scores and autocorrelations.
    """
    if "close" not in bars:
        raise ValueError("Falta la columna close.")
    if "timestamp" in bars:
        frame = bars.set_index("timestamp", drop=True)
    else:
        frame = bars
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Se requiere un índice temporal o la columna timestamp.")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Los timestamps deben ser únicos, válidos y crecientes.")
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    if not np.isfinite(close.to_numpy()).all() or close.le(0).any():
        raise ValueError("close debe contener precios finitos y positivos.")

    out = pd.DataFrame(index=frame.index)
    out["simple_return"] = close.pct_change(fill_method=None)
    out["log_return"] = np.log(close / close.shift(1))
    out["rolling_mean_20"] = close.rolling(20, min_periods=20).mean()
    out["rolling_std_20"] = close.rolling(20, min_periods=20).std(ddof=1)
    out["zscore_20"] = (close - out["rolling_mean_20"]) / out["rolling_std_20"].replace(0, np.nan)
    out["momentum_5"] = close / close.shift(5) - 1
    out["momentum_20"] = close / close.shift(20) - 1
    returns = out["log_return"]
    out["realized_volatility_20"] = np.sqrt(returns.pow(2).rolling(20, min_periods=20).sum())
    out["rolling_skew_20"] = returns.rolling(20, min_periods=20).skew()
    out["rolling_kurtosis_20"] = returns.rolling(20, min_periods=20).kurt()
    out["drawdown"] = close / close.cummax() - 1
    out["rolling_max_drawdown_60"] = close.rolling(60, min_periods=60).apply(_max_drawdown, raw=True)
    out["autocorr_lag1"] = returns.rolling(20, min_periods=20).apply(
        lambda values: _autocorr(values, 1), raw=True)
    out["autocorr_lag5"] = returns.rolling(20, min_periods=20).apply(
        lambda values: _autocorr(values, 5), raw=True)
    positions = np.arange(20, dtype=float)
    centered = positions - positions.mean()
    out["rolling_slope_20"] = close.rolling(20, min_periods=20).apply(
        lambda values: float(np.dot(centered, values) / np.dot(centered, centered)), raw=True)
    return out


def run_statistical_diagnostics(bars: pd.DataFrame, features: pd.DataFrame,
                                lags: int = 10) -> pd.DataFrame:
    """Return ADF on close and Ljung-Box/Jarque-Bera on log returns.

    Tests are descriptive over the full supplied sample, not causal features.
    """
    if "close" not in bars or "log_return" not in features:
        raise ValueError("Se requieren las columnas close y log_return.")
    close = pd.to_numeric(bars["close"], errors="raise").dropna()
    returns = pd.to_numeric(features["log_return"], errors="raise").dropna()
    if len(close) < 20 or len(returns) <= lags or lags < 1:
        raise ValueError("No hay suficientes observaciones para los diagnósticos.")
    adf = adfuller(close, autolag="AIC", result_object=False)
    lb = acorr_ljungbox(returns, lags=[lags], return_df=True).iloc[0]
    jb = stats.jarque_bera(returns)
    return pd.DataFrame({
        "statistic": [adf[0], lb["lb_stat"], jb.statistic],
        "pvalue": [adf[1], lb["lb_pvalue"], jb.pvalue],
        "nobs": [adf[3], len(returns), len(returns)],
        "lags": [adf[2], lags, np.nan],
    }, index=pd.Index(["ADF (close)", "Ljung-Box (log_return)",
                      "Jarque-Bera (log_return)"], name="test"))
