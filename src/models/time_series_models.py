"""Return forecasts fitted only on a chronological training sample."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


def fit_autoreg(train: pd.Series, max_lags: int = 10, criterion: str = "aic") -> dict:
    """Select lag count 1..max_lags with a common hold-back sample."""
    if criterion not in {"aic", "bic"} or max_lags < 1 or len(train) < max(40, 4 * max_lags):
        raise ValueError("AutoReg: criterio, rezagos o muestra inválidos.")
    candidates = []
    for lag in range(1, max_lags + 1):
        try:
            result = AutoReg(train.reset_index(drop=True), lags=lag,
                             hold_back=max_lags).fit()
            candidates.append((getattr(result, criterion), lag, result))
        except (ValueError, np.linalg.LinAlgError) as error:
            warnings.warn(f"AutoReg({lag}) omitido: {error}", stacklevel=2)
    if not candidates:
        raise RuntimeError("Ningún AutoReg pudo ajustarse.")
    _, lag, result = min(candidates, key=lambda item: item[0])
    return {"result": result, "lags": lag, "aic": result.aic, "bic": result.bic,
            "residuals": result.resid}


def autoreg_one_step(fit: dict, observed: pd.Series) -> pd.Series:
    """Fixed train coefficients; each prediction uses only preceding returns."""
    lag = fit["lags"]
    params = np.asarray(fit["result"].params)
    values = observed.to_numpy(dtype=float)
    forecast = np.full(len(values), np.nan)
    for t in range(lag, len(values)):
        forecast[t] = params[0] + np.dot(params[1:], values[t-lag:t][::-1])
    return pd.Series(forecast, index=observed.index, name="AutoReg")


def fit_arima_benchmark(train: pd.Series, p_values=range(6), d_values=range(2),
                        q_values=range(6), maxiter: int = 100) -> dict:
    """Controlled ARIMA grid, selected by train AIC; unsuitable fits are skipped."""
    if len(train) < 60:
        raise ValueError("ARIMA requiere al menos 60 observaciones.")
    candidates = []
    failed = 0
    for d in d_values:
        for p in p_values:
            for q in q_values:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = ARIMA(train.reset_index(drop=True), order=(p, d, q),
                                       enforce_stationarity=True,
                                       enforce_invertibility=True).fit(
                                           method_kwargs={"maxiter": maxiter})
                    if not result.mle_retvals.get("converged", False) or not np.isfinite(result.aic):
                        failed += 1
                        continue
                    candidates.append((result.aic, (p, d, q), result))
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    failed += 1
    if not candidates:
        raise RuntimeError("Ningún ARIMA convergió.")
    _, order, result = min(candidates, key=lambda item: item[0])
    if failed:
        warnings.warn(f"ARIMA: {failed} combinaciones fallaron o no convergieron.", stacklevel=2)
    return {"result": result, "order": order, "aic": result.aic, "bic": result.bic,
            "residuals": result.resid, "failed": failed}


def arima_one_step(fit: dict, test: pd.Series) -> pd.Series:
    """Causal one-step forecasts; Kalman state updates, parameters stay fixed."""
    result = fit["result"]
    predictions = []
    for timestamp, actual in test.items():
        predictions.append(float(np.asarray(result.forecast(steps=1))[0]))
        result = result.append([float(actual)], refit=False)
    return pd.Series(predictions, index=test.index, name="ARIMA")


def naive_one_step(observed: pd.Series, mode: str = "zero") -> pd.Series:
    """Zero return or last observed return benchmark."""
    if mode == "zero":
        return pd.Series(0.0, index=observed.index, name="naive_zero")
    if mode == "last":
        return observed.shift(1).rename("naive_last")
    raise ValueError("mode debe ser 'zero' o 'last'.")
