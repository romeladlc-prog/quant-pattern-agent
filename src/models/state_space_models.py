"""Causal local linear trend estimated with a Kalman filter."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


def fit_local_linear_trend(train_close: pd.Series) -> dict:
    """Estimate observation, level and slope noise using training prices only."""
    if len(train_close) < 60:
        warnings.warn("Kalman: muestra pequeña para estimar tres varianzas.", stacklevel=2)
    result = UnobservedComponents(train_close, level="local linear trend").fit(disp=False)
    if not result.mle_retvals.get("converged", False):
        warnings.warn("Kalman: optimización no convergió.", stacklevel=2)
    if any(value < 1e-10 for value in result.params):
        warnings.warn("Kalman: alguna varianza estimada está en el límite cero.", stacklevel=2)
    return {"result": result, "params": result.params, "aic": result.aic,
            "bic": result.bic}


def filter_local_linear_trend(fit: dict, observed_close: pd.Series) -> pd.DataFrame:
    """Filter all observations with frozen train parameters, never smooth."""
    model = UnobservedComponents(observed_close, level="local linear trend")
    filtered = model.filter(fit["params"])
    state = filtered.filtered_state
    covariance = filtered.filtered_state_cov
    output = pd.DataFrame({
        "level": state[0], "slope": state[1],
        "level_uncertainty": np.sqrt(np.maximum(covariance[0, 0], 0)),
        "slope_uncertainty": np.sqrt(np.maximum(covariance[1, 1], 0)),
        "innovation": filtered.forecasts_error[0],
        "innovation_uncertainty": np.sqrt(np.maximum(filtered.forecasts_error_cov[0, 0], 0)),
    }, index=observed_close.index)
    output["slope_change"] = output["slope"].diff()
    return output


def fit_state_specification(train_close: pd.Series, specification: str = "local linear trend") -> dict:
    """Fit a state model on train only; expose boundary and convergence diagnostics."""
    if specification not in {"local level", "local linear trend"}:
        raise ValueError("Unsupported state specification")
    result = UnobservedComponents(train_close.reset_index(drop=True), level=specification).fit(disp=False)
    params = result.params
    boundary = bool((params < 1e-8).any())
    return {"result": result, "params": params, "specification": specification,
            "converged": bool(result.mle_retvals.get("converged", False)),
            "boundary": boundary, "aic": result.aic, "bic": result.bic}


def filter_state_specification(fit: dict, observed_close: pd.Series) -> pd.DataFrame:
    """Filtered states and one-step innovations with frozen train parameters."""
    model = UnobservedComponents(observed_close.reset_index(drop=True),
                                 level=fit["specification"])
    result = model.filter(fit["params"])
    state = result.filtered_state
    output = pd.DataFrame({"level": state[0],
                           "innovation": result.forecasts_error[0]}, index=observed_close.index)
    output["slope"] = state[1] if state.shape[0] > 1 else np.nan
    output["slope_change"] = output.slope.diff()
    return output
