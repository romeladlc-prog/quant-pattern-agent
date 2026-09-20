"""ARCH-family and HAR-RV volatility models with causal one-step evaluation."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.api import OLS, add_constant


ARCH_SPECS = {"GARCH": ("GARCH", 0), "EGARCH": ("EGARCH", 0),
              "GJR-GARCH": ("GARCH", 1)}


def fit_arch_families(train_log_return: pd.Series) -> dict:
    """Fit three (1,1) families with Normal and Student-t errors on percent returns."""
    if len(train_log_return) < 150:
        warnings.warn("Menos de 150 retornos: parámetros ARCH pueden ser inestables.", stacklevel=2)
    scaled = train_log_return * 100
    fits = {}
    for name, (vol, asymmetry) in ARCH_SPECS.items():
        candidates = []
        for distribution in ("normal", "t"):
            try:
                model = arch_model(scaled, mean="Constant", vol=vol, p=1, o=asymmetry,
                                   q=1, dist=distribution, rescale=False)
                result = model.fit(disp="off", show_warning=False)
                if result.convergence_flag == 0 and np.isfinite(result.aic):
                    candidates.append((result.aic, distribution, result))
                else:
                    warnings.warn(f"{name}/{distribution} no convergió.", stacklevel=2)
            except (ValueError, np.linalg.LinAlgError) as error:
                warnings.warn(f"{name}/{distribution}: {error}", stacklevel=2)
        if candidates:
            _, distribution, result = min(candidates, key=lambda item: item[0])
            fits[name] = {"result": result, "distribution": distribution,
                          "vol_spec": vol, "asymmetry": asymmetry,
                          "aic": result.aic, "bic": result.bic,
                          "conditional_volatility": result.conditional_volatility / 100,
                          "standardized_residuals": result.std_resid}
        else:
            warnings.warn(f"{name}: ningún ajuste utilizable.", stacklevel=2)
    return fits


def arch_one_step(fit: dict, observed_log_return: pd.Series, train_size: int,
                  horizon_scale: int = 20) -> pd.Series:
    """Fixed train parameters, recursively filtered one-step variance.

    Output is a horizon_scale-bar volatility comparable to the existing
    sqrt(sum of squared log returns) realized-volatility feature.
    """
    result = fit["result"]
    model = arch_model(observed_log_return * 100, mean="Constant",
                       vol=fit["vol_spec"], p=1,
                       o=fit["asymmetry"],
                       q=1, dist=fit["distribution"], rescale=False)
    fixed = model.fix(result.params)
    variance = fixed.forecast(horizon=1, start=train_size - 1,
                              reindex=True).variance.iloc[:, 0].shift(1) / 10000
    # At t-1, the other 19 squared returns in RV20(t) are already observed.
    known = observed_log_return.pow(2).shift(1).rolling(
        horizon_scale - 1, min_periods=horizon_scale - 1).sum()
    return np.sqrt((known + variance).clip(lower=0)).rename("volatility")


def har_predictors(realized_volatility: pd.Series) -> pd.DataFrame:
    """Daily RV20, trailing 5-day mean and trailing 22-day mean, known at t."""
    return pd.DataFrame({"daily": realized_volatility,
                         "weekly": realized_volatility.rolling(5, min_periods=5).mean(),
                         "monthly": realized_volatility.rolling(22, min_periods=22).mean()})


def fit_har_rv(train_rv: pd.Series) -> dict:
    """Fit RV(t+1) to daily, weekly, monthly inputs available at t."""
    predictors = har_predictors(train_rv)
    target = train_rv.shift(-1).rename("target")
    sample = predictors.join(target).dropna()
    if len(sample) < 60:
        raise ValueError("HAR-RV requiere al menos 60 filas válidas.")
    x = add_constant(sample[["daily", "weekly", "monthly"]], has_constant="add")
    result = OLS(sample["target"], x).fit()
    if np.linalg.cond(x.to_numpy()) > 1e8:
        warnings.warn("HAR-RV: predictores casi singulares.", stacklevel=2)
    return {"result": result, "residuals": result.resid,
            "aic": result.aic, "bic": result.bic}


def har_one_step(fit: dict, observed_rv: pd.Series) -> pd.Series:
    """Predict the next RV20 using only the preceding day's known predictors."""
    x = har_predictors(observed_rv).shift(1)
    x = add_constant(x, has_constant="add")
    forecast = x @ fit["result"].params
    return forecast.clip(lower=1e-8).rename("HAR-RV")


def volatility_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """Evaluate positive volatility forecasts and QLIKE on variances."""
    joined = pd.concat([actual.rename("actual"), forecast.rename("forecast")], axis=1).dropna()
    joined = joined.loc[joined["forecast"].gt(0) & joined["actual"].ge(0)]
    if joined.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "qlike": np.nan}
    error = joined["forecast"] - joined["actual"]
    variance_forecast = joined["forecast"].pow(2).clip(lower=1e-12)
    variance_actual = joined["actual"].pow(2)
    qlike = np.log(variance_forecast) + variance_actual / variance_forecast
    return {"n": len(joined), "mae": error.abs().mean(),
            "rmse": np.sqrt(error.pow(2).mean()), "qlike": qlike.mean()}
