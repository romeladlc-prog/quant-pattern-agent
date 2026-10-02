"""Causal per-bar Quant Model Layer context for the Pattern Engine.

Models are refit on an expanding window at fixed bar positions (walk-forward)
and their *filtered* outputs are used between refits:

* refit positions depend only on how many valid rows exist up to each bar;
* a model refit at bar r is trained on rows <= r and filters rows r..r_next-1;
* lagged values (``*_lag``) come from the same parameter set as the current
  value, so a refit never creates an artificial jump between them;
* PELT change points run on a trailing window ending at the bar, so a change
  point can only be reported ``cp_min_size`` bars or more after it happens.

Only components marked validated in ``model_status`` are used: HMM2 and
Markov2 for regimes, Kalman local level for the level, GARCH for volatility.
Three-state models are never used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from src.features.statistical_features import calculate_statistical_features
from src.models.complexity_models import rolling_complexity
from src.models.model_status import model_status, validated_models
from src.models.regime_models import REGIME_INPUTS, detect_change_points, fit_hmm_regimes, \
    fit_markov_regimes
from src.models.state_space_models import fit_state_specification, filter_state_specification

QUANT_COLUMNS = ["hmm_p_high", "hmm_p_high_lag", "markov_p_high", "markov_p_high_lag",
                 "kalman_level", "kalman_level_change", "kalman_level_change_lag",
                 "garch_vol", "garch_vol_ratio", "cp_rv20_recent", "cp_slope_recent",
                 "cp_rv20_age", "cp_slope_age", "hurst", "permutation_entropy", "quant_fit_index"]
MODELS_USED = ("hmm2", "markov2", "kalman_local_level", "garch", "pelt", "hurst",
               "permutation_entropy")


@dataclass(frozen=True)
class QuantContextConfig:
    min_train: int = 252
    refit_every: int = 63
    lag_bars: int = 5
    cp_window: int = 120
    cp_min_size: int = 10
    cp_penalty: float = 3.0
    cp_recent_bars: int = 15
    complexity_window: int = 100
    regimes: bool = True
    kalman: bool = True
    garch: bool = True
    change_points: bool = True
    complexity: bool = True


def refit_positions(valid: np.ndarray, min_train: int, refit_every: int) -> list[int]:
    count = np.cumsum(valid)
    eligible = np.flatnonzero(valid & (count >= min_train))
    return list(range(int(eligible[0]), len(valid), refit_every)) if eligible.size else []


def _high_state(stats: pd.DataFrame) -> int:
    return int(stats["volatility"].idxmax())


def _fit_garch(train: pd.Series):
    """GARCH(1,1) on percent returns, Normal or Student-t by train AIC."""
    best = None
    for dist in ("normal", "t"):
        try:
            result = arch_model(train*100, mean="Constant", vol="GARCH", p=1, q=1, dist=dist,
                                rescale=False).fit(disp="off", show_warning=False)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if result.convergence_flag == 0 and np.isfinite(result.aic) and \
                (best is None or result.aic < best[0]):
            best = (result.aic, dist, result.params)
    return best


def garch_filter(returns_pct: np.ndarray, params: pd.Series, backcast: float) -> np.ndarray:
    """Causal GARCH(1,1) recursion: sigma2[t] uses residuals up to t-1 only.

    Written out instead of ``arch``'s fixed-parameter filter, whose variance
    bounds are computed from the whole supplied sample (a small look-ahead).
    """
    mu, omega, alpha, beta = (float(params[k]) for k in ("mu", "omega", "alpha[1]", "beta[1]"))
    resid = returns_pct - mu
    sigma2 = np.empty(len(resid))
    previous_var, previous_resid2 = backcast, backcast
    for t in range(len(resid)):
        sigma2[t] = omega + alpha*previous_resid2 + beta*previous_var
        previous_var, previous_resid2 = sigma2[t], resid[t]**2
    return np.sqrt(sigma2)


def _backcast(train_pct: np.ndarray, mu: float) -> float:
    """arch's GARCH backcast, from the first (training) observations only."""
    resid2 = (train_pct - mu)**2
    tau = min(75, len(resid2))
    w = 0.94**np.arange(tau)
    return float(np.sum(w*resid2[:tau])/np.sum(w))


def _change_points(series: pd.Series, i: int, cfg: QuantContextConfig) -> tuple[bool | None, float]:
    window = series.iloc[i-cfg.cp_window+1:i+1]
    if len(window) < cfg.cp_window or window.isna().any():
        return None, np.nan
    points = detect_change_points(pd.Series(window.to_numpy(), index=window.index),
                                  penalty=cfg.cp_penalty, min_size=cfg.cp_min_size)
    if not points:
        return False, np.nan
    age = i - int(max(points))
    return age <= cfg.cp_recent_bars, float(age)


def build_quant_context(bars: pd.DataFrame, config: QuantContextConfig | None = None) -> pd.DataFrame:
    """Per-bar quant context aligned to ``bars`` positions (RangeIndex)."""
    cfg = config or QuantContextConfig()
    for name in ("hmm2", "markov2"):
        if name not in validated_models("regime_description"):
            raise RuntimeError(f"{name} is not validated for regime_description")
    if "kalman_local_level" not in validated_models("level") or \
            "garch" not in validated_models("volatility_forecast"):
        raise RuntimeError("Kalman local level / GARCH not validated")
    n = len(bars)
    out = pd.DataFrame(np.nan, index=pd.RangeIndex(n), columns=QUANT_COLUMNS)
    out[["cp_rv20_recent", "cp_slope_recent"]] = out[["cp_rv20_recent", "cp_slope_recent"]].astype(object)
    out.attrs["models_used"] = {name: model_status(name).as_dict() for name in MODELS_USED}
    out.attrs["warnings"] = []
    if n == 0:
        return out
    features = calculate_statistical_features(bars)
    feats = features.reset_index(drop=True)
    stamps = features.index
    close = bars.close.astype(float).reset_index(drop=True)
    valid = feats[REGIME_INPUTS].notna().all(axis=1).to_numpy()
    refits = refit_positions(valid, cfg.min_train, cfg.refit_every)
    out.attrs["refit_positions"] = refits
    lag = cfg.lag_bars
    garch_values = pd.Series(np.nan, index=out.index)
    for k, r in enumerate(refits):
        end = min(r + cfg.refit_every, n) - 1
        seg = np.arange(r, end+1)
        out.loc[seg, "quant_fit_index"] = r
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if cfg.regimes:
                train, scope = features.iloc[:r+1], features.iloc[:end+1]
                for key, fitter in (("hmm", fit_hmm_regimes), ("markov", fit_markov_regimes)):
                    try:
                        fit = fitter(train, scope, n_states=2)
                    except Exception as error:  # a failed fit leaves NaN, never a fake value
                        out.attrs["warnings"].append(f"{key}@{r}: {type(error).__name__}")
                        continue
                    probability = fit["probabilities"][f"state_{_high_state(fit['stats'])}"]
                    by_position = probability.reindex(stamps[:end+1]).to_numpy()
                    out.loc[seg, f"{key}_p_high"] = by_position[seg]
                    lagged = seg - lag
                    out.loc[seg, f"{key}_p_high_lag"] = np.where(
                        lagged >= 0, by_position[np.clip(lagged, 0, None)], np.nan)
            if cfg.kalman:
                try:
                    fit = fit_state_specification(close.iloc[:r+1], "local level")
                    level = filter_state_specification(fit, close.iloc[:end+1])["level"].to_numpy()
                    change = level - np.r_[np.full(lag, np.nan), level[:-lag]]
                    out.loc[seg, "kalman_level"] = level[seg]
                    out.loc[seg, "kalman_level_change"] = change[seg]
                    lagged = seg - lag
                    out.loc[seg, "kalman_level_change_lag"] = np.where(
                        lagged >= 0, change[np.clip(lagged, 0, None)], np.nan)
                except Exception as error:
                    out.attrs["warnings"].append(f"kalman@{r}: {type(error).__name__}")
            if cfg.garch:
                returns = feats.log_return
                best = _fit_garch(returns.iloc[1:r+1].dropna())
                if best is None:
                    out.attrs["warnings"].append(f"garch@{r}: no converged fit")
                else:
                    _, dist, params = best
                    pct = returns.iloc[1:end+1].to_numpy()*100
                    backcast = _backcast(returns.iloc[1:r+1].to_numpy()*100, float(params["mu"]))
                    vol = np.r_[np.nan, garch_filter(pct, params, backcast)/100]
                    garch_values.iloc[seg] = vol[seg]
        out.attrs["warnings"].extend(f"{type(w.message).__name__}@{r}" for w in caught
                                     if not issubclass(w.category, DeprecationWarning))
    out["garch_vol"] = garch_values
    out["garch_vol_ratio"] = garch_values/garch_values.shift(1).rolling(60, min_periods=20).median()
    if cfg.change_points:
        for column, target in (("realized_volatility_20", "rv20"), ("rolling_slope_20", "slope")):
            series = feats[column]
            for i in range(cfg.cp_window-1, n):
                recent, age = _change_points(series, i, cfg)
                out.at[i, f"cp_{target}_recent"] = recent
                out.at[i, f"cp_{target}_age"] = age
    if cfg.complexity:
        complexity = rolling_complexity(feats.log_return, window=cfg.complexity_window)
        out["hurst"] = complexity.hurst.to_numpy()
        out["permutation_entropy"] = complexity.permutation_entropy.to_numpy()
    return out
