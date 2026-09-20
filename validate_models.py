"""Chronological out-of-sample comparison of complementary ARM daily models."""
from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.data.alpaca_client import download_historical_bars
from src.features import calculate_statistical_features
from src.models.complexity_models import rolling_complexity
from src.models.regime_models import (detect_change_points, fit_hmm_regimes,
                                      fit_markov_regimes, regime_changes)
from src.models.state_space_models import fit_local_linear_trend, filter_local_linear_trend
from src.models.time_series_models import (fit_autoreg, autoreg_one_step,
                                            fit_arima_benchmark, arima_one_step,
                                            naive_one_step)
from src.models.volatility_models import (fit_arch_families, arch_one_step,
                                           fit_har_rv, har_one_step, volatility_metrics)


def return_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    joined = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    error = joined["predicted"] - joined["actual"]
    correlation = joined.corr().iloc[0, 1] if joined["predicted"].std() > 0 else np.nan
    return {"n": len(joined), "mae": error.abs().mean(),
            "rmse": np.sqrt(error.pow(2).mean()),
            "directional_accuracy": (np.sign(joined["predicted"]) ==
                                     np.sign(joined["actual"])).mean(),
            "correlation": correlation}


def main() -> None:
    # Free SIP access excludes the latest 15 minutes; stop before today.
    end = date.today().isoformat()
    bars = download_historical_bars("ARM", "2023-09-14", end,
                                    timeframe="1Day", session="regular")
    features = calculate_statistical_features(bars).dropna(subset=["log_return"])
    close = bars.set_index("timestamp")["close"].astype(float)
    returns = features["log_return"]
    if len(returns) < 300:
        warnings.warn("Muestra pequeña para modelos de régimen y GARCH.", stacklevel=1)
    split = int(len(returns) * 0.75)
    train, test = returns.iloc[:split], returns.iloc[split:]
    print(f"ARM 1Day: {len(bars)} barras, {bars['timestamp'].iloc[0]} a "
          f"{bars['timestamp'].iloc[-1]}")
    print(f"Train {len(train)} / test {len(test)}, corte {test.index[0]}; sin shuffle")

    auto = fit_autoreg(train, max_lags=10, criterion="bic")
    arima = fit_arima_benchmark(train)
    predictions = {
        "AutoReg": autoreg_one_step(auto, returns).loc[test.index],
        "ARIMA benchmark": arima_one_step(arima, test),
        "naive zero": naive_one_step(returns, "zero").loc[test.index],
        "naive last": naive_one_step(returns, "last").loc[test.index],
    }
    print(f"\nRetorno: AutoReg lags={auto['lags']} AIC={auto['aic']:.2f} "
          f"BIC={auto['bic']:.2f}; ARIMA{arima['order']} AIC={arima['aic']:.2f} "
          f"({arima['failed']} grids fallidos)")
    print(pd.DataFrame({name: return_metrics(test, predicted)
                        for name, predicted in predictions.items()}).T.to_string())

    rv = features["realized_volatility_20"]
    arch_fits = fit_arch_families(train)
    har = fit_har_rv(rv.loc[:train.index[-1]])
    vol_predictions = {name: arch_one_step(fit, returns, split).loc[test.index]
                       for name, fit in arch_fits.items()}
    vol_predictions["HAR-RV"] = har_one_step(har, rv).loc[test.index]
    vol_metrics = pd.DataFrame({name: volatility_metrics(rv.loc[test.index], predicted)
                                for name, predicted in vol_predictions.items()}).T
    print("\nVolatilidad versus RV20 futura (horizonte 1 barra):")
    print(vol_metrics.to_string())
    for name, fit in arch_fits.items():
        print(f"{name}: dist={fit['distribution']}, AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}")

    kalman = fit_local_linear_trend(close.loc[:train.index[-1]])
    filtered = filter_local_linear_trend(kalman, close)
    kalman_test = filtered.loc[test.index]
    print("\nKalman (filtrado, parámetros train):")
    print(kalman_test[["level", "slope", "slope_change", "innovation",
                       "level_uncertainty", "slope_uncertainty"]].tail(3).to_string())
    print(f"slope mediana={kalman_test['slope'].median():.4f}; "
          f"level uncertainty mediana={kalman_test['level_uncertainty'].median():.4f}; "
          f"innovation RMSE={np.sqrt(kalman_test['innovation'].pow(2).mean()):.4f}")

    complexity = rolling_complexity(returns, window=100)
    print("\nComplejidad (features, última observación):")
    print(complexity.tail(1).to_string())

    print("\nRegímenes; probabilidades filtradas, parámetros train:")
    train_features = features.loc[:train.index[-1]]
    for n_states in (2, 3):
        for label, fit_function in (("Markov", fit_markov_regimes), ("HMM", fit_hmm_regimes)):
            try:
                fitted = fit_function(train_features, features, n_states=n_states)
                current = fitted["states"].iloc[-1]
                print(f"{label} {n_states}: estado actual={current}, "
                      f"probabilidades={fitted['probabilities'].iloc[-1].round(3).to_dict()}, "
                      f"cambios={len(regime_changes(fitted['states']))}")
                print(fitted["stats"].round(4).to_string())
                print(f"últimos cambios: {list(regime_changes(fitted['states'])[-5:])}")
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
                warnings.warn(f"{label} {n_states} no disponible: {error}", stacklevel=1)

    print("\nChange points PELT retrospectivos (no usados en forecasts):")
    signals = {"log_return": returns, "realized_volatility_20": rv,
               "kalman_slope": filtered["slope"]}
    for name, signal in signals.items():
        dates = detect_change_points(signal, penalty=5.0, min_size=20)
        print(f"{name}: {len(dates)} cambios; últimos={dates[-5:]}")

    print("\nNotas: RV20 se solapa entre días; sus errores OOS favorecen persistencia. "
          "QLIKE compara varianzas y puede ser negativo. Estados son índices arbitrarios. "
          "Change points retrospectivos no son señales en tiempo real.")


if __name__ == "__main__":
    main()
