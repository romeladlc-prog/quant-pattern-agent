"""Phase 4.1 cross-asset, chronological model stability audit."""
from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning

from src.data.alpaca_client import download_historical_bars
from src.features import calculate_statistical_features
from src.models.time_series_models import fit_autoreg, fit_arima_benchmark
from src.models.volatility_models import fit_arch_families, fit_har_rv
from src.models.state_space_models import fit_state_specification, filter_state_specification
from src.models.regime_models import (fit_hmm_regimes, fit_markov_regimes,
                                      detect_change_points)
from src.validation.walk_forward import (WalkForwardConfig, walk_forward_splits,
                                         score_rows, subperiod_scores)

ASSETS = ("ARM", "NVDA", "AMD", "AVGO", "QQQ")


def record(asset, model, index, train_slice, position, forecast, actual, **extra):
    return {"asset": asset, "model": model, "train_start": index[train_slice.start],
            "train_end": index[train_slice.stop - 1], "test_date": index[position],
            "forecast": float(forecast), "actual": float(actual),
            "error": float(forecast - actual), **extra}


def returns_for_fold(asset, returns, train_slice, position, config):
    train = returns.iloc[train_slice]
    actual = returns.iloc[position]
    rows = [record(asset, "zero", returns.index, train_slice, position, 0, actual),
            record(asset, "last", returns.index, train_slice, position,
                   train.iloc[-1], actual)]
    warnings_list = []
    try:
        fit = fit_autoreg(train, max_lags=10, criterion="bic")
        lag = fit["lags"]
        history = list(train.iloc[-lag:].to_numpy())
        for _ in range(config.horizon):
            value = float(np.asarray(fit["result"].params)[0] +
                          np.dot(np.asarray(fit["result"].params)[1:], history[-lag:][::-1]))
            history.append(value)
        rows.append(record(asset, "AutoReg", returns.index, train_slice, position,
                           value, actual, parameter=lag))
    except Exception as exc:
        warnings_list.append(f"AutoReg {asset} {returns.index[position]}: {exc}")
    try:
        # The full 0..5 grid remains available in the model layer. This audit
        # uses a bounded 0..2 grid at each fold to keep repeated refits practical.
        fit = fit_arima_benchmark(train, p_values=range(3), q_values=range(3),
                                  maxiter=60)
        value = float(np.asarray(fit["result"].forecast(steps=config.horizon))[-1])
        rows.append(record(asset, "ARIMA", returns.index, train_slice, position,
                           value, actual, parameter=str(fit["order"])))
        if fit["failed"]:
            warnings_list.append(f"ARIMA {asset} {returns.index[position]}: {fit['failed']} failed grids")
    except Exception as exc:
        warnings_list.append(f"ARIMA {asset} {returns.index[position]}: {exc}")
    return rows, warnings_list


def volatility_for_fold(asset, returns, rv, train_slice, position):
    train = returns.iloc[train_slice]
    actual = rv.iloc[position]
    rows, params, issues = [], [], []
    try:
        fits = fit_arch_families(train)
        for name, fit in fits.items():
            result = fit["result"]
            # One-bar variance from the training endpoint; RV20(t) contains
            # 19 already known squared returns plus the unknown next return.
            next_var = float(result.forecast(horizon=1).variance.iloc[-1, 0]) / 10000
            known = float(returns.iloc[position - 19:position].pow(2).sum())
            forecast = np.sqrt(max(known + next_var, 0))
            rows.append(record(asset, name, returns.index, train_slice, position,
                               forecast, actual))
            for key, value in result.params.items():
                params.append({"asset": asset, "model": name, "test_date": returns.index[position],
                               "parameter": key, "value": float(value),
                               "converged": result.convergence_flag == 0})
        for missing in set(("GARCH", "EGARCH", "GJR-GARCH")) - fits.keys():
            issues.append(f"{asset} {returns.index[position]}: {missing} did not converge")
    except Exception as exc:
        issues.append(f"ARCH {asset} {returns.index[position]}: {exc}")
    try:
        fit = fit_har_rv(rv.iloc[train_slice])
        known = rv.iloc[:position]
        inputs = [1, known.iloc[-1], known.iloc[-5:].mean(), known.iloc[-22:].mean()]
        forecast = max(float(np.dot(inputs, fit["result"].params)), 1e-8)
        rows.append(record(asset, "HAR-RV", returns.index, train_slice, position,
                           forecast, actual))
        for key, value in fit["result"].params.items():
            params.append({"asset": asset, "model": "HAR-RV", "test_date": returns.index[position],
                           "parameter": key, "value": float(value), "converged": True})
    except Exception as exc:
        issues.append(f"HAR-RV {asset} {returns.index[position]}: {exc}")
    return rows, params, issues


def kalman_diagnostics(asset, close, config):
    records = []
    for train_slice, position in walk_forward_splits(close.index, config):
        train = close.iloc[train_slice]
        for spec in ("local level", "local linear trend"):
            try:
                fit = fit_state_specification(train, spec)
                observed = close.iloc[:position + 1]
                filtered = filter_state_specification(fit, observed)
                row = filtered.iloc[position]
                records.append({"asset": asset, "specification": spec,
                                "test_date": close.index[position],
                                "converged": fit["converged"], "boundary": fit["boundary"],
                                "innovation": row.innovation, "slope": row.slope,
                                "slope_change": row.slope_change,
                                "params": fit["params"].to_dict()})
            except Exception as exc:
                warnings.warn(f"Kalman {asset} {spec}: {exc}")
    return records


def changepoint_sensitivity(asset, signals):
    rows = []
    for name, signal in signals.items():
        detections = []
        for penalty in (3.0, 5.0, 8.0):
            for min_size in (15, 25, 40):
                dates = detect_change_points(signal, penalty=penalty, min_size=min_size)
                detections.append(dates)
                for stamp in dates:
                    rows.append({"asset": asset, "signal": name, "date": stamp,
                                 "penalty": penalty, "min_size": min_size})
        print(f"CP {asset}/{name}: counts={[len(x) for x in detections]}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Cluster dates within five trading bars; robust means at least 5/9 settings.
    clusters = []
    for (_, _), group in frame.groupby(["asset", "signal"]):
        ordered = group.sort_values("date")
        current = []
        for row in ordered.itertuples():
            if current and (row.date - current[-1].date).days > 8:
                clusters.append(current)
                current = []
            current.append(row)
        if current:
            clusters.append(current)
    labels = {}
    for cluster in clusters:
        settings = {(x.penalty, x.min_size) for x in cluster}
        label = "robust" if len(settings) >= 5 else "sensitive"
        for x in cluster:
            labels[(x.asset, x.signal, x.date, x.penalty, x.min_size)] = label
    frame["stability"] = [labels[tuple(row)] for row in frame.itertuples(index=False, name=None)]
    return frame


def regime_diagnostics(asset, features, config):
    train_end = max(config.min_train, len(features) - config.step)
    train = features.iloc[:train_end]
    records = []
    for states in (2, 3):
        for name, function in (("Markov", fit_markov_regimes), ("HMM", fit_hmm_regimes)):
            try:
                fit = function(train, features, n_states=states)
                stable = bool(fit.get("converged", True))
                if name == "HMM":
                    stable &= min(fit["assignment_agreement"]) >= 0.8
                print(f"{asset} {name}{states}: converged={fit.get('converged')} "
                      f"current={fit['states'].iloc[-1]} "
                      f"p={fit['probabilities'].iloc[-1].round(3).to_dict()}")
                print(fit["stats"].round(4).to_string())
                if name == "HMM":
                    print(f"  init loglik={fit['initialization_scores']}; "
                          f"aligned agreement={fit['assignment_agreement']}; "
                          f"label swaps={fit['label_permutations']}")
                records.append({"asset": asset, "model": f"{name}{states}",
                                "converged": fit.get("converged", True), "stable": stable})
            except Exception as exc:
                warnings.warn(f"{asset} {name}{states}: {exc}")
                records.append({"asset": asset, "model": f"{name}{states}",
                                "converged": False, "stable": False})
    return records


def main():
    warnings.filterwarnings("ignore", message="A date index has been provided", category=ValueWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--window", type=int)
    args = parser.parse_args()
    config = WalkForwardConfig(args.min_train, args.step, args.horizon, args.window)
    if config.horizon != 1:
        warnings.warn("Volatility and state audits require horizon=1; returns use requested horizon")
    data = {}
    for asset in ASSETS:
        try:
            bars = download_historical_bars(asset, "2023-09-14", date.today().isoformat(),
                                            timeframe="1Day", session="regular")
            data[asset] = bars.set_index("timestamp")
            print(f"{asset}: {len(bars)} bars {bars.timestamp.iloc[0]} .. {bars.timestamp.iloc[-1]}")
        except Exception as exc:
            warnings.warn(f"Download {asset}: {exc}")
    if len(data) != len(ASSETS):
        raise RuntimeError("All five assets are required for the cross-asset audit")
    common = sorted(set.intersection(*(set(frame.index) for frame in data.values())))
    if len(common) < config.min_train + config.horizon + 2:
        raise RuntimeError("Insufficient common history")
    print(f"Common 1Day history: {len(common)} dates, {common[0]} .. {common[-1]}; {config}")
    return_rows, vol_rows, parameter_rows, kalman_rows, cp_rows, regime_rows, issues = [], [], [], [], [], [], []
    for asset, frame in data.items():
        bars = frame.loc[common].reset_index()
        features = calculate_statistical_features(bars)
        returns = features.log_return.dropna()
        rv = features.realized_volatility_20.reindex(returns.index)
        close = frame.loc[returns.index, "close"]
        for train_slice, position in walk_forward_splits(returns.index, config):
            rows, errors = returns_for_fold(asset, returns, train_slice, position, config)
            return_rows.extend(rows)
            issues.extend(errors)
            if config.horizon == 1:
                rows, params, errors = volatility_for_fold(asset, returns, rv, train_slice, position)
                vol_rows.extend(rows)
                parameter_rows.extend(params)
                issues.extend(errors)
        kalman_rows.extend(kalman_diagnostics(asset, close, config))
        try:
            fit = fit_state_specification(close.iloc[:max(config.min_train, len(close)-config.step)],
                                          "local linear trend")
            slope = filter_state_specification(fit, close).slope
            cp_rows.append(changepoint_sensitivity(asset, {"log_return": returns,
                "realized_volatility_20": rv, "kalman_slope": slope}))
        except Exception as exc:
            issues.append(f"CP {asset}: {exc}")
        regime_rows.extend(regime_diagnostics(asset, features, config))
    root = Path("phase4_1_results")
    root.mkdir(exist_ok=True)
    ret = pd.DataFrame(return_rows)
    vol = pd.DataFrame(vol_rows)
    params = pd.DataFrame(parameter_rows)
    kalman = pd.DataFrame(kalman_rows)
    cp = pd.concat(cp_rows, ignore_index=True) if cp_rows else pd.DataFrame()
    regimes = pd.DataFrame(regime_rows)
    for name, table in (("return_folds", ret), ("volatility_folds", vol),
                        ("volatility_parameters", params), ("kalman", kalman),
                        ("change_points", cp), ("regimes", regimes)):
        table.to_csv(root / f"{name}.csv", index=False)
    (root / "warnings.txt").write_text("\n".join(issues), encoding="utf-8")
    ret_score, vol_score = score_rows(ret, "return"), score_rows(vol, "volatility")
    print("\nRETURN PER ASSET\n", ret_score.to_string(index=False))
    print("\nRETURN SUBPERIODS\n", subperiod_scores(ret, "return").to_string(index=False))
    print("\nVOLATILITY PER ASSET\n", vol_score.to_string(index=False))
    print("\nVOLATILITY SUBPERIODS\n", subperiod_scores(vol, "volatility").to_string(index=False))
    for title, scores, baseline in (("return", ret_score, "zero"),
                                    ("volatility", vol_score, "HAR-RV")):
        if scores.empty:
            continue
        pivot = scores.pivot(index="asset", columns="model", values="rmse")
        print(f"\n{title.upper()} CROSS ASSET mean/median/std and wins vs {baseline}")
        for model in pivot.columns:
            values = pivot[model].dropna()
            wins = int((pivot[model] < pivot[baseline]).sum()) if baseline in pivot else 0
            print(f"{model}: mean={values.mean():.6f} median={values.median():.6f} "
                  f"std={values.std():.6f} wins={wins}/{len(pivot)}")
    print("Technical stability scores and benchmark_only flags: "
          "phase4_1_results/stability_scores.csv (run summarize_phase4_1.py).")
    if not kalman.empty:
        print("\nKALMAN")
        for spec, group in kalman.groupby("specification"):
            print(f"{spec}: convergence={group.converged.mean():.2f} boundary={group.boundary.mean():.2f} "
                  f"innovation_RMSE={np.sqrt(np.mean(group.innovation**2)):.4f} "
                  f"slope_std={group.slope.std():.4f} slope_change_std={group.slope_change.std():.4f}")
    if not cp.empty:
        print("\nCHANGE POINTS", cp.groupby(["signal", "stability"]).size().to_dict())
    print("\nREGIME STABILITY\n", regimes.groupby("model")[["converged", "stable"]].mean().to_string())
    print(f"Warnings: {len(issues)}; see {root / 'warnings.txt'}")


if __name__ == "__main__":
    main()
