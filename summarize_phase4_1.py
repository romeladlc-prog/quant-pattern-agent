"""Build a reproducible technical stability summary from fold-level CSV files."""
from __future__ import annotations

from pathlib import Path
import ast
import numpy as np
import pandas as pd

from src.validation.walk_forward import score_rows

ROOT = Path("phase4_1_results")


def bounded(value):
    return float(np.clip(value, 0, 1)) if np.isfinite(value) else 0.0


def forecast_scores(rows, baseline, parameters=None):
    metrics = score_rows(rows, "return" if baseline == "zero" else "volatility")
    pivot = metrics.pivot(index="asset", columns="model", values="rmse")
    counts = rows.groupby("model").size()
    expected = rows[["asset", "test_date"]].drop_duplicates().shape[0]
    output = []
    for model in pivot.columns:
        values = pivot[model]
        base = pivot[baseline]
        convergence = bounded(counts[model] / expected)
        if model == baseline:
            parameter_stability = 1.0
            fold_consistency = asset_consistency = advantage = 0.5
        else:
            if parameters is not None and model in set(parameters.model):
                part = parameters[parameters.model == model]
                cv = part.groupby(["asset", "parameter"]).value.apply(
                    lambda s: s.std() / max(abs(s.mean()), 1e-8)).replace([np.inf, -np.inf], np.nan)
                parameter_stability = bounded(1 / (1 + cv.median()))
            elif model == "AutoReg":
                lag = pd.to_numeric(rows.loc[rows.model == model, "parameter"], errors="coerce")
                parameter_stability = bounded(1 / (1 + lag.std() / max(lag.mean(), 1)))
            elif model == "ARIMA":
                orders = rows.loc[rows.model == model, "parameter"]
                parameter_stability = bounded(orders.value_counts(normalize=True).iloc[0])
            else:
                parameter_stability = 1.0
            matched = rows[rows.model.isin([model, baseline])].pivot_table(
                index=["asset", "test_date"], columns="model", values="error")
            matched = matched.dropna(subset=[model, baseline])
            fold_consistency = bounded((matched[model].abs() < matched[baseline].abs()).mean())
            asset_consistency = bounded((values < base).mean())
            advantage = bounded((base.mean() - values.mean()) / max(base.mean(), 1e-12) * 10)
        score = np.mean([convergence, parameter_stability, fold_consistency,
                         asset_consistency, advantage])
        output.append({"component": model, "stability_score": round(float(score), 3),
                       "convergence": round(convergence, 3),
                       "parameter_stability": round(parameter_stability, 3),
                       "fold_consistency": round(fold_consistency, 3),
                       "asset_consistency": round(asset_consistency, 3),
                       "advantage": round(advantage, 3),
                       "benchmark_only": baseline == "zero" and model != "zero" and asset_consistency < 0.6})
    return pd.DataFrame(output)


def diagnostic_scores(kalman, cp, regimes):
    rows = []
    for spec, frame in kalman.groupby("specification"):
        convergence = frame.converged.mean()
        boundary_free = 1 - frame.boundary.mean()
        innovation = np.sqrt(np.mean(frame.innovation**2))
        other = kalman[kalman.specification != spec]
        other_innovation = np.sqrt(np.mean(other.innovation**2))
        residual_quality = bounded(other_innovation / max(innovation, 1e-8))
        score = np.mean([convergence, boundary_free, residual_quality])
        rows.append({"component": f"Kalman {spec}", "stability_score": round(score, 3),
                     "convergence": round(convergence, 3), "boundary_free": round(boundary_free, 3),
                     "residual_quality": round(residual_quality, 3)})
    for model, frame in regimes.groupby("model"):
        convergence = frame.converged.mean()
        stable = frame.stable.mean()
        rows.append({"component": model, "stability_score": round((convergence + stable) / 2, 3),
                     "convergence": round(convergence, 3), "initialization_or_fit_stability": round(stable, 3)})
    for signal, frame in cp.groupby("signal"):
        robust = frame.stability.eq("robust").mean()
        rows.append({"component": f"PELT {signal}", "stability_score": round(robust, 3),
                     "parameter_grid_agreement": round(robust, 3)})
    return pd.DataFrame(rows)


def main():
    ret = pd.read_csv(ROOT / "return_folds.csv", parse_dates=["test_date"])
    vol = pd.read_csv(ROOT / "volatility_folds.csv", parse_dates=["test_date"])
    params = pd.read_csv(ROOT / "volatility_parameters.csv")
    kalman = pd.read_csv(ROOT / "kalman.csv")
    cp = pd.read_csv(ROOT / "change_points.csv")
    regimes = pd.read_csv(ROOT / "regimes.csv")
    state_parameters = pd.DataFrame([
        {"asset": row.asset, "specification": row.specification,
         **ast.literal_eval(row.params)} for row in kalman.itertuples()])
    state_cv = (state_parameters.melt(id_vars=["asset", "specification"],
                                      var_name="parameter", value_name="value")
                .groupby(["asset", "specification", "parameter"]).value
                .agg(lambda s: s.std() / max(abs(s.mean()), 1e-8)).rename("cv"))
    vol_cv = (params.groupby(["asset", "model", "parameter"]).value
              .agg(lambda s: s.std() / max(abs(s.mean()), 1e-8)).rename("cv"))
    scores = pd.concat([forecast_scores(ret, "zero"),
                        forecast_scores(vol, "HAR-RV", params),
                        diagnostic_scores(kalman, cp, regimes)], ignore_index=True)
    scores.to_csv(ROOT / "stability_scores.csv", index=False)
    return_metrics = score_rows(ret, "return")
    vol_metrics = score_rows(vol, "volatility")
    winner = vol_metrics.groupby("model").rmse.mean().idxmin()
    lines = ["# Fase 4.1: auditoría walk-forward", "",
             "40 folds programados por familia de forecast: 8 fechas por cada uno de 5 activos; "
             "ventana expansiva de 252 barras y paso de 63; horizonte 1.", "",
             "Las métricas detalladas por activo y fold están en los CSV de esta carpeta. "
             "El score mide estabilidad técnica (0..1), no utilidad de trading.", "",
             "## Resultados por activo", "",
             "### Retorno", "", "```text", return_metrics.round(5).to_string(index=False), "```", "",
             "### Volatilidad", "", "```text", vol_metrics.round(5).to_string(index=False), "```", "",
             "### Kalman por activo", "", "```text",
             kalman.groupby(["asset", "specification"]).agg(
                 converged=("converged", "mean"), boundary=("boundary", "mean"),
                 innovation_rmse=("innovation", lambda x: np.sqrt(np.mean(x**2))),
                 slope_median=("slope", "median"),
                 slope_std=("slope", "std"),
                 slope_change_std=("slope_change", "std")
             ).round(4).to_string(), "```", "",
             "### Variación de parámetros entre folds (CV)", "", "```text",
             "Kalman\n" + state_cv.round(3).to_string() + "\n\nVolatilidad\n" + vol_cv.round(3).to_string(),
             "```", "",
             "### Regímenes por activo", "", "```text",
             regimes.to_string(index=False), "```", "",
             "### Change points por activo", "", "```text",
             cp.groupby(["asset", "signal", "stability"]).size().to_string(), "```", "",
             "## Stability score", "", "```text", scores.fillna("").to_string(index=False), "```", "",
             f"Menor RMSE promedio de volatilidad: **{winner}**.", "",
             "AutoReg y ARIMA quedan como benchmark_only: superaron el retorno cero "
             "en menos de 3 de 5 activos por RMSE. HAR-RV quedó por detrás de las "
             "familias ARCH en esta muestra. GARCH y GJR-GARCH son candidatos a "
             "redundancia, pero GJR falló en un fold de AVGO.", "",
             "Local level tuvo menos parámetros en el borde y menor RMSE de innovación; "
             "es la especificación Kalman más estable aquí. No estima slope; "
             "local linear trend conserva utilidad descriptiva del slope, con advertencia de borde.", "",
             "HMM2 y Markov2 convergieron en los cinco activos. HMM3 y Markov3 "
             "no alcanzaron estabilidad uniforme. Las etiquetas de estados son arbitrarias.", "",
             "Los change points son retrospectivos; robusto significa que apareció "
             "en al menos 5 de 9 combinaciones de penalización y segmento dentro de "
             "una agrupación de hasta ocho días calendario. La fracción de filas "
             "robustas no estima precisión predictiva.", "",
             "Los parámetros Kalman estimados de cada fold están en kalman.csv; "
             "los parámetros ARCH/HAR y sus variaciones están en volatility_parameters.csv. "
             "Las probabilidades y estadísticas descriptivas de cada estado están en "
             "phase4_1_validation.log.", "",
             "Los scores HMM y Markov miden el último corte de entrenamiento por activo, "
             "sin una prueba walk-forward de persistencia de regímenes; un 1.0 en ellos "
             "no demuestra estabilidad futura. Kalman local level no produce slope.", "",
             "Límites: ocho fechas por activo dan evidencia de estabilidad todavía "
             "escasa. RV20 se solapa entre fechas; tres estados pueden ocupar muy "
             "pocas barras. El grid ARIMA del audit usa p,q=0..2 por costo de reentrenamiento."]
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
