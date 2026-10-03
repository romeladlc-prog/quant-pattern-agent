# Fase 4.1: revalidación de regímenes (HMM2 / Markov2)

**Estado:** herramienta lista y probada sin red. **Todavía no hay resultados con datos reales.** Las conclusiones de regímenes de Fase 4.1 están marcadas como `legacy_pre_revalidation` hasta ejecutar este audit con Alpaca en local.

## 1. Cómo se validaron los regímenes en la Fase 4.1 original

Fuente: `validate_walk_forward.py` (`regime_diagnostics`), `summarize_phase4_1.py` y `phase4_1_results/regimes.csv` (local, no versionado).

| Aspecto | Fase 4.1 original |
|---|---|
| Activos | ARM, NVDA, AMD, AVGO, QQQ, 1Day `regular`, fechas comunes desde 2023-09-14 (756 fechas). |
| Folds | **Ninguno para regímenes.** Un solo corte por activo: entrenamiento con las primeras `max(252, n − 63)` filas (693) y filtrado de toda la muestra. Los 8 folds de 4.1 solo se usaron para forecasts y Kalman. |
| Modelos | HMM2, HMM3, Markov2, Markov3 (`fit_hmm_regimes`, `fit_markov_regimes`). |
| HMM | Tres semillas (42, 43, 44), se elige la de mayor log-verosimilitud de entrenamiento. Probabilidades con filtro forward (no suavizado). |
| Markov | `MarkovRegression` con varianza conmutada, `search_reps=5`, parámetros congelados y filtrado. |
| Criterio de convergencia HMM | `monitor_.converged` de hmmlearn. Es **permisivo**: también vale True si el EM agota `n_iter` o si la log-verosimilitud **baja** (`Model is not converging`). |
| Criterio de convergencia Markov | `mle_retvals["converged"]` de statsmodels. |
| "Estable" | HMM: convergido **y** acuerdo mínimo entre semillas ≥ 0.8. Markov: igual a convergido. |
| Métricas | `converged` y `stable` por activo y modelo. `stability_score = (convergence + stable)/2` en `summarize_phase4_1.py`. Probabilidades y estadísticas de estado solo en `phase4_1_validation.log`. |
| Conclusiones escritas | "HMM2 y Markov2 convergieron en los cinco activos" (score 1.0); "HMM3 tuvo acuerdo entre inicializaciones irregular; Markov3 convergió en 2/5 activos". HMM2/Markov2 quedaron `stable_for_current_use=true` para `regime_description`; HMM3/Markov3 `experimental`. |

## 2. Resultados que quedan como `legacy_pre_revalidation`

No se borra nada. `phase4_1_results/` no se toca: el script nuevo escribe en otra carpeta y copia la tabla original como `legacy_pre_revalidation_regimes.csv` junto al refit.

| Resultado original | Por qué ya no es definitivo |
|---|---|
| `regimes.csv`: `converged`/`stable` de **Markov2 y Markov3** (los 5 activos) | Con statsmodels ≥ 0.15 el ajuste ignoraba la semilla global: cada ejecución usaba arranques distintos. Ahora `rng=random_state`. |
| `regimes.csv`: `converged` de **HMM2 y HMM3** | Usaba el flag permisivo de hmmlearn. Algunos "convergidos" pueden ser EM con log-verosimilitud decreciente. |
| `regimes.csv`: `stable` de HMM (acuerdo entre semillas) | Parámetros reproducibles solo hasta los últimos bits (KMeans/OpenMP). El efecto en el acuerdo debería ser nulo salvo empates, pero no está demostrado. |
| `stability_scores.csv`: filas HMM2/HMM3/Markov2/Markov3 | Se derivan de las dos anteriores. |
| `summary.md` de 4.1: "HMM2 y Markov2 convergieron en los cinco activos" | Texto fijo en `summarize_phase4_1.py`, no calculado. Ahora el texto remite a esta revalidación. |
| `MODEL_DECISIONS.md` y `PHASE_4_1_STABILITY.md`: evidencia de HMM2/Markov2/HMM3/Markov3 | Marcada `legacy_pre_revalidation`. **La decisión no cambia** hasta tener datos reales. |
| `validate_models.py` (Fase 4): convergencia de HMM/Markov sobre ARM | Mismas razones, aunque es un solo ajuste descriptivo. |

No cambian por esto: AutoReg, ARIMA, GARCH/EGARCH/GJR/HAR-RV (forecasts), Kalman y PELT.

## 3. Revalidación

`validate_regime_reproducibility.py` y `src/validation/regime_revalidation.py`. No modifican ningún modelo, parámetro ni regla de selección.

**Cortes por activo** (mismo universo de fechas comunes que 4.1):
- 8 folds walk-forward expansivos (`min_train=252`, `step=63`, como 4.1). Cada fold entrena con las filas `[0, train_end]` y filtra **solo hasta su fecha de test**: ninguna fila posterior entra en el ajuste ni en el filtro.
- El corte único original (`legacy_last_cut`), para comparar directamente con `regimes.csv`.

**Por corte y modelo** (HMM2 y Markov2; `--states 2 3` añade los experimentales solo para inspección):
- 3 ajustes idénticos (misma entrada, configuración y semilla 42) para medir la **reproducibilidad**.
- 1 ajuste con semilla 43 para medir la **sensibilidad a la inicialización**.

**Métricas** (`regime_revalidation.csv`, una fila por ajuste):
- `status` ∈ {`converged`, `not_converged`, `fit_failed`};
- `monitor_converged`, `iterations`, `hit_max_iter` y `loglik_decreases` (HMM);
- `seeds_not_converged` (de las 3 semillas internas de HMM);
- `train_loglik`, AIC/BIC (Markov);
- `final_regime` (rango de volatilidad del estado argmax en la fecha de test), `p_high_final` y `p_state_final`;
- `obs_per_regime` en entrenamiento;
- `init_loglik_spread` e `init_min_agreement` (HMM);
- `error` si el ajuste falla (mensaje redactado).

**Convergencia HMM.** `converged` exige `monitor_.converged` **y** ninguna bajada de log-verosimilitud del EM **y** no agotar `n_iter` (`fit_quality` de `fit_hmm_regimes`). Si no se cumple, es `not_converged`. Una excepción, o una salida cuyos estados no se pueden describir, es `fit_failed`. Un `not_converged` aislado **no** descarta el modelo: se cuentan las ventanas, el porcentaje, los activos y los folds afectados (`convergence_summary.csv`).

**Reproducibilidad.** HMM se ajusta con un hilo OpenMP; Markov recibe `rng=random_state`. Una ventana es reproducible si sus 3 repeticiones tienen el mismo `status` y el mismo `final_regime`, y no superan estas tolerancias (`TOLERANCES`):

| Magnitud | Tolerancia | Motivo |
|---|---|---|
| Probabilidades filtradas | 1e-9 absoluta | En una misma máquina la igualdad es exacta. La tolerancia cubre diferencias de BLAS/CPU entre máquinas, muy por debajo de lo que podría cambiar un estado. |
| Parámetros | 1e-7 absoluta | Escala estandarizada (HMM) o porcentual (Markov). |
| Log-verosimilitud | 1e-9 relativa | |

La comprobación por prefijo del Pattern Engine (Fase 7) sí exige igualdad exacta: ahí la referencia y el prefijo corren en la misma máquina.

**Estabilidad entre folds.** Para cada par de folds consecutivos se mide el acuerdo del régimen de alta volatilidad (`p_high ≥ 0.5`) y la diferencia media de `p_high` en las fechas que ambos filtraron (`regime_fold_stability.csv`). Las etiquetas de estado se alinean por volatilidad, no por índice.

**Convergencia, reproducibilidad y estabilidad son distintas.** Un ajuste puede converger y no ser estable entre folds, o ser reproducible sin converger. Ninguna de las tres implica utilidad predictiva: `converged=True` solo dice que el optimizador terminó limpio.

## 4. Salidas (`phase4_1_revalidation/`, no versionado)

| Archivo | Contenido |
|---|---|
| `regime_revalidation.csv` | Una fila por ajuste (activo × modelo × corte × repetición/semilla). |
| `regime_reproducibility.csv` | Una fila por ventana: diferencias máximas entre repeticiones y `reproducible`. |
| `regime_seed_sensitivity.csv` | Semilla 42 frente a 43: acuerdo de régimen y \|Δp_high\| medio. |
| `regime_fold_stability.csv` | Acuerdo entre folds consecutivos. |
| `convergence_summary.csv` | Ventanas por estado, porcentajes, activos y folds afectados. |
| `legacy_pre_revalidation_regimes.csv` | `phase4_1_results/regimes.csv` original junto al refit estricto del mismo corte (solo si existe localmente). |
| `summary.md` | Resumen legible. No cambia la decisión de modelos por sí mismo. |
| `run_metadata.json` | Fecha, commit, fuente y feed, fechas por activo, configuración, tolerancias y versiones de paquetes. Sin secretos. |

`--synthetic` escribe en `phase4_1_revalidation/synthetic/` con series sintéticas; esos números no describen mercados.

## 5. Ejecución local (Windows, Python 3.12)

Desde `C:\Users\romel\Projects\quant-pattern-agent`, con `.env` (las credenciales no se imprimen; los errores pasan por `redact_secrets`):

```powershell
git fetch origin; git checkout claude/phase4-1-regime-revalidation; git pull; .\.venv312\Scripts\python.exe -m pip install -r requirements.txt; .\.venv312\Scripts\python.exe -u validate_regime_reproducibility.py *> phase4_1_revalidation.log
```

Opciones: `--repeats`, `--states 2 3`, `--min-train`, `--step`, `--assets`, `--csv-dir` (CSV locales en lugar de descarga) y `--out`.

## 6. Riesgos metodológicos que quedan

- 8 folds por activo siguen siendo pocos para hablar de estabilidad futura. El acuerdo entre folds compara filtros con parámetros distintos sobre fechas pasadas; no mide persistencia hacia delante.
- Las tolerancias son entre repeticiones en la misma máquina. Comparar con otra máquina puede dar diferencias mayores sin que eso implique un error.
- HMM elige entre 3 semillas por log-verosimilitud. La sensibilidad se mide solo con una semilla alternativa (43).
- Los datos IEX (feed por defecto) cambian volumen y algunas barras respecto a SIP. La revalidación usa el feed configurado y lo registra.
- El universo de fechas comunes depende de ARM (desde 2023-09-14), como en 4.1.
- `arch_one_step` (GARCH de Fase 4) conserva su fuga mínima de varianza. No afecta a regímenes.
