# Changelog

## 2026-10-03 — Fase 4.1-R: revalidación de regímenes preparada (sin resultados reales)

- `validate_regime_reproducibility.py` y `src/validation/regime_revalidation.py`: HMM2/Markov2 en 8 folds walk-forward más el corte original de 4.1, con convergencia estricta (`converged`/`not_converged`/`fit_failed`), reproducibilidad entre repeticiones con tolerancias documentadas, sensibilidad a semilla y acuerdo entre folds. Salidas en `phase4_1_revalidation/`. `phase4_1_results/` no se modifica.
- Resultados de regímenes de 4.1 marcados `legacy_pre_revalidation` en la documentación y en el texto de `summarize_phase4_1.py`. La decisión sobre HMM2/Markov2 no cambia hasta tener datos reales.
- `tests/test_regime_revalidation.py` (8 tests) y paso sintético en CI.

## 2026-10-03 — Fase 7.1 (cont.): `macro_status` causal por barra

- Causa 3, hallada con datos reales: sin `FRED_API_KEY` las filas del CSV de FRED tienen `available_at` = hora de descarga. Truncadas a T desaparecían (`missing`), pero la ejecución completa marcaba todas las barras como `excluded_not_asof_safe`.
- `build_external_context` decide `macro_status` y `news_quality` por barra con filas `available_at <= cierre`. Estados: `missing` (no suministrado), `not_yet_available`, `excluded_not_asof_safe`, `asof_safe`.
- `--debug-prefix` muestra la primera fila externa que solo ve la ejecución completa. `--synthetic` añade macro tipo FRED CSV en SYN_A.
- `tests/test_macro_context_causality.py`: 10 tests (108 en total).

## 2026-10-02 — Fase 7.1: causa raíz del fallo de prefijo con datos reales

- Causa 1: un timeframe de contexto sin barras cerradas en T (4Hour/1Hour, descargados solo 400/90 días) se omitía en la ejecución hasta T y se adjuntaba en la completa, así que `mtf_context` cambiaba de claves en todas las barras pasadas. Ahora siempre se adjunta, con valores ausentes (`attach_empty_timeframe`).
- Causa 2, oculta tras la 1: HMM2 no era reproducible bit a bit (KMeans de sklearn con OpenMP multihilo dentro de hmmlearn). `fit_hmm_regimes` ajusta con un hilo OpenMP.
- `Model is not converging` (hmmlearn, EM con log-verosimilitud decreciente) no causa el fallo de prefijo, pero ya no pasa sin bandera: `fit_quality`/`seed_quality` en `fit_hmm_regimes`, columnas `*_fit_converged` por barra en el contexto cuantitativo, claves en `regime_context`/`volatility_context` y nota `model_not_converged:<modelos>`.
- `validate_patterns.py --debug-prefix`: ante un fallo de prefijo informa del primer timestamp, patrón, campo e input divergente y su módulo, y relanza el mismo error.
- `--synthetic` incluye contexto 4Hour/1Hour tardío (último 20% de SYN_A), que reproduce el fallo real con el código anterior.
- 6 tests nuevos (98 en total). Detalle e impacto en [docs/phases/PHASE_7_1_PREFIX_FIX.md](docs/phases/PHASE_7_1_PREFIX_FIX.md).

## 2026-10-02 — Fase 6.1: corrección de dtype en `detect_breakouts`

- `failure_bars` y `relative_volume` salen siempre como `float64` (ausente = NaN). Antes, un frame sin ningún fallo dejaba `failure_bars` como `object`/`None` y uno con fallos como `float64`/`NaN`, por lo que la comparación por prefijo de `validate_structure.py` fallaba (`None != nan`) aunque los eventos fueran idénticos. Sin cambios en la lógica de detección.
- 4 tests nuevos (92 en total), incluido el caso `random_walk()` cortado en 30 barras.

## 2026-10-02 — Fase 7: Pattern Engine

- `src/patterns/`: frame de evidencia causal por barra, contexto cuantitativo walk-forward (HMM2, Markov2, Kalman local level, GARCH, PELT en ventana móvil, Hurst/entropía) y contexto externo as-of.
- Detectores: agotamiento alcista/bajista, breakout (con subtipos), failed breakout, mean reversion, continuación de tendencia, transición de régimen y divergencias, con máquina de estados determinista (`inactive`→`candidate`→`developing`→`confirmed`/`invalidated`/`expired`).
- `PatternResult` con `convergence_score` en [0,1] por componentes, evidencia a favor/en contra y contextos; tabla de episodios `phase7_results/pattern_events.csv` y métricas descriptivas.
- `validate_patterns.py` (Alpaca o `--synthetic`) con comprobación de leakage por prefijo.
- Corrección: `fit_markov_regimes` pasa `rng` a statsmodels ≥ 0.15 (antes no era determinista). El contexto GARCH usa una recursión causal explícita en lugar del filtro fijo de `arch`, cuyos límites de varianza usan toda la muestra.
- 44 tests nuevos de Fase 7 (88 en total) y paso de validación sintética en CI.

## 2026-10-02 — Fase 6.1: hardening causal y sincronización documental

- Niveles históricos as-of (`levels_history`, `build_levels_asof`); `detect_breakouts` compara cada cruce con los niveles conocidos al cierre de la barra anterior. `detect_breakouts_fixed_levels` para niveles externos.
- `breakout_state` / `fake_breakout_state` con recencia `breakout_max_age_bars` (5) y `latest_*_event` con `age_bars`.
- Fuerza relativa sin relleno implícito (`fill_method=None`), columna `benchmark_available`.
- 1Week con `is_complete`.
- Feed único `ALPACA_DATA_FEED` (IEX por defecto); `validate_structure.py` usa la Data Layer y termina antes del día actual.
- `src/models/model_status.py`: benchmark_only, roles, estabilidad y modelos de 3 estados experimentales; salidas de ajuste con `model_status`.
- Redacción de credenciales en errores de FRED y Alpaca.
- 44 tests pytest sin red, `requirements-dev.txt` y workflow de GitHub Actions con Python 3.12.
- Documentación sincronizada hasta Fase 6.1.

## 2026-09-20 — Fase 6: Market Structure Engine

- Detectores causales de pivots, estructura, niveles, rupturas/fallos, compresión/expansión, aceleración, gaps y fuerza relativa.
- Snapshot descriptivo por activo y timeframe; validación de ARM, NVDA, AMD, AVGO y QQQ en 4Hour, 1Day y 1Week, más ARM 1Hour.
- Tests sintéticos y log/resumen de validación. Sin señales ni recomendaciones.

## 2026-09-20 — Documentación hasta Fase 5

- Documentadas arquitectura, decisiones de modelos, fuentes, validaciones y roadmap.
- Fase 5: External Context Engine independiente para VIX, liquidez macro y del activo, amplitud y noticias, con `available_at` y validación de cinco activos.
- Fase 4.1: Python 3.12, walk-forward cross-asset, diagnósticos de estabilidad y benchmarks documentados.

## 2026-09-19 — Fases 1 a 4

- Fase 1: Data Layer Alpaca RAW y NORMALIZED con validación temporal.
- Fase 2: visor Streamlit de barras y sesiones.
- Fase 3: motor de 15 features estadísticas causales.
- Fase 4: familias cuantitativas de retorno, volatilidad, Kalman, complejidad, change points y regímenes, sin señales.
