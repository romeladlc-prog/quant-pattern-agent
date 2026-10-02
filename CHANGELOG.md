# Changelog

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
