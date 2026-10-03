# Fase 4.1 — Estabilidad y validación robusta

**Estado:** completada con Python 3.12 y `hmmlearn`. `src/validation/walk_forward.py` proporciona cortes cronológicos configurables: entrenamiento mínimo, ventana expansiva o rolling, paso y horizonte. `validate_walk_forward.py` registra por fold inicio y fin de train, fecha test, forecast, valor real, error y modelo. El horizonte mayor que uno se aplica a retornos; las comparaciones de volatilidad usan uno.

El audit empleó 756 fechas comunes de ARM, NVDA, AMD, AVGO y QQQ, ocho fechas de prueba por activo, sin shuffle. Registró 160 filas de retorno y 159 de volatilidad por un fallo GJR-GARCH en AVGO. `summarize_phase4_1.py` produce métricas por activo y score técnico local. AutoReg y ARIMA mejoraron al retorno cero en 2/5 activos; GARCH tuvo el menor RMSE promedio de volatilidad. Kalman local level resultó más estable; HMM2 y Markov2 convergieron en cinco activos, mientras los modelos de tres estados no fueron uniformemente estables.

**Regímenes: `legacy_pre_revalidation`.** "HMM2 y Markov2 convergieron en cinco activos" usó el flag permisivo de hmmlearn y un Markov sin RNG reproducible (statsmodels ≥ 0.15), sobre un solo corte por activo. Está pendiente de [revalidación](PHASE_4_1_REVALIDATION.md). Los modelos de 3 estados siguen experimentales.

**Límites:** ocho fechas por activo no prueban estabilidad futura. RV20 de días contiguos se solapa; el grid ARIMA repetido es p,q=0..2 por costo. El audit de regímenes usa el último corte por activo, no un walk-forward completo. El score no es una señal ni una probabilidad de acierto. Véase [MODEL_DECISIONS.md](../MODEL_DECISIONS.md).
