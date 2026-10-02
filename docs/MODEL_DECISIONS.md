# Decisiones de modelos y estructura

Decisiones tomadas con el audit de Fase 4.1 sobre 756 fechas comunes de ARM, NVDA, AMD, AVGO y QQQ, usando ocho fechas de prueba por activo. Son decisiones de arquitectura, sujetas a nueva evidencia.

| Componente | Decisión | Evidencia o límite |
|---|---|---|
| AutoReg y ARIMA | `benchmark_only=true`; mantener código. | Cada uno mejoró el RMSE de retorno cero en 2/5 activos; no hubo ventaja estable. ARIMA usó grid reducido p,q=0..2 en walk-forward. |
| GARCH | Modelo principal de volatilidad **por ahora**. | Menor RMSE promedio cross-asset: 0.00490 frente a RV20 siguiente. |
| EGARCH | Mantener como complemento. | RMSE promedio 0.00506; ganó en ARM y QQQ, sin superioridad uniforme. |
| GJR-GARCH | Posible redundancia; no eliminar aún. | Resultados próximos a GARCH en muchos folds, con un fallo de ajuste en AVGO. |
| HAR-RV | Benchmark de volatilidad. | RMSE promedio 0.00806, inferior a las familias ARCH en esta muestra. |
| Kalman local level | Especificación recomendada para nivel. | 3% de ajustes en borde; RMSE de innovación 10.76, menor que local linear trend. No produce slope. |
| Kalman local linear trend | Solo para describir slope, con advertencia. | 57% de ajustes en borde; RMSE de innovación 11.47. |
| HMM2 y Markov2 | Conservar como descripciones de regímenes. | Convergieron en los cinco activos; el audit de regímenes usa el último corte por activo, no una prueba de estabilidad futura. Etiquetas arbitrarias. |
| HMM3 y Markov3 | No considerar estables. | HMM3 tuvo acuerdo entre inicializaciones irregular; Markov3 convergió en 2/5 activos. |
| Hurst, Shannon y permutation entropy | Features descriptivas. | No son motores de decisión ni señales. |
| PELT change points | Descripción retrospectiva. | Fracción robusta ante parámetros: RV20 0.790, slope Kalman 0.814, retornos 0.466. |

Estas decisiones están codificadas en `src/models/model_status.py` (Fase 6.1). Cada ajuste devuelve `model_status`; HMM3 y Markov3 se pueden ejecutar, pero llevan `experimental=true`, `warning="unstable_across_assets"` y emiten `UserWarning`. Una capa futura debe usar `validated_models(purpose)` (solo componentes estables, no experimentales ni benchmark) o `allowed_models(purpose)` si acepta advertencias documentadas:

| Modelo | Metadata |
|---|---|
| AutoReg, ARIMA | `benchmark_only=true`, `role=benchmark_return` |
| GARCH | `role=primary_volatility`, estable para `volatility_forecast` |
| EGARCH | `role=complementary_volatility`, estable para `volatility_forecast` |
| GJR-GARCH | `role=possibly_redundant_volatility`, sin validación de uso |
| HAR-RV | `role=benchmark_volatility`, `benchmark_only=true` |
| Kalman local level | `role=stable_level_state`, validado para `level` |
| Kalman local linear trend | `role=slope_description`, permitido para `slope_description` con `warning=boundary_estimates_57pct_less_stable`; no aparece en `validated_models` |
| HMM2, Markov2 | `stable_for_current_use=true` para `regime_description` |
| HMM3, Markov3 | `experimental=true`, `warning=unstable_across_assets` |
| PELT | Descripción retrospectiva; útil en RV20 y slope Kalman, no en retornos |
| Hurst, entropías | `descriptive_feature` |

La implementación de Kalman de Fase 4 no cambió: `fit_local_linear_trend` sigue disponible y su salida declara su estado.

## Decisiones de Market Structure (Fases 6 y 6.1)

| Componente | Decisión | Evidencia o límite |
|---|---|---|
| Niveles históricos | **As-of obligatorio**. Todo uso histórico (validación, Pattern Engine, backtesting) debe usar `levels_history`, `build_levels_asof` o `detect_breakouts(bars)`; nunca `cluster_levels` de muestra completa para eventos pasados. | El clustering final fusiona, desplaza u omite niveles con swings posteriores; los tests de prefijo demuestran la invariancia de la versión as-of. |
| Breakouts | Cruce de la barra *i* contra niveles as-of de *i−1*. `breakout_state` activo solo durante `breakout_max_age_bars` (5 por defecto). | Antes el snapshot devolvía el último breakout histórico sin límite. |
| Latencia de pivots | Un swing fractal existe desde `confirmed_at` (2 barras tras el extremo, con 3/2 barras); el ATR reversal tiene latencia variable registrada. Las capas futuras deben fechar por `confirmed_at`, no por `timestamp`. | El timestamp del extremo es anterior a su disponibilidad. |
| Gaps | Ruidosos con IEX; **no** pueden ser condición principal del Pattern Engine y su peso futuro debe ser bajo. Requieren calibración con feed consolidado antes de confiar en ellos. No se optimizan todavía. | 548–690 de 756 días diarios marcados como gap por activo en Fase 6 (IEX, umbral 0.2%). |
| Feed | `ALPACA_DATA_FEED` único para módulos y validadores; IEX por defecto, SIP nunca asumido. Feeds distintos solo en comparaciones explícitas (`feed=`). | Fase 6 forzaba IEX mientras Data Layer usaba el feed por defecto de la suscripción. IEX afecta a volumen, gaps, ATR y pivots. |
| Semanas | Usar solo barras 1Week con `is_complete=True` en análisis histórico. | La última semana puede estar abierta cuando `end` cae antes del viernes 16:00 ET. |
| Fuerza relativa | Unión por timestamp exacto sin relleno; filas con `benchmark_available=false` dan NaN. | Evita forward-fill implícito de pandas 2.x. |

## Decisiones del Pattern Engine (Fase 7)

| Componente | Decisión | Motivo o límite |
|---|---|---|
| Score | `convergence_score` en [0,1]: media ponderada de componentes, cada uno `S/(E+A)`. Pesos heurísticos en `src/patterns/config.py`, **no optimizados**. | Mide convergencia de evidencia, no probabilidad, retorno ni señal. Optimizarlo exigiría la Fase 8 con validación fuera de muestra. |
| Evidencia ausente | `None`, excluida del componente; un componente sin evidencia evaluable se excluye del score. | La ausencia de contexto no se convierte en cero ni en penalización. |
| Evidencia en contra | Siempre se calcula y se reporta en `evidence_against`; reduce el componente. | No se oculta evidencia contradictoria. |
| Modelos de régimen | Solo HMM2 y Markov2 (`validated_models("regime_description")`); 3 estados nunca se consultan. | Decisión de Fase 4.1. |
| Re-ajuste | Walk-forward por posición de barra (`min_train` 252 en 1Day, 150 en 4Hour; cada 63/42 barras); salidas filtradas, no suavizadas. | Prefijo invariante; coste razonable. |
| Markov2 | `fit_markov_regimes` pasa `rng` a statsmodels ≥ 0.15. | Sin ello el ajuste cambiaba entre ejecuciones (hallado por los tests de prefijo). |
| GARCH en patrones | Recursión GARCH(1,1) explícita con backcast del entrenamiento. | El filtro fijo de `arch` usa límites de varianza de toda la muestra (fuga de ~2e-7). |
| Change points | PELT en ventana móvil de 120 barras, `min_size` 10, "reciente" ≤ 15 barras. | PELT completo es retrospectivo; la ventana móvil añade latencia explícita. |
| Hurst y entropía | Solo evidencia opcional (peso 0.5) y nunca requerida. | Features descriptivas (Fase 4.1). |
| Gaps | Evidencia auxiliar, peso 0.25, nunca requerida, solo si ≥ 0.5 ATR y ≥ 0.5%; se registran descartes. | Decisión de Fase 6 (ruido IEX). |
| Mean reversion | Requiere z extremo **y** evidencia de agotamiento; con estructura y régimen persistentes el score se limita a 0.40 y no confirma. | Evitar activación fuerte contra tendencias persistentes. |
| Divergencias | Una divergencia sola no confirma; hace falta ruptura de neckline. | Requisito de evidencia no divergente. |
| Multi-timeframe | Solo el estado del timeframe superior entra al score (una evidencia de estructura); los demás son contexto descriptivo. | Sin score multi-timeframe optimizado todavía. |
| Episodios abiertos | `is_open=True` en `pattern_events.csv`; su estado final es provisional. | La Fase 8 no debe tratarlos como cerrados. |

El `stability_score` de Fase 4.1 es técnico y usa criterios distintos para forecasts y diagnósticos. No se debe leer como probabilidad de acierto. El archivo local `phase4_1_results/stability_scores.csv` se regenera con `summarize_phase4_1.py`; no se versiona porque es un resultado de ejecución.
