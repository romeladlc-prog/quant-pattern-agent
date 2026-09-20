# Decisiones del Quant Model Layer

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

El `stability_score` de Fase 4.1 es técnico y usa criterios distintos para forecasts y diagnósticos. No se debe leer como probabilidad de acierto. El archivo local `phase4_1_results/stability_scores.csv` se regenera con `summarize_phase4_1.py`; no se versiona porque es un resultado de ejecución.
