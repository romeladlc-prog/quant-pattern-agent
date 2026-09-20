# Validaciones ejecutadas

Última referencia: ejecuciones locales del **20 de septiembre de 2026** con Python 3.12. Los logs y CSV resultantes se ignoran en Git para evitar datos temporales o sensibles. La tabla resume evidencia observada; repetir los scripts puede producir resultados distintos cuando las fuentes se actualicen.

| Fase | Script | Activos / universo | Rango verificado | Resultado | Límite principal |
|---|---|---|---|---|---|
| 1 | `validate_market_data.py` | ARM | Diario 2024-01-02 a 2025-03-31; también intradía y semanal | RAW y NORMALIZED OK; 312 barras 1Day y 82 semanas NORMALIZED | Depende de Alpaca y de su suscripción. |
| 3 | `validate_features.py` | ARM | 2024-01-02 a 2025-03-31 | 312 barras, 15 features; índice, fórmulas y ausencia de look-ahead OK | Diagnósticos estadísticos descriptivos. |
| 4 | `validate_models.py` | ARM | 2023-09-14 a 2026-09-18 | 756 barras; 566 retornos train y 189 test; retornos, volatilidad, Kalman, complejidad, change points y HMM ejecutados con Python 3.12 | Una sola división temporal; Markov3 requiere cautela. |
| 4.1 | `validate_walk_forward.py` | ARM, NVDA, AMD, AVGO, QQQ | 756 fechas comunes, 2023-09-14 a 2026-09-18 | 8 folds por activo; 160 filas de retorno y 159 de volatilidad; código de salida 0 | Solo 8 fechas de test por activo; una falla GJR-GARCH en AVGO; RV20 solapada. |
| 5 | `validate_context.py` | ARM, NVDA, AMD, AVGO, QQQ; cesta fija de 10 para breadth | OHLCV 2023-09-14 a 2026-09-18; VIX 1990-01-02 a 2026-09-18 | Código de salida 0; cinco snapshots; checks de disponibilidad, duplicados, unidades y snapshot histórico | Macro CSV solo actual sin `FRED_API_KEY`; titulares puntuables escasos. |

## Resultados que afectan decisiones

- Retorno: AutoReg y ARIMA superaron al retorno cero en solo 2/5 activos por RMSE. Quedan como benchmarks.
- Volatilidad: RMSE promedio GARCH 0.00490, EGARCH 0.00506 y HAR-RV 0.00806; GARCH es la referencia actual.
- Estado espacial: local level tuvo 3% de estimaciones en borde frente a 57% para local linear trend.
- Regímenes: HMM2 y Markov2 convergieron en cinco activos; los modelos de tres estados no tuvieron estabilidad uniforme.
- Fase 5: VIX y medidas de volumen disponibles; macro actual disponible pero su histórico no es apto para backtest sin vintages; tres tickers tuvieron menos de tres titulares puntuados en siete días.

Para reproducir y revisar cada fold, consulta [PHASE4_1_SETUP.md](../PHASE4_1_SETUP.md). Para fechas de publicación y límites de contexto, consulta [DATA_SOURCES.md](DATA_SOURCES.md).
