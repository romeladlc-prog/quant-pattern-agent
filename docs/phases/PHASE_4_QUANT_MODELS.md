# Fase 4 — Quant Model Layer

**Estado:** familias implementadas y validadas sobre ARM 1Day. `src/models/` separa:

- `time_series_models.py`: AutoReg con selección AIC/BIC, ARIMA benchmark y retornos ingenuos.
- `volatility_models.py`: GARCH, EGARCH, GJR-GARCH con Normal/Student-t y HAR-RV.
- `state_space_models.py`: Kalman local linear trend causal, y especificaciones comparables de Fase 4.1.
- `complexity_models.py`: Hurst y entropías como features.
- `regime_models.py`: PELT, Markov Switching y Gaussian HMM con estados descriptivos.

`validate_models.py` usó 756 barras ARM (2023-09-14 a 2026-09-18), 566 retornos de entrenamiento y 189 de prueba. AutoReg y ARIMA no mejoraron al retorno cero en esa división. EGARCH obtuvo un margen pequeño en volatilidad; ese resultado motivó el audit de Fase 4.1. Los estados no reciben etiquetas bull/bear durante el ajuste y los change points son retrospectivos. No se generan señales.
