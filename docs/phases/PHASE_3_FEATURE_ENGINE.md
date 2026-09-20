# Fase 3 — Feature Engine

**Estado:** completada y validada. `src/features/statistical_features.py` calcula 15 columnas sobre OHLCV NORMALIZED: retornos, media/desviación/z-score rolling, momentum, RV20, skew, kurtosis, drawdown, autocorrelación y slope de precio, entre otras. Una ventana usa la barra actual y anteriores; las observaciones iniciales indefinidas permanecen `NaN`.

**Verificación:** `validate_features.py` comprobó índice, fórmulas y ausencia de look-ahead sobre 312 barras ARM 1Day de 2024-01-02 a 2025-03-31. ADF, Ljung-Box y Jarque-Bera son diagnósticos descriptivos, no pruebas de utilidad predictiva.
