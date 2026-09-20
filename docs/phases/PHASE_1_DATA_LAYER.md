# Fase 1 — Data Layer

**Estado:** completada y validada. `src/data/alpaca_client.py` descarga datos de Alpaca en los marcos 15Min, 1Hour, 4Hour, 1Day y 1Week. Mantiene la respuesta **RAW** sin alterar y crea una copia **NORMALIZED** con timestamp UTC, OHLCV ordenado, intervalo `[start, end)` y sesiones regular/extended cuando corresponden. Las agregaciones intradía y semanal se realizan sobre barras de origen documentadas en el módulo.

**Verificación:** `validate_market_data.py` recorrió RAW y NORMALIZED; ARM 1Day tuvo 312 barras entre 2024-01-02 y 2025-03-31 y 1Week NORMALIZED tuvo 82 observaciones. La cobertura y permisos de Alpaca pueden variar.

**Límite:** el timestamp de barra no es por sí mismo una fecha de publicación de variables externas. Las capas posteriores deben usar su propia regla `available_at`.
