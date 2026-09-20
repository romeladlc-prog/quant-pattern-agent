# Fuentes de datos y disponibilidad

| Fuente | Dato | Frecuencia | Timestamp y `available_at` | Riesgo / límite |
|---|---|---|---|---|
| Alpaca Historical Bars | OHLCV de acciones y QQQ; RAW y NORMALIZED | 15Min, 1Hour, 4Hour, 1Day, 1Week según consulta | `timestamp` UTC de barra; contexto diario disponible a las 16:30 ET de la sesión | Cobertura y acceso dependen de la suscripción; ajustes corporativos requieren revisión. |
| Cboe VIX histórico | Cierre VIX | Diario | Fecha de sesión; `available_at` conservador a las 20:00 ET | No incluye estructura de futuros ni put/call ratio. |
| FRED/ALFRED | WALCL, WRESBAL, WDTGAL, RRPONTSYD, SOFR, DGS2, DGS10, BAMLH0A0HYM2 | Semanal o diario | `observation_date`; con clave, `release_date` del vintage y uso desde 00:00 UTC del día siguiente | CSV público refleja revisiones actuales y **no** es apto para backtest histórico. |
| Alpaca News | Titular, resumen, fuente y fechas | Por artículo | `created_at` como publicación; `updated_at` como disponibilidad conservadora del texto actual | Léxico de titulares ingleses tiene poca cobertura y no representa sentimiento completo. |
| Cesta fija de breadth | Diez acciones Alpaca: AAPL, MSFT, NVDA, AMZN, META, GOOGL, GOOG, AVGO, TSLA, AMD | Diario | Tras cierre, 16:30 ET | No es el Nasdaq 100 completo; sesgo de supervivencia histórico. |

La liquidez del activo (`dollar_volume`, `relative_volume`, Amihud y gap) se deriva de OHLCV; no es una fuente macro. Turnover queda faltante sin acciones en circulación fechadas. VIX term structure y put/call ratio quedan faltantes. Para detalles de unidades, fórmula de liquidez neta y desfases, véase [PHASE5_SOURCES.md](../PHASE5_SOURCES.md).

Las fuentes originales documentadas por el proveedor son [Alpaca Historical API](https://docs.alpaca.markets/us/docs/historical-api), [Cboe VIX Historical Data](https://www.cboe.com/tradable_products/vix/vix_historical_data) y [FRED/ALFRED API](https://fred.stlouisfed.org/docs/api/fred/).
