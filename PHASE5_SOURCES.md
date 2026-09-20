# Fase 5: fuentes y tiempo de disponibilidad

La capa `src/context` no produce señales. Todas las funciones de snapshot reciben `as_of` con zona horaria y excluyen registros cuyo `available_at` sea posterior.

| Contexto | Fuente | Unidad | Regla de disponibilidad |
|---|---|---|---|
| Fed assets, reservas, TGA | FRED/ALFRED: WALCL, WRESBAL, WDTGAL | millones USD | Con clave FRED: vintage `realtime_start` y uso desde 00:00 UTC del día siguiente; sin clave: solo snapshot actual tras descarga. |
| Reverse repo | FRED/ALFRED: RRPONTSYD | miles de millones USD, convertidos a millones para net liquidity | Misma regla. |
| SOFR, Treasury 2Y/10Y, spread corporativo HY | FRED/ALFRED: SOFR, DGS2, DGS10, BAMLH0A0HYM2 | porcentaje o puntos porcentuales | Misma regla. 2s10s = 10Y menos 2Y. |
| VIX | CSV diario Cboe | puntos VIX | 20:00 America/New_York del día de observación, convención conservadora. |
| Liquidez del activo y amplitud | Alpaca NORMALIZED 1Day | USD, volumen, ratios | 16:30 America/New_York de la sesión, tras el cierre. |
| Noticias | Alpaca News | texto, recuentos, score −1..1 | `created_at` como publicación y `updated_at` como disponibilidad conservadora del texto actual; score reproducible sobre titular. |

La cesta de amplitud **no es el Nasdaq 100 completo**. Usa de forma fija AAPL, MSFT, NVDA, AMZN, META, GOOGL, GOOG, AVGO, TSLA y AMD, con igual peso por componente. Su historial tiene sesgo de supervivencia por usar una membresía fija actual. La SMA exige la ventana completa y al menos 8 componentes válidos; los máximos/mínimos usan 252 barras anteriores, excluyendo la actual. No se rellena un precio faltante. Las series macro solo se mantienen hasta su próxima observación al construir el estado de la fecha; no se interpolan.

El CSV público de FRED devuelve los valores revisados hoy. **No es seguro para backtests históricos**. Sin `FRED_API_KEY`, el módulo registra `release_date` desconocida y `asof_safe=false`, y esas filas se hacen disponibles solo en el instante de descarga. Con clave, consulta vintages ALFRED y conserva `observation_date`, `release_date` y `available_at` por revisión. Una fecha de vintage no ofrece hora intradía exacta; el desfase de un día evita usarla antes de publicarse en una validación diaria.

La métrica `net_liquidity = WALCL - WDTGAL - 1000 × RRPONTSYD` se expresa en millones USD. TGA es nivel de miércoles y RRP es diario; se toma el RRP más reciente no posterior al miércoles. Las variaciones son diferencias de una y cuatro observaciones semanales; el slope es la diferencia de cuatro semanas dividida entre cuatro. `expanding` y `contracting` dependen del signo de la diferencia de cuatro semanas, y `neutral` corresponde a cero. Es una convención descriptiva, no una identidad económica.

El turnover requiere acciones en circulación con fecha y queda faltante. El sentimiento de noticias se calcula solo con un léxico fijo de palabras en inglés del titular; texto y score se guardan en columnas distintas. Si no hay palabras reconocidas, el score es faltante. No se utiliza un LLM ni se infiere sentimiento de artículos sin texto verificable.

Fuentes oficiales: [FRED/ALFRED API](https://fred.stlouisfed.org/docs/api/fred/), [FRED real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html), [Cboe VIX historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data), [Alpaca historical API](https://docs.alpaca.markets/us/docs/historical-api).
