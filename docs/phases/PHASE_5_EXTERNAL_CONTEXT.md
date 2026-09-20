# Fase 5 — External Context Engine

**Estado:** completada como capa separada `src/context/`. `MarketContextSnapshot` compone contextos para ticker y `as_of` con zona horaria; cada fuente filtra por `available_at`. No crea señales.

| Familia | Implementación | Límite |
|---|---|---|
| VIX | Cierre, cambio diario y semanal, z-score rolling, CSV de Cboe | No hay term structure ni put/call ratio. |
| Liquidez macro | WALCL, reservas, TGA, RRP, SOFR, 2Y/10Y, 2s10s y spread HY; `net_liquidity = WALCL - TGA - RRP` con conversión de unidades | Sin vintages FRED/ALFRED, solo snapshot descriptivo actual. No usar su histórico en backtest. |
| Liquidez del activo | Dollar volume, relative volume, Amihud, volatilidad/volumen y gap overnight | Turnover faltante sin shares outstanding fechadas. |
| Breadth | Porcentaje sobre SMA20/50/200, advances/declines, máximos/mínimos de cesta fija de diez acciones | No es Nasdaq 100 completo; sesgo de supervivencia. |
| Noticias | Recuentos 1d/7d, intensidad y score reproducible por léxico de titulares | Pocos titulares puntuables; no hay LLM ni sentimiento robusto. |

`validate_context.py` pasó en ARM, NVDA, AMD, AVGO y QQQ. Cboe proporcionó 9 276 cierres VIX hasta 2026-09-18 y Alpaca noticias de siete días; solo 1, 10, 1, 0 y 7 titulares tuvieron score para ARM, NVDA, AMD, AVGO y QQQ. Los checks incluyeron duplicados, unidades, faltantes, fechas y un snapshot histórico sin datos publicados después del `as_of`.

**Candidatos para un futuro Ensemble:** VIX, dollar volume y relative volume. Breadth puede aportar contexto con su sesgo declarado. News sentiment y macro histórico aún no reúnen las condiciones para incorporarse como features históricas fiables. Detalles en [DATA_SOURCES.md](../DATA_SOURCES.md) y [PHASE5_SOURCES.md](../../PHASE5_SOURCES.md).
