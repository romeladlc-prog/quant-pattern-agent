# Changelog

## Fase 6 — Market Structure Engine

- Detectores causales de pivots, estructura, niveles, rupturas/fallos, compresión/expansión, aceleración, gaps y fuerza relativa.
- Snapshot descriptivo por activo y timeframe; validación de ARM, NVDA, AMD, AVGO y QQQ en 4Hour, 1Day y 1Week, más ARM 1Hour.
- Tests sintéticos y log/resumen de validación. Sin señales ni recomendaciones.

## 2026-09-20 — Documentación hasta Fase 5

- Documentadas arquitectura, decisiones de modelos, fuentes, validaciones y roadmap.
- Fase 5: External Context Engine independiente para VIX, liquidez macro y del activo, amplitud y noticias, con `available_at` y validación de cinco activos.
- Fase 4.1: Python 3.12, walk-forward cross-asset, diagnósticos de estabilidad y benchmarks documentados.

## 2026-09-19 — Fases 1 a 4

- Fase 1: Data Layer Alpaca RAW y NORMALIZED con validación temporal.
- Fase 2: visor Streamlit de barras y sesiones.
- Fase 3: motor de 15 features estadísticas causales.
- Fase 4: familias cuantitativas de retorno, volatilidad, Kalman, complejidad, change points y regímenes, sin señales.
