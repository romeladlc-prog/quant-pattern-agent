# Arquitectura

```text
Data Layer
    ↓
Visualization
    ↓
Feature Engine
    ↓
Quant Model Layer
    ↓
External Context Engine
    ↓
Market Structure Engine
    ↓
Pattern Engine (futuro)
    ↓
Backtesting (futuro)
    ↓
Supervised ML (futuro)
    ↓
Ensemble (futuro)
    ↓
Knowledge Layer / Obsidian (futuro)
    ↓
Agent / Scanner (futuro)
```

El diagrama representa la evolución prevista. El visor de Fase 2 consume la Data Layer directamente; External Context no modifica el ajuste de los modelos cuantitativos. Hoy no existe un pipeline ejecutable de extremo a extremo hasta Ensemble.

## Fronteras actuales

- `src/data/alpaca_client.py`: descarga y normaliza. **RAW** es la respuesta del proveedor sin transformar, con su índice y campos originales. **NORMALIZED** es una copia ordenada con `timestamp` UTC y OHLCV, limitada al intervalo solicitado `[start, end)`, con filtros de sesión y agregación cuando procede. La normalización no altera RAW.
- `app.py`: visualiza barras NORMALIZED. No calcula decisiones.
- `src/features/statistical_features.py`: features causales sobre NORMALIZED. Las ventanas incluyen la barra actual y anteriores, nunca futuras; los valores iniciales no definidos quedan `NaN`.
- `src/models/`: familias separadas de retorno, volatilidad, estado espacial, regímenes y complejidad. Los change points son retrospectivos. `src/validation/` genera folds temporales auditables.
- `src/context/`: VIX, macro, liquidez del activo, breadth y noticias. `MarketContextSnapshot` compone valores disponibles hasta un `as_of` con zona horaria. No crea señales.
- `src/structure/`: describe pivots confirmados, HH/HL/LH/LL, niveles, rupturas, compresión, aceleración, gaps y fuerza relativa. Consume OHLCV NORMALIZED sin modificar `src/models/` ni `src/context/`. `MarketStructureSnapshot` es independiente por ticker y timeframe; no combina periodos ni emite señales.

## Regla temporal

La fecha de observación **no** implica disponibilidad. La capa de contexto conserva `observation_date`, `release_date` cuando existe y `available_at`. Los snapshots filtran por `available_at <= as_of`. Las barras diarias se consideran disponibles después del cierre; FRED sin vintages se limita al snapshot actual. Más detalle en [DATA_SOURCES.md](DATA_SOURCES.md).
