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
Pattern Engine
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

- `src/data/alpaca_client.py`: descarga y normaliza con el feed único `ALPACA_DATA_FEED` (`data_feed()`, IEX por defecto). 1Week se construye desde 1Day cerradas y marca `is_complete`. **RAW** es la respuesta del proveedor sin transformar, con su índice y campos originales. **NORMALIZED** es una copia ordenada con `timestamp` UTC y OHLCV, limitada al intervalo solicitado `[start, end)`, con filtros de sesión y agregación cuando procede. La normalización no altera RAW.
- `app.py`: visualiza barras NORMALIZED. No calcula decisiones.
- `src/features/statistical_features.py`: features causales sobre NORMALIZED. Las ventanas incluyen la barra actual y anteriores, nunca futuras; los valores iniciales no definidos quedan `NaN`.
- `src/models/`: familias separadas de retorno, volatilidad, estado espacial, regímenes y complejidad. Los change points son retrospectivos. `model_status.py` declara para qué está validado cada modelo; las capas futuras deben consultar `validated_models()` en lugar de elegir modelos por nombre. `src/validation/` genera folds temporales auditables.
- `src/context/`: VIX, macro, liquidez del activo, breadth y noticias. `MarketContextSnapshot` compone valores disponibles hasta un `as_of` con zona horaria. No crea señales.
- `src/structure/`: describe pivots confirmados, HH/HL/LH/LL, niveles, rupturas, compresión, aceleración, gaps y fuerza relativa. Consume OHLCV NORMALIZED sin modificar `src/models/` ni `src/context/`. `MarketStructureSnapshot` es independiente por ticker y timeframe; no combina periodos ni emite señales. Los niveles históricos se reconstruyen as-of cada barra (`levels_history`, `build_levels_asof`) y los breakouts se comparan con los niveles de la barra anterior; `breakout_state` solo es activo dentro de `breakout_max_age_bars`.
- `src/patterns/`: Pattern Engine. Construye un frame de evidencia causal por barra (features, estructura as-of, breakouts 6.1, contexto cuantitativo re-ajustado walk-forward y contexto externo as-of), y cada detector recorre las barras con una máquina de estados que solo ve filas 0..i (`History`). Solo el estado del timeframe superior entra al score; otros timeframes son contexto descriptivo. Produce `PatternResult` y una tabla de episodios; no emite señales.
- `src/security.py`: redacción de credenciales en mensajes de error y logs.

## Regla temporal

La fecha de observación **no** implica disponibilidad. La capa de contexto conserva `observation_date`, `release_date` cuando existe y `available_at`. Los snapshots filtran por `available_at <= as_of`. Las barras diarias se consideran disponibles después del cierre; FRED sin vintages se limita al snapshot actual. En estructura, un pivot existe desde `confirmed_at`, un nivel desde la barra as-of en que todos sus swings están confirmados, y un cruce se mide contra niveles conocidos antes de la barra que cruza. Más detalle en [DATA_SOURCES.md](DATA_SOURCES.md).
