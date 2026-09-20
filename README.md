# Quant Pattern Agent

Sistema cuantitativo en desarrollo para estudiar patrones de mercado con horizonte de *swing trading*. El alcance actual llega hasta la **Fase 5: External Context Engine**. No genera señales de compra o venta, recomendaciones ni órdenes.

## Estado y arquitectura

| Capa | Implementación actual |
|---|---|
| Data Layer | Descarga Alpaca, respuesta RAW preservada y barras NORMALIZED en UTC. |
| Visualization | Visor local Streamlit de OHLCV, sesiones y marcos temporales. |
| Feature Engine | 15 features estadísticas causales sobre barras NORMALIZED. |
| Quant Model Layer | Retorno, volatilidad, Kalman, complejidad, change points y regímenes. |
| External Context Engine | VIX, liquidez macro y del activo, amplitud de una cesta fija y noticias. |

Las capas futuras figuran en [docs/ROADMAP.md](docs/ROADMAP.md). La [arquitectura](docs/ARCHITECTURE.md) explica los límites entre capas y RAW frente a NORMALIZED.

## Instalación (Python 3.12)

Desde PowerShell en la raíz del proyecto:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe --version
```

Si `py -3.12` no está disponible, instala Python 3.12 o sigue [PHASE4_1_SETUP.md](PHASE4_1_SETUP.md) para usar `uv` en un entorno separado. El `.venv` antiguo de Python 3.14 no se necesita para las validaciones actuales.

## Configuración

Copia `.env.example` a `.env` y completa `ALPACA_API_KEY` y `ALPACA_SECRET_KEY`. `ALPACA_BASE_URL` tiene el valor de paper trading en el ejemplo. `FRED_API_KEY` es opcional para el snapshot macro actual, pero necesaria para solicitar vintages ALFRED y evaluar contexto macro histórico sin usar revisiones futuras. Nunca subas `.env` ni copies claves en código, logs o incidencias.

```powershell
Copy-Item .env.example .env
```

## Uso

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
.\.venv312\Scripts\python.exe main.py
```

`app.py` es un visor de datos, no un scanner. `main.py` muestra una consulta de barras de ARM. Ambos requieren acceso a Alpaca.

## Validación

```powershell
.\.venv312\Scripts\python.exe validate_market_data.py
.\.venv312\Scripts\python.exe validate_features.py
.\.venv312\Scripts\python.exe validate_models.py
.\.venv312\Scripts\python.exe validate_walk_forward.py
.\.venv312\Scripts\python.exe summarize_phase4_1.py
.\.venv312\Scripts\python.exe validate_context.py
```

Los scripts consultan fuentes externas y pueden tardar varios minutos. El audit walk-forward usa ARM, NVDA, AMD, AVGO y QQQ. Los resultados generados, CSV y logs son locales e ignorados por Git; [docs/VALIDATION.md](docs/VALIDATION.md) resume las últimas ejecuciones verificadas. Los parámetros de ventanas están en [PHASE4_1_SETUP.md](PHASE4_1_SETUP.md).

## Fuentes y límites

Alpaca aporta OHLCV y noticias; Cboe aporta VIX; FRED/ALFRED aporta series monetarias y tipos. La amplitud usa una cesta fija de diez acciones, **no** el Nasdaq 100 completo. [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) documenta frecuencia, timestamps y disponibilidad.

Las conclusiones actuales están en [docs/MODEL_DECISIONS.md](docs/MODEL_DECISIONS.md). AutoReg y ARIMA siguen como benchmarks; GARCH es la referencia principal de volatilidad por ahora. Las noticias tienen poca cobertura de titulares puntuables, y la liquidez macro histórica **no es apta para backtesting** sin vintages apropiados. No se han implementado Market Structure, Pattern Engine, backtesting, ML supervisado ni Ensemble.
