# Fase 4.1: entorno y validación

Usa Python 3.12 en un entorno separado. El `.venv` existente no se borra ni se modifica para crearlo.

En PowerShell, desde la raíz del proyecto:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -c "import sys, arch, hmmlearn, ruptures, statsmodels; print(sys.version); print(arch.__version__, hmmlearn.__version__, ruptures.__version__, statsmodels.__version__)"
.\.venv312\Scripts\python.exe validate_market_data.py
.\.venv312\Scripts\python.exe validate_features.py
.\.venv312\Scripts\python.exe validate_models.py
.\.venv312\Scripts\python.exe -u validate_walk_forward.py *> phase4_1_validation.log
```

Si `py -3.12` no existe, instala Python 3.12 y repite. En este checkout se usó `uv` sin tocar `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install uv
.\.venv\Scripts\uv.exe python install 3.12 --install-dir .python312
$env:UV_CACHE_DIR = (Join-Path (Get-Location) '.uv_cache')
.\.venv\Scripts\uv.exe venv --python .\.python312\cpython-3.12.14-windows-x86_64-none\python.exe .venv312
.\.venv\Scripts\uv.exe pip install --python .venv312\Scripts\python.exe -r requirements.txt
```

Comprueba el nombre exacto de la carpeta de Python descargada si `uv` instala otra versión de 3.12.

`validate_walk_forward.py` usa ARM, NVDA, AMD, AVGO y QQQ en fechas NORMALIZED 1Day comunes. Por defecto usa 252 barras iniciales, paso de 63 barras, horizonte 1 y ventana expansiva. Opciones: `--min-train`, `--step`, `--horizon`, `--window` (ventana rolling). El horizonte mayor que 1 se aplica a retornos; volatilidad y diagnósticos de estado requieren 1. El grid ARIMA en este audit repetido es p,q=0..2; el modelo de Fase 4 conserva el grid completo 0..5.

Cada fold conserva fecha inicial y final de entrenamiento, fecha de prueba, forecast, valor real y error en `phase4_1_results/`. El score de estabilidad es técnico: media de convergencia, estabilidad relativa de parámetros, consistencia por fold, consistencia por activo y ventaja normalizada frente al benchmark. No equivale a capacidad predictiva ni a una señal.

Tras el audit, ejecuta `.\.venv312\Scripts\python.exe summarize_phase4_1.py` para actualizar `phase4_1_results/summary.md` y `stability_scores.csv`. Las puntuaciones de Kalman, regímenes y PELT usan sus diagnósticos propios, identificados en el CSV; no son directamente comparables con una mejora de forecast. El audit de regímenes usa el último corte de entrenamiento por activo, mientras los forecasts y Kalman se evalúan en ocho folds por activo.

**Revalidación de regímenes (tras Fase 7).** Los resultados de HMM/Markov de este audit son `legacy_pre_revalidation`. No sobrescribas `phase4_1_results/`: ejecuta

```powershell
.\.venv312\Scripts\python.exe -u validate_regime_reproducibility.py *> phase4_1_revalidation.log
```

Escribe en `phase4_1_revalidation/` y compara con `phase4_1_results/regimes.csv` si existe. Ver [docs/phases/PHASE_4_1_REVALIDATION.md](docs/phases/PHASE_4_1_REVALIDATION.md).

Alpaca requiere las credenciales de `.env`. La consulta termina antes del día actual para excluir datos SIP recientes. Si una descarga falla, el audit se detiene y no atribuye resultados de otro universo a los cinco activos pedidos.
