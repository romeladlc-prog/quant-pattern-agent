# Fase 6.1 — Hardening, causalidad y sincronización documental

**Estado:** completada en código y tests sin red. La validación en vivo (`validate_structure.py` y los demás `validate_*`) requiere credenciales Alpaca/FRED y **no se ha vuelto a ejecutar** tras estos cambios. No hay Pattern Engine, señales, Pattern Score ni ML supervisado.

## Cambios de causalidad

| Área | Antes (Fase 6) | Fase 6.1 |
|---|---|---|
| Niveles históricos | `detect_breakouts(bars, levels)` recibía niveles agrupados con toda la muestra: centros, fusiones y tolerancia ATR dependían de swings y precios posteriores. | `levels_history` reconstruye el conjunto de niveles **as-of** cada barra; `build_levels_asof(bars, timestamp)` da el conjunto vigente en cualquier instante. Solo entran swings con `confirmation_index` ≤ barra as-of; ATR, cierre y mediana de volumen son los conocidos al cierre de esa barra. |
| Breakouts históricos | Comparados con niveles finales. | `detect_breakouts(bars)` compara el cruce de la barra *i* con los niveles as-of de la barra *i−1*. Cada evento lleva `level_asof_timestamp` < `timestamp` y `bar_index`. `detect_breakouts_fixed_levels` queda solo para niveles externos cuyo precio se conoce en `available_at` (usable desde la barra siguiente). |
| `breakout_state` del snapshot | Último breakout de toda la historia, aunque fuese antiguo. | Activo solo si `age_bars = última_barra − bar_index ≤ breakout_max_age_bars` (por defecto 5, inclusivo; edad 0 = barra del snapshot). Si no, `"inactive"`. Igual para `fake_breakout_state`. El último evento se conserva en `latest_breakout_event` / `latest_fake_breakout_event` con `age_bars`. |
| Fuerza relativa | `pct_change` con relleno implícito de pandas 2.x. | `pct_change(fill_method=None)`; unión por timestamp exacto (`validate="one_to_one"`); columna `benchmark_available`. Un hueco del benchmark deja NaN las métricas que usan esa barra. |
| 1Week | Sin marca de semana parcial. | `is_complete=True` solo si `start` ≤ lunes 00:00 ET y `end` ≥ viernes 16:00 ET de esa semana. Se mantienen `[start,end)`, la construcción desde 1Day cerradas y `source_last_session_close`. Sin calendario bursátil, una semana con viernes festivo queda incompleta hasta el viernes 16:00 ET (conservador). |

Las salidas as-of son equivalentes a recalcular sobre cada prefijo; los tests lo comprueban comparando prefijos contra la serie completa.

## Feed de datos

`ALPACA_DATA_FEED` (`.env`) es la única configuración de feed; `src.data.alpaca_client.data_feed()` la lee y valida contra los valores de alpaca-py (`iex`, `sip`, `delayed_sip`, …). Si falta, se usa **`iex`**, disponible en todos los planes; SIP nunca se asume. `download_raw_bars` y `download_historical_bars` aceptan `feed=` solo para comparaciones explícitas entre feeds y guardan el feed usado en `attrs["feed"]`. `validate_structure.py` ya no crea su propio cliente IEX: usa la Data Layer como el resto de validadores, y ahora termina antes del día actual.

Hasta la Fase 6 la Data Layer no pasaba `feed`, por lo que Alpaca aplicaba el feed por defecto de la suscripción, mientras la Fase 6 forzaba IEX. Para reproducir aquellas ejecuciones de Fases 1–5, define el feed que tu plan permita (`sip` si está incluido).

**Limitaciones IEX:** solo operaciones ejecutadas en IEX (una fracción pequeña del volumen consolidado). Efectos esperados: volumen y volumen relativo mucho menores y más ruidosos; aperturas y cierres de la primera/última operación IEX, que inflan los gaps diarios (548–690 de 756 días en Fase 6); máximos y mínimos menos extremos, con efecto sobre ATR, pivots y niveles. Las conclusiones de estructura deben confirmarse con feed consolidado antes de calibrar umbrales.

## Metadata de modelos

`src/models/model_status.py` codifica las decisiones de la Fase 4.1 sin impedir ningún ajuste: `benchmark_only` (AutoReg, ARIMA, HAR-RV), `role` (GARCH `primary_volatility`, EGARCH `complementary_volatility`, HAR-RV `benchmark_volatility`), `stable_for_current_use` (HMM2, Markov2, Kalman local level, GARCH, EGARCH) y `experimental` + `warning="unstable_across_assets"` (HMM3, Markov3). Cada ajuste devuelve `model_status`; los modelos de 3 estados además emiten `UserWarning`. `validated_models(purpose)` lista solo componentes estables, no experimentales y no benchmark; `allowed_models(purpose)` incluye los que llevan advertencia documentada (Kalman local linear trend para `slope_description`).

## Seguridad

`src/security.redact_secrets` sustituye los valores de `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FRED_API_KEY` y parámetros `api_key=`/`token=` en mensajes. FRED solo acepta la clave como parámetro de URL, que `requests` incluye en el texto de `HTTPError`; `fetch_fred_series` convierte cualquier `RequestException` en `RuntimeError` redactado y sin encadenar la excepción original (`from None`), de modo que el traceback tampoco la muestra. Lo mismo se aplica a errores de Alpaca bars y news, y al traceback que `validate_structure.py` escribe en su log.

## Tests y CI

`pytest` (44 tests, red bloqueada por `tests/conftest.py`): niveles as-of e invariancia por prefijo, `confirmed_at`, breakouts causales e invariantes al añadir barras, recencia de `breakout_state`, RS con fechas faltantes, `is_complete` semanal (completa, parcial al inicio, corte a mitad de semana, antes del cierre del viernes), sanitización de claves, configuración de feed y metadata de modelos. `.github/workflows/tests.yml` usa Python 3.12, instala `requirements-dev.txt`, ejecuta solo estos tests y compila los scripts; no usa secrets ni llama a Alpaca/FRED.

## Riesgos que siguen abiertos

- La validación en vivo de Fase 6 (`phase6_results/summary.md`) es anterior a 6.1: sus recuentos de rupturas se obtuvieron con niveles de muestra completa. Hay que repetirla con credenciales.
- El as-of cuesta O(barras × swings): ~3–4 s para 756 barras diarias sintéticas.
- Gaps ruidosos en IEX, breadth con sesgo de supervivencia y macro sin vintages siguen como en Fase 6.
