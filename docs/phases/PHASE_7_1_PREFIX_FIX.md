# Fase 7.1: causa raíz del fallo de prefijo con datos reales

## Síntoma

`validate_patterns.py` con datos Alpaca (2026-10-02, laptop de Romel) falló en el primer corte de ARM 1Day:

```
AssertionError: look-ahead: 5362 results differ at T=2025-03-25 04:00:00+00:00,
e.g. [('bullish_exhaustion', 0), ('bullish_exhaustion', 1), ('bullish_exhaustion', 2)]
```

5362 = 14 patrones × 383 barras: **todos** los resultados ≤ T diferían, desde la barra 0. Los tests sintéticos pasaban.

## Reproducción sin red

No hay credenciales en la nube, así que se reprodujo con un fixture derivado con la misma forma temporal que la descarga real: 1Day desde el inicio de la serie, 1Week derivado, 4Hour solo desde hace 400 días y 1Hour solo desde hace 90 días (como `load_live`). Con ese fixture el fallo sale idéntico: mismo T (2025-03-25), todas las barras y el mismo ejemplo (`bullish_exhaustion`, 0).

`validate_patterns.py --synthetic` ahora incluye ese caso: SYN_A lleva 4Hour y 1Hour solo en el último 20% de la muestra. Con el código anterior falla igual que en datos reales.

## Causa 1: el esquema de `mtf_context` dependía de datos futuros

`run_pattern_engine` omitía un timeframe de contexto cuando no tenía ninguna barra completa:

```python
if other_tf == timeframe or other_bars is None or complete_bars(other_bars).empty:
    continue
```

- **A) datos hasta T = 2025-03-25:** 4Hour (desde ~2025-08) y 1Hour (desde ~2026-07) quedan vacíos, se omiten y `mtf_context = {"1Week"}`.
- **B) muestra completa:** 4Hour y 1Hour tienen barras, se adjuntan y `mtf_context = {"1Week", "4Hour", "1Hour"}`, con todos los valores en `None` para las barras ≤ T.

**Primer input que diverge:** las columnas `h4_*` y `h1_*` del frame de evidencia, desde la barra 0, como diferencia de esquema (existen solo en B), con origen en `attach_timeframe` (contexto multi-timeframe). El único campo de `PatternResult` que difiere es `mtf_context` (claves `4Hour` y `1Hour`). Score, scores por componente, evidencia, estado de la máquina, niveles, breakouts, Kalman, GARCH, HMM2, Markov2, change points, fuerza relativa, contexto semanal y contexto externo son idénticos en ese corte.

No es una fuga de valores, pero sí de información: la presencia de la clave revela que más adelante habrá datos intradía. La propiedad `pattern(data[:T])[T] == pattern(data_full)[T]` exige la misma forma.

**Corrección:** un timeframe de contexto pedido siempre se adjunta. Si aún no tiene barras cerradas, sus columnas existen con valores ausentes (`attach_empty_timeframe`) y su clave en `mtf_context` vale `None` en todos los campos. `meta["context_timeframes_with_data"]` lista los que tienen datos. `higher_timeframe_available` se basa en datos reales, no en la presencia de columnas.

## Causa 2 (oculta detrás de la 1): HMM2 no era determinista bit a bit

Corregida la causa 1, el mismo layout con 5 tickers dio otro fallo en cortes posteriores (ARM 1Day T=2025-12-29 y NVDA 1Day T=2024-05-17):

```
first divergent input: hmm_p_high at bar 586 [value] prefix=0.04457636206881064 full=0.04457636206880516
origin: HMM2 (quant_context)
```

La barra 586 es un re-ajuste. El entrenamiento es idéntico en A y B (filas ≤ 586), pero los parámetros diferían en los últimos bits. hmmlearn inicializa las medias con `sklearn.cluster.KMeans`, cuyas reducciones OpenMP con varios hilos no son reproducibles bit a bit: el mismo ajuste repetido 12 veces dio 9 resultados distintos con 4 hilos y 0 con un hilo OpenMP. No depende de la longitud de la muestra; es ruido entre ejecuciones. Con datos reales, en una máquina con más núcleos, es probable que este fallo apareciera al corregir el primero.

**Corrección:** `fit_hmm_regimes` ajusta bajo `threadpool_limits(1, "openmp")`. El algoritmo, las semillas y la selección no cambian; el ajuste pasa a ser determinista.

## Warnings `Model is not converging`

- **Origen:** `hmmlearn` (logger `hmmlearn.base`, `ConvergenceMonitor.report`), dentro de `fit_hmm_regimes`, es decir, HMM2 en cada re-ajuste del contexto cuantitativo. Se emite cuando la log-verosimilitud del EM **baja** entre iteraciones (deltas de ~1e-6 a 1e-5, en ajustes con covarianza completa y `min_covar`).
- **Problema de calidad:** `monitor_.converged` de hmmlearn devuelve True también si el EM baja o si agota `n_iter`. El código anterior solo avisaba con `not monitor_.converged`, así que esos ajustes se usaban sin ninguna bandera.
- **¿Explican el fallo de prefijo?** No. Un re-ajuste ≤ T usa las mismas filas en A y B, así que su resultado (convergente o no) es el mismo en ambos. El test `test_hmm_em_non_convergence_is_flagged_and_prefix_stable` lo comprueba con un ajuste no convergente real. La no reproducibilidad de la causa 2 es independiente: ocurre también con ajustes que convergen.
- **Qué hace ahora el Pattern Engine:** el ajuste se sigue usando (sin cambio metodológico), pero queda marcado:
  - `fit_hmm_regimes` devuelve `fit_quality` (modelo elegido) y `seed_quality` (las tres semillas), con `monitor_converged`, `iterations`, `hit_max_iter`, `loglik_decreases` y `converged` (estricto: sin bajadas y sin agotar iteraciones). Los mensajes de hmmlearn se recogen en un `UserWarning` atribuido a la semilla en lugar de imprimirse sueltos.
  - Columnas por barra `hmm_fit_converged`, `markov_fit_converged`, `kalman_fit_converged` y `garch_fit_converged` del ajuste vigente (`None` si no hay ajuste), expuestas en `regime_context` y `volatility_context`.
  - `notes` añade `model_not_converged:<modelos>` en cada resultado cubierto por un ajuste marcado.
  - `attrs["fit_diagnostics"]` del contexto cuantitativo lista cada re-ajuste con su bandera.

## Diagnóstico `--debug-prefix`

```
python validate_patterns.py --debug-prefix
```

Si `prefix_check` falla, imprime y guarda en `phase7_validation.log` el primer timestamp divergente, los patrones, los campos de `PatternResult` que difieren (incluidas claves anidadas), el primer input divergente (columna, barra, valores A/B o diferencia de esquema), el módulo de origen y si los re-ajustes y diagnósticos de ajuste ≤ T coinciden. Después relanza el mismo `AssertionError`. No relaja ni reduce la comprobación, y sin fallo no escribe nada extra.

## ¿Cambian los PatternResults históricos?

Medido sobre los 4 × 8 400 resultados de `--synthetic` (mismo input, código anterior frente a nuevo):

- `state`, episodios, timestamps, `score`, `score_*`, `evidence_for/against` y condiciones: **sin cambios**.
- `regime_context` y `volatility_context`: **4 claves nuevas** de convergencia en todos los resultados.
- `notes`: `model_not_converged:hmm2` añadido en 6 370 resultados (barras con un HMM2 cuyo EM bajó). Las notas previas no cambian.
- `hmm2_p_high` (y sus copias en `details`): diferencias ≤ 3.4e-15 en 6 048 resultados, por el ajuste ahora determinista. No cambió ningún estado ni score.
- `mtf_context`: en la ejecución completa no cambia si todos los timeframes tienen datos. En ejecuciones donde un timeframe pedido aún no tiene barras, aparece su clave con valores `None`.

## Impacto en Fase 4/4.1

- `fit_hmm_regimes` es compartida con `validate_models.py` y `validate_walk_forward.py`. Con un hilo OpenMP sus ajustes son reproducibles. Antes, los parámetros podían variar en los últimos bits entre ejecuciones, y en casos límite cambiar la semilla elegida o un estado argmax. Se espera que las métricas de 4/4.1 se muevan solo a nivel numérico, pero se deben regenerar.
- "Convergió" en los informes de Fase 4/4.1 (`MODEL_DECISIONS.md`: HMM2 "convergieron en los cinco activos") usa el `monitor_.converged` permisivo de hmmlearn. Con `fit_quality.converged` algunos de esos ajustes pueden quedar marcados. La decisión de modelos no se cambia aquí.
- Sigue pendiente la revalidación de 4.1 por el cambio de Markov2 (Fase 7). Esta corrección se suma a esa misma re-ejecución.

## Tests añadidos

`tests/test_prefix_regression.py`. Los seis fallan con el código de main (el último, por la falta de reproducibilidad de HMM2):

- `test_late_intraday_context_passes_prefix_check`: layout real (intradía tardío) con cortes antes y durante el contexto intradía.
- `test_mtf_context_schema_does_not_depend_on_future_data`: mismas claves en `mtf_context` y mismas columnas `h4_*`/`h1_*`.
- `test_debug_prefix_names_the_first_divergent_input`: el diagnóstico señala `h4_*` (esquema, barra 0, contexto 4Hour) en el caso anterior y 0 diferencias en el corregido.
- `test_hmm_em_non_convergence_is_flagged_and_prefix_stable`: un HMM2 con bajada del EM real (Student-t, semilla 1, re-ajuste 169) queda marcado, no llega al logger y da los mismos valores con datos hasta T.
- `test_non_converged_fit_is_visible_in_pattern_results`: `hmm_fit_converged=False` y `model_not_converged:hmm2` en los resultados afectados; prefijo idéntico.
- `test_hmm2_fit_is_bit_reproducible`: ocho ajustes del mismo entrenamiento (el re-ajuste 586 del caso NVDA-like) son idénticos bit a bit.
