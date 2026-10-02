# Fase 7 — Pattern Engine

**Estado:** implementada y validada con tests sin red y con una validación sintética offline. La validación con datos de mercado (`python validate_patterns.py`) necesita credenciales Alpaca y **no se ha ejecutado**; no hay resultados de mercado de Fase 7.

El motor describe qué patrón compuesto parece formarse, con qué evidencia a favor y en contra, en qué estado está, desde cuándo, cuándo se confirma o invalida y en qué contexto de régimen y volatilidad. **No predice ni genera señales**: no hay compra/venta, recomendación, retorno esperado, PnL, win rate ni tamaño de posición.

## Arquitectura

```text
bars (NORMALIZED, solo is_complete)            otros timeframes (solo is_complete)
   │                                                    │
   ├─ quant_context.py   HMM2, Markov2, Kalman local level, GARCH, PELT, Hurst/entropía
   │                     (re-ajuste walk-forward, salidas filtradas)
   ├─ external_context.py VIX, breadth, macro asof_safe, noticias (available_at <= cierre)
   ▼                                                    ▼
evidence.py  frame causal por barra  ◄── attach_timeframe (close_time <= cierre)
   ▼
detectores (exhaustion, breakout, mean_reversion, trend_continuation,
            regime_transition, divergence) + máquina de estados (base.py)
   ▼
PatternResult por barra y patrón  →  episode_table → phase7_results/pattern_events.csv
                                  →  describe_patterns (métricas descriptivas)
```

| Archivo | Rol |
|---|---|
| `src/patterns/config.py` | Pesos, umbrales, catálogo de evidencia, máquinas de estado. Heurísticos, no optimizados. |
| `src/patterns/base.py` | `PatternResult`, `Evidence`, `History` (vista de filas 0..i), scoring y máquina de estados. |
| `src/patterns/evidence.py` | Frame causal: features, estructura as-of, niveles as-of, breakouts 6.1, gaps filtrados, RS, multi-timeframe. |
| `src/patterns/quant_context.py` | Contexto cuantitativo causal por barra. |
| `src/patterns/external_context.py` | Contexto externo as-of por barra. |
| `src/patterns/{exhaustion,breakout,mean_reversion,trend_continuation,regime_transition,divergence}.py` | Detectores. `breakout.py` incluye también el failed breakout. |
| `src/patterns/engine.py` | `run_pattern_engine`, tabla de episodios, métricas. |
| `src/patterns/checks.py` | Comprobaciones de estados, timestamps, scores y prefijo, compartidas por tests y validador. |
| `validate_patterns.py` | Validación con Alpaca o `--synthetic` sin red. |

## Causalidad

Regla: **si se añaden barras futuras, el estado y el score de un patrón en una fecha pasada no cambian.**

- **Evaluación al cierre.** Cada barra *i* se evalúa en su `close_time`: 16:00 ET para 1Day, fin del bloque para 4Hour/1Hour y `source_last_session_close` para 1Week.
- **Detectores ciegos al futuro.** Los detectores reciben `History(rows, i)`, que solo expone las filas 0..*i*. La máquina de estados avanza barra a barra y solo guarda estado pasado.
- **Swings.** Cuentan desde `confirmation_index` (su `confirmed_at`). HH/HL/LH/LL y `structural_state` se reconstruyen por barra.
- **Niveles.** Son los niveles as-of de Fase 6.1 en la barra *i*.
- **Breakouts.** Son los eventos de `detect_breakouts` (6.1), probados contra niveles de *i−1* y fechados en la barra donde se observan. `level_asof_timestamp` < timestamp del evento.
- **Recencia.** La recencia de breakout es la de 6.1: 5 barras. La evidencia de un evento dura esas 5 barras; un episodio solo se abre en la barra del evento.
- **Modelos.** Se re-ajustan en posiciones fijas: la primera con `min_train` filas válidas y después cada `refit_every` barras. Cada ajuste usa filas ≤ *r* y filtra hasta el siguiente re-ajuste. Los valores `*_lag` salen del mismo ajuste, para no inventar saltos.
- **Change points.** PELT corre en una ventana móvil que termina en la barra: un cambio se ve con al menos `cp_min_size` barras de latencia.
- **Contexto externo.** Se une por `available_at <= close_time`. El VIX (20:00 ET) y la breadth (16:30 ET) de la misma sesión no están disponibles al cierre diario de las 16:00 y se usan desde la barra siguiente.
- **Otros timeframes.** Se unen por `close_time` y solo con barras completas. Una semana parcial nunca entra.

### Correcciones de causalidad halladas por los tests de prefijo

1. **Markov2 no era determinista.** statsmodels 0.15 ignora `np.random.seed` y usa su propio `rng`, así que `fit_markov_regimes` daba ajustes distintos en cada ejecución. Ahora pasa `rng=random_state`. Es un cambio mínimo en `src/models/regime_models.py` que también corrige la reproducibilidad de Fase 4.
2. **GARCH con filtro fijo de `arch`.** Ese filtro calcula límites de varianza con toda la muestra suministrada, una fuga pequeña (~2e-7) que cambiaba valores pasados. Se sustituyó por la recursión GARCH(1,1) explícita (`garch_filter`), con backcast solo de entrenamiento y parámetros estimados por `arch`.

## PatternResult

Hay uno por barra y por patrón. Campos principales:
- Identificación: `ticker`, `timestamp`, `timeframe`, `pattern_name`, `pattern_family`, `direction`, `subtype`.
- Estado: `state`, `first_detected_at`, `confirmed_at`, `invalidated_at`, `expired_at`, `bars_active` y `episode_id`.
- Scores: `score` y `score_structure|volatility|regime|context|momentum`.
- Evidencia: `evidence_for`, `evidence_against`, `required_conditions_met|missing` y `optional_conditions_met`.
- Contexto: `regime_context`, `volatility_context`, `structure_context`, `external_context`, `mtf_context`.
- Otros: `details` (por ejemplo, nivel y tiempos del breakout) y `notes`. `notes` incluye `not_evaluable:...` con la evidencia sin datos.

`score_kind = "convergence_score"`: **mide convergencia de evidencia; no es probabilidad de éxito, retorno esperado ni señal.**

## Máquina de estados

Estados: `inactive`, `candidate`, `developing`, `confirmed`, `invalidated`, `expired`.

- **Apertura.** No hay episodio abierto, terminó el cooldown, se cumplen **todas** las condiciones requeridas y `score ≥ enter_score`. Los patrones por evento abren solo en la barra del evento.
- **Por barra, en este orden:**
  1. `invalidated` si se cumple la regla de invalidación (también desde `confirmed`).
  2. `confirmed` si está pendiente, se cumple la regla de confirmación y `score ≥ confirm_score`.
  3. Si sigue pendiente, una barra sin mantenimiento cuenta como *lapse*. Con más de `grace_bars` lapses, o más de `max_pending_bars` sin confirmar, pasa a `expired`. Si no, `candidate` → `developing` tras `developing_after_bars`.
  4. `confirmed` cierra a los `confirmed_ttl_bars`: el estado final sigue siendo `confirmed` y `expired_at` marca cuándo dejó de estar vigente.
- **Tras el cierre** vuelve a `inactive` y aplica `cooldown_bars` antes de un episodio nuevo. La tolerancia (`grace_bars`) y el cooldown evitan que un patrón aparezca y desaparezca en cada barra.

Transiciones permitidas, verificadas en `checks.py`:
- `inactive` → `candidate`
- `candidate` → `developing` | `confirmed` | `invalidated` | `expired`
- `developing` → `confirmed` | `invalidated` | `expired`
- `confirmed` → `invalidated` o cierre
- `invalidated` | `expired` → `inactive`

| Familia | enter / confirm score | pendiente máx. | TTL confirmado |
|---|---|---|---|
| exhaustion, mean_reversion | 0.40 / 0.35 | 10 | 5 |
| trend_continuation | 0.40 / 0.35 | 15 | 5 |
| breakout | evento / evento | 3 | 10 |
| failed_breakout | evento / regla | 6 | 10 |
| regime_transition | cruce / regla | 10 | 5 |
| divergence | pivot / ruptura de neckline | 15 | 5 |

## Scoring

Cada evidencia tiene nombre, componente, peso (1.0 salvo lo indicado en `EVIDENCE_WEIGHTS`) y valor `True`, `False` o `None` (no evaluable).

- **Por componente** *c*: `score_c = S_c / (E_c + A_c)`.
  - `S_c`: peso de la evidencia a favor presente.
  - `E_c`: peso de la evidencia a favor evaluable.
  - `A_c`: peso de la evidencia en contra presente.
  - Sin evidencia evaluable, `score_c = None`. Se excluye; **no vale 0**.
- **Total:** `pattern_score = Σ W_c·score_c / Σ W_c`, solo sobre componentes disponibles. W: estructura 0.30, momentum 0.25, volatilidad 0.20, régimen 0.15, contexto 0.10.
- **Mean reversion:** se limita a 0.40 cuando estructura y régimen favorecen la persistencia, y además no puede confirmarse.

## Catálogo

En la tabla, «a favor» incluye las condiciones requeridas.

| Patrón | Requerido | A favor (opcional) | En contra | Confirma | Invalida |
|---|---|---|---|---|---|
| `bearish_/bullish_exhaustion` (agotamiento de subida / bajada) | z-score ≥ 1.5 en la dirección del movimiento; aceleración en las últimas 10 barras; desaceleración (`slope_change`) | nivel opuesto ≤ 1 ATR, deterioro de slope, RS debilitándose, flag de agotamiento de 6, expansión de volatilidad, volumen relativo ≥ 1.5, GARCH en subida, change point, régimen de alta volatilidad en subida, VIX extremo, breadth divergente, gap (0.25) | tendencia del timeframe superior intacta, RS aún fuerte, breakout activo a favor del movimiento, régimen de baja volatilidad estable, Hurst persistente (0.5) | cierre de vuelta a la media (z ≤ 0) | cierre más allá del extremo + 0.5 ATR |
| `bullish_/bearish_breakout` | evento `breakout_candidate` de 6.1; nivel conocido antes del evento | nivel con ≥ 3 toques, compresión previa, estructura y timeframe superior alineados, expansión, volumen, RS, slope, régimen estable, VIX y breadth, gap (0.25) | volumen bajo, RS en contra, desaceleración, cierre cerca del nivel, timeframe superior en contra, pico de VIX | evento `breakout_confirmed` del mismo nivel | evento fake o cierre de vuelta dentro |
| Subtipos de breakout | `breakout_under_confirmation`, `breakout_with_expansion`, `weak_breakout`, `failed_breakout_risk` | | | | |
| `bearish_/bullish_failed_breakout` (fallo de ruptura alcista / bajista) | evento `fake_breakout_*` de 6.1 | vuelta al rango ≥ 0.25 ATR, nivel fuerte, momentum, RS, volumen, expansión, régimen, VIX | recuperación del nivel, RS fuerte, timeframe superior alineado con la ruptura | 2 cierres seguidos dentro del rango | re-ruptura > 0.25 ATR |
| `bullish_/bearish_mean_reversion` | \|z\| ≥ 2; evidencia de agotamiento (desaceleración, flag o nivel opuesto) | nivel, desaceleración, RV en percentil ≥ 0.8, Hurst < 0.45 (0.5), entropía alta (0.5), breadth o VIX extremos | estructura persistente (1.5), régimen persistente, timeframe superior alineado, aceleración en curso | \|z\| ≤ 1 | estiramiento +1 z o 1 ATR más allá del extremo |
| `bullish_/bearish_trend_continuation` | HH+HL / LH+LL; slope alineado; pullback entre swings con z ≤ 0.5 | compresión, timeframe superior, Kalman (cambio del nivel local), RS, volatilidad compatible, régimen estable, breadth, VIX | RS en contra, expansión, régimen cambiando, timeframe superior en contra, LH/HL de aviso | cierre más allá del swing conocido al detectar | cierre más allá del swing opuesto o ruptura de estructura |
| `to_high_vol_/to_low_vol_regime_transition` | HMM2 y Markov2 validados en `model_status`; cruce de p = 0.5 frente a 5 barras antes | acuerdo de ambos modelos, change point en RV20 o slope, RV20 y GARCH, cambio de estructura, giro de Kalman, VIX | modelos en desacuerdo, RV20 en contra | ambos modelos ≥ 0.6 durante 3 barras | ambos < 0.4 |
| `bearish_/bullish_divergence` | nuevo pivot HH / LL confirmado; al menos un indicador divergente | divergencias de momentum, slope, RS, volumen y breadth; nivel, desaceleración, expansión | tendencia fuerte con timeframe superior, breakout activo, RS fuerte | **ruptura de neckline** (evidencia no divergente) | nuevo extremo + 0.25 ATR |

Hurst y entropía son solo opcionales y nunca abren un patrón por sí solas.

## Contexto externo

Todo es opcional:
- Si falta una familia, sus valores quedan NaN, la evidencia es «no evaluable» y el componente se excluye.
- **Macro:** solo filas `asof_safe=True`. Si no las hay, `macro_status = excluded_not_asof_safe`.
- **Noticias:** se usan con al menos 3 titulares puntuados en 7 días (`news_quality = ok`). Con menos, el sentimiento queda NaN: ni favorece ni penaliza.
- **Liquidez del activo:** el volumen relativo sale de la propia barra (Fase 6).

## Gaps

Siguen la decisión de Fase 6:
- Nunca son requeridos.
- Pesan 0.25.
- Solo cuentan los gaps con tamaño ≥ 0.5 ATR **y** ≥ 0.5%.
- Solo se usan en 1Day.

`run.meta` registra `gaps_detected`, `gaps_kept` y `gaps_discarded`. En la validación sintética se descartaron alrededor del 62%.

## Multi-timeframe

1Week es el contexto superior de 1Day; 1Day lo es de 4Hour, y 4Hour de 1Hour.

- Solo el estado estructural del timeframe superior entra al score, como una evidencia de estructura.
- Los demás timeframes aparecen en `mtf_context` de forma descriptiva: estructura, slope, z-score, aceleración, breakout y expansión de la última barra cerrada.
- No hay un score multi-timeframe único.
- 4Hour usa solo bloques completos (09:30–13:30 ET): una barra por sesión.

## Persistencia y métricas

`phase7_results/pattern_events.csv` tiene una fila por episodio. Incluye:
- Identificación: ticker, timeframe, patrón, familia, dirección y subtipo.
- Estado: estado final, timestamps de estado, `bars_active`, `is_open` y `reached_developing`.
- Scores: final, máximo, en la detección y por componente.
- Contexto: régimen, volatilidad y `level_asof_timestamp`.
- Evidencia clave a favor y en contra (en la detección y al final) y `details` en JSON.

Los episodios con `is_open=True` siguen vivos al final de la muestra y su estado final es provisional. La Fase 8 debe tratarlos así.

`describe_patterns` reporta:
- episodios por ticker y patrón, y frecuencia anual;
- duración media;
- % que llega a developing, % confirmado (también condicionado a developing), % invalidado y % expirado;
- distribución de `score_max`;
- solapamiento entre patrones: jaccard por barras activas, con redundancia potencial si jaccard ≥ 0.5 o ≥ 80% de cobertura;
- número de patrones simultáneos.

## Validación

- **Tests:** `test_patterns.py`, `test_pattern_state_machine.py` y `test_pattern_causality.py`, sin red.
- **`validate_patterns.py`** (ARM, NVDA, AMD, AVGO y QQQ en 1Day y 4Hour, con 1Week y 1Hour como contexto) comprueba:
  - estados y transiciones;
  - timestamps;
  - scores en [0,1];
  - evidencia a favor y en contra disjunta;
  - `available_at` del contexto;
  - niveles as-of;
  - recencia de breakout;
  - nombres prohibidos (probability, signal, buy…);
  - **prefijo:** en tres cortes T (50%, 75% y 90% de la muestra), todos los insumos (barras, benchmark, otros timeframes y contexto externo) se truncan a lo conocido en T, y los resultados ≤ T deben ser idénticos a los de la ejecución completa. Si alguno cambia, la validación falla.
- **`--synthetic`** ejecuta lo mismo sin red sobre 4 series sintéticas. Sus métricas describen datos sintéticos, no mercados.

## Limitaciones

- Pesos y umbrales son heurísticos y no se han calibrado. Los porcentajes de confirmación no son tasas de acierto.
- Sin validación con datos reales todavía. IEX afecta a volumen, gaps y pivots (Fase 6).
- Con menos de `min_train` barras no hay contexto de régimen (`unavailable`). En 4Hour, con una barra por sesión, el régimen se ajusta con 150 barras.
- Breadth con sesgo de supervivencia. Noticias solo de los últimos 7 días (sin histórico). Macro sin clave FRED excluido.
- Solo un episodio por detector y dirección a la vez: otros eventos de breakout durante un episodio abierto se ignoran.
- La latencia es estructural: pivots (2 barras), change points (≥ 10 barras) y confirmaciones.
- Coste: unos 25–30 s por serie diaria de 756 barras con modelos.
