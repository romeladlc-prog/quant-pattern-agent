# Fase 6 — Market Structure Engine

La capa `src/structure/` consume barras OHLCV **NORMALIZED**. No modifica Quant Model Layer ni External Context Engine. `MarketStructureSnapshot` reúne descriptores por activo y timeframe; 1Day es el periodo principal para swing, 1Week el contexto superior, 4Hour la estructura intermedia y 1Hour la confirmación. No existe score combinado ni señal.

## Reglas y disponibilidad

| Componente | Regla | Cuándo se conoce |
|---|---|---|
| Swing fractal | Máximo/mínimo estrictamente mayor/menor que 3 barras izquierdas y 2 derechas | `confirmed_at`, dos barras después de `timestamp` original |
| Swing alternativo | Reversión desde extremo de al menos 1.5 × ATR(14), o porcentaje configurable | Barra que confirma la reversión; latencia variable |
| Estructura | HH/LH compara swing highs consecutivos; HL/LL compara swing lows; diferencia ≤0.5%=EQ | Tras confirmar cada pivot. Uptrend=HH+HL; downtrend=LH+LL; EQ o LH+HL=range; HH+LL=transition; sin comparaciones suficientes=insufficient_swings |
| Niveles | Agrupa swings confirmados a tolerancia `max(0.5×ATR, 0.5% precio)` y exige dos toques | As-of cada barra (Fase 6.1): solo swings confirmados hasta esa barra y ATR/cierre/volumen de ese momento; `strength_score` combina toques, recencia y volumen sin implicar utilidad predictiva |
| Breakout | Cierre cruza un nivel as-of de la barra anterior y supera 0.1 ATR: candidate. Dos cierres fuera y volumen relativo ≥1: confirmed | Cada evento se fecha en su barra; retorno dentro de cinco barras: fake breakout, con amplitud y tiempo hasta fallo. El snapshot solo lo considera activo durante `breakout_max_age_bars` |
| Compresión/expansión | Al menos tres de ATR relativo, Bollinger bandwidth, RV y rango medio en percentil 20/80 de las 60 barras **anteriores** | Al cierre de la barra. Se informa duración consecutiva |
| Aceleración | Slope log 20 barras, cambio de slope a cinco barras, z-score, volumen relativo y expansión ATR | Al cierre. Exhaustion exige al menos tres métricas y retorno acumulado extendido; no depende de RSI |
| Gap | Apertura diaria regular al menos 0.2% frente al cierre previo; fill cuando el rango vuelve a cubrir el cierre previo | Gap en apertura; fill como evento separado en la barra donde ocurre |
| Fuerza relativa | Retornos diferenciales 5/20/60, ratio, slope y z-score contra QQQ u otro benchmark | Solo timestamps exactos y datos ya observados; no rellena huecos con datos futuros |

Las ventanas incluyen la barra actual y anteriores. El clustering final puede **omitir** niveles que existían antes y luego se fusionaron; por eso, desde la Fase 6.1, los eventos históricos usan niveles reconstruidos as-of (véase [PHASE_6_1_HARDENING.md](PHASE_6_1_HARDENING.md)). Siguen siendo descriptivos, no resultados de backtest. Los gaps se calculan solo en 1Day regular. Los bloques intradía `is_complete=False` se excluyen; la segunda ventana regular de 4Hour dura 2.5 horas y no se trata como una barra completa de cuatro horas.

## Validación

`python -m unittest discover -s tests -v` prueba series sintéticas deterministas. `python validate_structure.py` descarga barras con la Data Layer y el feed de `ALPACA_DATA_FEED` (la corrida de Fase 6 usó IEX), revisa timestamps, causalidad de pivots y gaps, estructura y cobertura en cinco activos × tres periodos, además de ARM 1Hour. IEX cubre solo operaciones de esa bolsa y puede diferir del SIP agregado. Las salidas `phase6_validation.log` y `phase6_results/summary.md` documentan cobertura, ejemplos y límites.

Fase 7 podrá evaluar secuencias de pivots, interacción con niveles, retests y compresión seguida de expansión; esta fase solo aporta eventos y variables descriptivas.
