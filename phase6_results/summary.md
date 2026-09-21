# Fase 6: Market Structure Engine

Validación: 15 combinaciones de 5 activos × 3 timeframes; ARM 1Hour adicional.
Fuente: Alpaca IEX OHLCV NORMALIZED, sesión regular; IEX cubre solo operaciones IEX, no el mercado SIP completo. Los bloques intradía con `is_complete=False` se excluyen.

## Reglas y parámetros

- Swings: fractal estricto 3 barras izquierda, 2 derecha; latencia exacta de 2 barras. Alternativa: reversión ATR(14) × 1.5, latencia variable.
- HH/HL/LH/LL: comparación con pivot confirmado anterior del mismo tipo, igualdad dentro de 0.5%. Uptrend=HH+HL, downtrend=LH+LL; EQ o LH+HL=range; HH+LL=transition.
- Niveles: clustering de pivots confirmados a max(0.5 ATR, 0.5% precio), mínimo 2 toques. Strength combina toques, recencia y volumen.
- Breakouts: cruce del cierre, distancia ≥0.1 ATR, volumen relativo ≥1 para confirmación de 2 cierres; fallo si regresa dentro de 5 barras.
- Compresión/expansión: ≥3 de 4 métricas en percentil 20/80 de 60 barras previas; duración contigua.
- Aceleración: slope log 20 barras, cambio de 5 barras. Exhaustion requiere 3 condiciones simultáneas y retorno extendido.
- Gap: apertura frente al cierre previo, mínimo 0.2%; fill por rango desde la barra actual, con evento fechado cuando ocurre.
- Fuerza relativa: retorno 5/20/60, ratio, slope y z-score contra QQQ en timestamps exactos.

## Cobertura y ejemplos

- ARM 4Hour: 2026-04-23 13:30:00+00:00 a 2026-09-18 13:30:00+00:00; 101 barras; estado uptrend; 25 swings; 7 niveles; 0 rupturas confirmadas; 0 fallidas; 0 gaps; 0 fills.
- NVDA 4Hour: 2026-04-23 13:30:00+00:00 a 2026-09-18 13:30:00+00:00; 103 barras; estado transition; 22 swings; 6 niveles; 1 rupturas confirmadas; 3 fallidas; 0 gaps; 0 fills.
- AMD 4Hour: 2026-04-23 13:30:00+00:00 a 2026-09-18 13:30:00+00:00; 103 barras; estado uptrend; 24 swings; 7 niveles; 4 rupturas confirmadas; 5 fallidas; 0 gaps; 0 fills.
- AVGO 4Hour: 2026-04-23 13:30:00+00:00 a 2026-09-18 13:30:00+00:00; 103 barras; estado downtrend; 22 swings; 6 niveles; 6 rupturas confirmadas; 4 fallidas; 0 gaps; 0 fills.
- QQQ 4Hour: 2026-04-23 13:30:00+00:00 a 2026-09-18 13:30:00+00:00; 103 barras; estado range; 19 swings; 5 niveles; 1 rupturas confirmadas; 1 fallidas; 0 gaps; 0 fills.
- ARM 1Day: 2023-09-14 04:00:00+00:00 a 2026-09-18 04:00:00+00:00; 756 barras; estado uptrend; 172 swings; 17 niveles; 2 rupturas confirmadas; 1 fallidas; 690 gaps; 667 fills.
- NVDA 1Day: 2023-09-14 04:00:00+00:00 a 2026-09-18 04:00:00+00:00; 756 barras; estado transition; 176 swings; 29 niveles; 20 rupturas confirmadas; 12 fallidas; 650 gaps; 620 fills.
- AMD 1Day: 2023-09-14 04:00:00+00:00 a 2026-09-18 04:00:00+00:00; 756 barras; estado uptrend; 170 swings; 17 niveles; 8 rupturas confirmadas; 2 fallidas; 661 gaps; 630 fills.
- AVGO 1Day: 2023-09-14 04:00:00+00:00 a 2026-09-18 04:00:00+00:00; 756 barras; estado downtrend; 165 swings; 34 niveles; 32 rupturas confirmadas; 25 fallidas; 643 gaps; 624 fills.
- QQQ 1Day: 2023-09-14 04:00:00+00:00 a 2026-09-18 04:00:00+00:00; 756 barras; estado range; 164 swings; 35 niveles; 39 rupturas confirmadas; 36 fallidas; 548 gaps; 518 fills.
- ARM 1Week: 2023-09-14 04:00:00+00:00 a 2026-09-14 04:00:00+00:00; 158 barras; estado uptrend; 28 swings; 4 niveles; 1 rupturas confirmadas; 0 fallidas; 0 gaps; 0 fills.
- NVDA 1Week: 2023-09-14 04:00:00+00:00 a 2026-09-14 04:00:00+00:00; 158 barras; estado range; 35 swings; 10 niveles; 9 rupturas confirmadas; 9 fallidas; 0 gaps; 0 fills.
- AMD 1Week: 2023-09-14 04:00:00+00:00 a 2026-09-14 04:00:00+00:00; 158 barras; estado uptrend; 33 swings; 4 niveles; 0 rupturas confirmadas; 0 fallidas; 0 gaps; 0 fills.
- AVGO 1Week: 2023-09-14 04:00:00+00:00 a 2026-09-14 04:00:00+00:00; 158 barras; estado range; 36 swings; 9 niveles; 9 rupturas confirmadas; 10 fallidas; 0 gaps; 0 fills.
- QQQ 1Week: 2023-09-14 04:00:00+00:00 a 2026-09-14 04:00:00+00:00; 158 barras; estado downtrend; 26 swings; 6 niveles; 5 rupturas confirmadas; 4 fallidas; 0 gaps; 0 fills.

## Latencia, falsos positivos y limitaciones

- Pivot fractal confirmado dos barras después; el timestamp original se conserva, pero solo `confirmed_at` indica disponibilidad. ATR reversal espera una reversión, con latencia variable.
- Los breakouts se evalúan alrededor de niveles construidos al cierre de la muestra; para backtesting histórico hay que reconstruir los niveles en cada prefijo. La validación presente no estima rendimiento predictivo.
- Un cruce intrabar y regreso inmediato puede generar candidato y fake breakout el mismo día; son posibles falsos positivos descriptivos. Los niveles agrupados son sensibles a la tolerancia ATR.
- En IEX, 548–690 de 756 barras diarias aparecen como gaps por activo: cobertura parcial del mercado y umbral 0.2% producen demasiados candidatos. No pasar el descriptor gap a Fase 7 sin feed consolidado o calibración adicional.
- El segundo bloque regular de 4Hour dura 2.5 horas y la Data Layer lo marca incompleto; se excluye. Gaps se analizan solo en 1Day regular.
- ATR relativo, rango medio y realized volatility pueden ser redundantes en compresión; slope y cumulative return también están relacionados.
- Para Fase 7: secuencias de pivots confirmados, interacción con niveles, retest, compresión seguida de expansión y gap/fill como eventos descriptivos.
- Sin señal, score de patrón, recomendación ni evaluación de rentabilidad.
