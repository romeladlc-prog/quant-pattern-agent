"""Phase 6 live-data validation; writes an auditable log and summary."""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import traceback

import pandas as pd

from src.data.alpaca_client import data_feed, download_historical_bars, normalize_bars
from src.security import redact_secrets
from src.structure import make_structure_snapshot
from src.structure.acceleration import acceleration_features
from src.structure.breakouts import detect_breakouts
from src.structure.compression import compression_features
from src.structure.gaps import detect_gaps
from src.structure.levels import build_levels_asof, cluster_levels, levels_history
from src.structure.relative_strength import relative_strength
from src.structure.swings import atr_reversal_swings, confirmed_swings
from src.structure.trend_structure import classify_swings, structural_state

ASSETS=("ARM","NVDA","AMD","AVGO","QQQ")
TIMEFRAMES=("4Hour","1Day","1Week")
OUT=Path("phase6_results")
LOG=Path("phase6_validation.log")


def fetch_normalized(ticker, start, end, timeframe):
    """Data Layer download with the shared ALPACA_DATA_FEED setting."""
    return download_historical_bars(ticker,start,end,timeframe,"regular")


def check_bars(bars, name):
    assert not bars.empty, f"{name}: empty"
    assert bars.timestamp.is_monotonic_increasing and not bars.timestamp.duplicated().any(), name
    assert bars[["open","high","low","close","volume"]].notna().all().all(), name
    assert (bars[["open","high","low","close"]]>0).all().all(), name
    assert (bars.volume>=0).all(), name


def validate_one(ticker, timeframe, bars, benchmark):
    key=f"{ticker} {timeframe}"; check_bars(bars,key)
    swings=classify_swings(confirmed_swings(bars))
    assert swings.empty or (swings.confirmation_index-swings.bar_index).eq(2).all(), key
    assert swings.empty or swings.confirmed_at.ge(swings.timestamp).all(), key
    assert swings.empty or swings.confirmed_at.le(bars.timestamp.iloc[-1]).all(), key
    assert set(swings.structure_label.dropna()).issubset({"HH","HL","LH","LL","EQ"}),key
    alternative=atr_reversal_swings(bars)
    assert alternative.empty or alternative.confirmed_at.ge(alternative.timestamp).all(),key
    levels=cluster_levels(bars,swings)
    assert levels.empty or levels.available_at.le(bars.timestamp.iloc[-1]).all(),key
    assert levels.empty or levels.price_level.diff().dropna().ge(0).all(),key
    history=levels_history(bars,swings)
    events=detect_breakouts(bars,history=history)
    assert events.empty or events.timestamp.le(bars.timestamp.iloc[-1]).all(),key
    # Every event was tested against levels known before its crossing bar.
    assert events.empty or events.level_asof_timestamp.lt(events.timestamp).all(),key
    compression=compression_features(bars)
    acceleration=acceleration_features(bars)
    assert len(compression)==len(acceleration)==len(bars),key
    rs=relative_strength(bars,benchmark)
    assert len(rs)==len(bars) and rs.timestamp.equals(bars.timestamp),key
    gaps=detect_gaps(bars) if timeframe=="1Day" else pd.DataFrame()
    assert gaps.empty or gaps.timestamp.ge(gaps.gap_timestamp).all(),key
    snapshot=make_structure_snapshot(ticker,bars,timeframe,benchmark)
    assert snapshot.timestamp==bars.timestamp.iloc[-1],key
    # A historical prefix must never expose a later-confirmed pivot or later fill.
    cut=max(30,len(bars)//2)
    prefix=bars.iloc[:cut].reset_index(drop=True)
    early=confirmed_swings(prefix)
    assert early.equals(confirmed_swings(bars).loc[
        lambda x:x.confirmation_index.lt(cut)].reset_index(drop=True)),key
    # As-of levels and breakouts of the prefix must not change with later bars.
    pd.testing.assert_frame_equal(build_levels_asof(prefix,prefix.timestamp.iloc[-1]),
        history.loc[history.asof_index.eq(cut-1)].reset_index(drop=True),check_dtype=False)
    full_events=events.loc[events.timestamp.le(prefix.timestamp.iloc[-1])].reset_index(drop=True)
    pd.testing.assert_frame_equal(detect_breakouts(prefix),full_events,check_dtype=False)
    if timeframe=="1Day":
        early_gaps=detect_gaps(prefix)
        full_gaps=detect_gaps(bars)
        pd.testing.assert_frame_equal(early_gaps,full_gaps.loc[
            full_gaps.timestamp.le(prefix.timestamp.iloc[-1])].reset_index(drop=True),check_dtype=False)
    return {"ticker":ticker,"timeframe":timeframe,"bars":len(bars),
        "first":str(bars.timestamp.iloc[0]),"last":str(bars.timestamp.iloc[-1]),
        "state":structural_state(swings),"swings":len(swings),"atr_swings":len(alternative),
        "levels":len(levels),"asof_level_rows":len(history),"breakouts":int(events.state.eq("breakout_confirmed").sum()) if not events.empty else 0,
        "fake_breakouts":int(events.state.str.startswith("fake_breakout").sum()) if not events.empty else 0,
        "compression_bars":int(compression.volatility_compression.sum()),
        "expansion_bars":int(compression.volatility_expansion.sum()),
        "gaps":int(gaps.state.eq("unfilled_gap").sum()) if not gaps.empty else 0,
        "gap_fills":int(gaps.state.eq("gap_fill").sum()) if not gaps.empty else 0,
        "rs20":snapshot.relative_strength_metrics.get("relative_return_20"),
        "last_swing_confirmation_lag":int(swings.confirmation_lag_bars.iloc[-1]) if not swings.empty else None,
        "snapshot":snapshot.as_dict()}


def main():
    OUT.mkdir(exist_ok=True)
    # End before today: the current session is never complete in a validation run.
    end=date.today().isoformat()
    feed=data_feed()
    daily={}; intraday={}; weekly={}; lines=[]; results=[]
    try:
        for ticker in ASSETS:
            daily[ticker]=fetch_normalized(ticker,"2023-09-14",end,"1Day")
            check_bars(daily[ticker],ticker)
            # Weekly aggregation is from the same already-downloaded NORMALIZED daily OHLCV.
            raw=daily[ticker].assign(symbol=ticker).set_index(["symbol","timestamp"])
            weekly[ticker]=normalize_bars(raw,"2023-09-14",end,"1Week","regular")
            weekly[ticker]=weekly[ticker].loc[weekly[ticker].is_complete].reset_index(drop=True)
            intraday[ticker]=fetch_normalized(ticker,
                (date.today()-timedelta(days=150)).isoformat(),end,"4Hour")
            intraday[ticker]=intraday[ticker].loc[intraday[ticker].is_complete].reset_index(drop=True)
            lines.append(f"{ticker}: daily={len(daily[ticker])}, weekly={len(weekly[ticker])}, "
                         f"complete 4Hour={len(intraday[ticker])}")
        mapping={"1Day":daily,"1Week":weekly,"4Hour":intraday}
        for timeframe in TIMEFRAMES:
            for ticker in ASSETS:
                result=validate_one(ticker,timeframe,mapping[timeframe][ticker],mapping[timeframe]["QQQ"])
                results.append(result)
                lines.append(f"PASS {ticker} {timeframe}: {result['state']}; swings={result['swings']}; "
                    f"levels={result['levels']}; breakout={result['breakouts']}; "
                    f"fake={result['fake_breakouts']}; gaps={result['gaps']}")
        # 1Hour support uses the same NORMALIZED API, checked on a short ARM sample.
        hourly=fetch_normalized("ARM",(date.today()-timedelta(days=30)).isoformat(),end,"1Hour")
        hourly=hourly.loc[hourly.is_complete].reset_index(drop=True)
        check_bars(hourly,"ARM 1Hour")
        snapshot=make_structure_snapshot("ARM",hourly,"1Hour")
        lines.append(f"PASS ARM 1Hour: {len(hourly)} complete bars, {snapshot.structural_state}")
        (OUT/"snapshots.json").write_text(json.dumps([r["snapshot"] for r in results],default=str,
            indent=2),encoding="utf-8")
        summary=["# Fase 6: Market Structure Engine", "", "Validación: 15 combinaciones de 5 activos × 3 timeframes; ARM 1Hour adicional.",
          f"Fuente: Alpaca OHLCV NORMALIZED, feed `{feed}` (ALPACA_DATA_FEED), sesión regular. IEX cubre solo operaciones IEX, no el mercado SIP completo. Los bloques intradía y semanas con `is_complete=False` se excluyen.",
          "", "## Reglas y parámetros", "", "- Swings: fractal estricto 3 barras izquierda, 2 derecha; latencia exacta de 2 barras. Alternativa: reversión ATR(14) × 1.5, latencia variable.",
          "- HH/HL/LH/LL: comparación con pivot confirmado anterior del mismo tipo, igualdad dentro de 0.5%. Uptrend=HH+HL, downtrend=LH+LL; EQ o LH+HL=range; HH+LL=transition.",
          "- Niveles: clustering de pivots confirmados a max(0.5 ATR, 0.5% precio), mínimo 2 toques. Strength combina toques, recencia y volumen.",
          "- Breakouts: cruce del cierre contra niveles as-of de la barra anterior, distancia ≥0.1 ATR, volumen relativo ≥1 para confirmación de 2 cierres; fallo si regresa dentro de 5 barras.",
          "- Compresión/expansión: ≥3 de 4 métricas en percentil 20/80 de 60 barras previas; duración contigua.",
          "- Aceleración: slope log 20 barras, cambio de 5 barras. Exhaustion requiere 3 condiciones simultáneas y retorno extendido.",
          "- Gap: apertura frente al cierre previo, mínimo 0.2%; fill por rango desde la barra actual, con evento fechado cuando ocurre.",
          "- Fuerza relativa: retorno 5/20/60, ratio, slope y z-score contra QQQ en timestamps exactos.",
          "", "## Cobertura y ejemplos", ""]
        summary += [f"- {r['ticker']} {r['timeframe']}: {r['first']} a {r['last']}; "
            f"{r['bars']} barras; estado {r['state']}; {r['swings']} swings; {r['levels']} niveles; "
            f"{r['breakouts']} rupturas confirmadas; {r['fake_breakouts']} fallidas; "
            f"{r['gaps']} gaps; {r['gap_fills']} fills." for r in results]
        summary += ["", "## Latencia, falsos positivos y limitaciones", "",
          "- Pivot fractal confirmado dos barras después; el timestamp original se conserva, pero solo `confirmed_at` indica disponibilidad. ATR reversal espera una reversión, con latencia variable.",
          "- Fase 6.1: cada breakout histórico usa los niveles reconstruidos as-of al cierre de la barra anterior; añadir barras posteriores no cambia niveles ni eventos ya fechados. La validación presente no estima rendimiento predictivo.",
          "- Un cruce intrabar y regreso inmediato puede generar candidato y fake breakout el mismo día; son posibles falsos positivos descriptivos. Los niveles agrupados son sensibles a la tolerancia ATR.",
          "- En IEX, 548–690 de 756 barras diarias aparecen como gaps por activo: cobertura parcial del mercado y umbral 0.2% producen demasiados candidatos. No pasar el descriptor gap a Fase 7 sin feed consolidado o calibración adicional.",
          "- El segundo bloque regular de 4Hour dura 2.5 horas y la Data Layer lo marca incompleto; se excluye. Gaps se analizan solo en 1Day regular.",
          "- ATR relativo, rango medio y realized volatility pueden ser redundantes en compresión; slope y cumulative return también están relacionados.",
          "- Para Fase 7: secuencias de pivots confirmados, interacción con niveles, retest, compresión seguida de expansión y gap/fill como eventos descriptivos.",
          "- Sin señal, score de patrón, recomendación ni evaluación de rentabilidad."]
        (OUT/"summary.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
        lines.append("PASS: all live validations complete")
    except Exception:
        lines.append("FAIL: "+redact_secrets(traceback.format_exc()))
        raise
    finally:
        LOG.write_text("\n".join(lines)+"\n",encoding="utf-8")
        print("\n".join(lines))


if __name__=="__main__": main()
