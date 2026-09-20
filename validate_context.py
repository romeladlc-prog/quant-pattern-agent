"""Phase 5 source, coverage, unit and point-in-time validation."""
from __future__ import annotations

from datetime import date, timedelta
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.context.breadth import BREADTH_BASKET, calculate_breadth
from src.context.liquidity import FRED_SERIES, asset_liquidity, fetch_fred_series, macro_asof
from src.context.market_context import make_snapshot
from src.context.sentiment import fetch_alpaca_news, fetch_vix
from src.data.alpaca_client import download_historical_bars

ASSETS = ("ARM", "NVDA", "AMD", "AVGO", "QQQ")
OUT = Path("phase5_results")


def check_bars(ticker, frame):
    if frame.empty or frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise AssertionError(f"{ticker}: missing, duplicate or unsorted bars")
    if frame[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise AssertionError(f"{ticker}: missing OHLCV")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
        raise AssertionError(f"{ticker}: invalid OHLCV units")


def main():
    load_dotenv()
    OUT.mkdir(exist_ok=True)
    as_of = pd.Timestamp.now(tz="UTC")
    start, end = "2023-09-14", date.today().isoformat()
    bars = {}
    issues = []
    for ticker in sorted(set(ASSETS) | set(BREADTH_BASKET)):
        try:
            frame = download_historical_bars(ticker, start, end, "1Day", "regular")
            check_bars(ticker, frame)
            bars[ticker] = frame
            print(f"Alpaca NORMALIZED 1Day {ticker}: {len(frame)} rows, "
                  f"{frame.timestamp.iloc[0]} .. {frame.timestamp.iloc[-1]}")
        except Exception as exc:
            issues.append(f"Alpaca bars {ticker}: {exc}")
    if not set(ASSETS).issubset(bars) or not set(BREADTH_BASKET).issubset(bars):
        raise RuntimeError("Cannot validate all assets and the declared breadth basket: " + "; ".join(issues))
    liquid = {ticker: asset_liquidity(bars[ticker]) for ticker in ASSETS}
    for ticker, frame in liquid.items():
        if frame.index.duplicated().any() or not frame.available_at.gt(frame.index).all():
            raise AssertionError(f"{ticker}: asset liquidity timestamp violation")
        if (frame[["dollar_volume", "amihud_illiquidity"]].dropna() < 0).any().any():
            raise AssertionError(f"{ticker}: asset liquidity unit violation")
        frame.to_csv(OUT / f"asset_liquidity_{ticker}.csv")
    breadth = calculate_breadth({key: bars[key] for key in BREADTH_BASKET})
    if breadth.index.duplicated().any() or not breadth.available_at.gt(breadth.index).all():
        raise AssertionError("Breadth timestamp violation")
    for name in ("breadth_sma20", "breadth_sma50", "breadth_sma200"):
        if not breadth[name].dropna().between(0, 1).all():
            raise AssertionError(f"{name} outside 0..1")
    breadth.to_csv(OUT / "breadth.csv")
    print(f"Breadth basket: {','.join(BREADTH_BASKET)}; full SMA200 coverage "
          f"{breadth.breadth_sma200.notna().mean():.1%}")

    try:
        vix = fetch_vix()
        if vix.observation_date.duplicated().any() or not vix.available_at.gt(vix.observation_date).all():
            raise AssertionError("VIX timestamp violation")
        vix.to_csv(OUT / "vix.csv", index=False)
        print(f"Cboe VIX: {len(vix)} rows, {vix.observation_date.iloc[0]} .. "
              f"{vix.observation_date.iloc[-1]}")
    except Exception as exc:
        issues.append(f"Cboe VIX: {exc}")
        vix = pd.DataFrame()

    macro_parts = []
    for name, spec in FRED_SERIES.items():
        try:
            series = fetch_fred_series(name, start)
            if series.empty or series.observation_date.duplicated().any() and not series.asof_safe.all():
                raise AssertionError("empty or duplicate observations")
            if not series.available_at.ge(series.observation_date).all():
                raise AssertionError("macro available before observation")
            macro_parts.append(series)
            print(f"FRED {spec.series_id} ({spec.unit}): {len(series)} rows; "
                  f"asof_safe={series.asof_safe.all()}; "
                  f"last observation={series.observation_date.max()}")
        except Exception as exc:
            issues.append(f"FRED {spec.series_id}: {exc}")
    macro = pd.concat(macro_parts, ignore_index=True) if macro_parts else pd.DataFrame()
    if not macro.empty:
        macro.to_csv(OUT / "macro_observations.csv", index=False)
        historical = pd.Timestamp("2025-01-02", tz="UTC")
        if not os.getenv("FRED_API_KEY") and not macro_asof(macro, historical).empty:
            raise AssertionError("Current-revision FRED data leaked into historical as-of")

    news_parts = []
    news_start = (as_of - pd.Timedelta(days=7)).to_pydatetime()
    for ticker in ASSETS:
        try:
            frame = fetch_alpaca_news(ticker, news_start, as_of.to_pydatetime())
            if not frame.empty:
                if (frame.published_at.gt(pd.Timestamp.now(tz="UTC")).any()
                        or frame.available_at.gt(pd.Timestamp.now(tz="UTC")).any()
                        or not frame.available_at.ge(frame.published_at).all()
                        or not frame.score.dropna().between(-1, 1).all()):
                    raise AssertionError("news timestamp or score violation")
                if frame.duplicated(["published_at", "headline"]).any():
                    issues.append(f"Alpaca news {ticker}: duplicate headline/timestamp")
            news_parts.append(frame)
            print(f"Alpaca news {ticker}: {len(frame)} 7d articles, "
                  f"{frame.score.notna().sum()} scored headlines")
            if len(frame) and frame.score.notna().sum() < 3:
                issues.append(f"Alpaca news {ticker}: fewer than 3 scored headlines; sentiment unstable")
        except Exception as exc:
            issues.append(f"Alpaca news {ticker}: {exc}")
    news = pd.concat(news_parts, ignore_index=True) if news_parts else pd.DataFrame()
    if not news.empty:
        news.to_csv(OUT / "news_text_and_scores.csv", index=False)

    # Public FRED rows become available only when this download finishes.
    as_of = pd.Timestamp.now(tz="UTC")
    snapshots = []
    for ticker in ASSETS:
        snapshot = make_snapshot(ticker, as_of, vix=vix, macro_observations=macro,
                                 asset=liquid[ticker], breadth=breadth, news=news)
        snapshots.append(snapshot.as_dict())
        print(f"SNAPSHOT {ticker}: vix_z={snapshot.volatility_context.get('vix_zscore_252d')}, "
              f"net_liquidity_4w={snapshot.macro_liquidity_context.get('net_liquidity_4w_change')}, "
              f"breadth_sma50={snapshot.breadth_context.get('breadth_sma50')}, "
              f"relative_volume={snapshot.asset_liquidity_context.get('relative_volume')}, "
              f"news_sentiment_7d={snapshot.news_sentiment_context.get('news_sentiment_7d')}")
    (OUT / "snapshots.json").write_text(json.dumps(snapshots, default=str, indent=2), encoding="utf-8")

    # Historical snapshot must not contain a bar/VIX/news observation published later.
    historical = pd.Timestamp("2025-01-02 00:00:00+00:00")
    past = make_snapshot("ARM", historical, vix=vix, macro_observations=macro,
                         asset=liquid["ARM"], breadth=breadth, news=news)
    if past.volatility_context and pd.Timestamp(past.volatility_context["available_at"]) > historical:
        raise AssertionError("VIX look-ahead")
    if past.asset_liquidity_context and pd.Timestamp(past.asset_liquidity_context["available_at"]) > historical:
        raise AssertionError("Asset liquidity look-ahead")
    if past.breadth_context and pd.Timestamp(past.breadth_context["available_at"]) > historical:
        raise AssertionError("Breadth look-ahead")
    if not os.getenv("FRED_API_KEY") and past.macro_liquidity_context["fed_assets"] is not None:
        raise AssertionError("FRED latest vintage leaked into historical snapshot")
    if past.news_sentiment_context.get("news_count_7d", 0) != 0:
        raise AssertionError("News look-ahead")
    print(f"Historical as-of {historical}: no future VIX, bars, macro or news exposed")

    available = {"asset_liquidity": True, "breadth": True, "vix": not vix.empty,
                 "macro_current_descriptive": not macro.empty,
                 "macro_historical_asof": bool(not macro.empty and macro.asof_safe.all()),
                 "news_7d": not news.empty, "news_scoring_7d": bool(not news.empty and news.score.notna().any()),
                 "asset_turnover": False, "vix_term_structure": False, "put_call_ratio": False}
    summary = ["# Fase 5: External Context Engine", "",
               f"Validación UTC: {as_of}; activos: {', '.join(ASSETS)}.",
               f"Barras Alpaca NORMALIZED 1Day: {start} a {end} (fin exclusivo).", "",
               "## Fuentes", "",
               "- Alpaca: OHLCV NORMALIZED y noticias con published_at.",
               "- Cboe: VIX_History.csv, cierre diario; disponible de forma conservadora tras la sesión.",
               "- FRED: WALCL, WRESBAL, WDTGAL, RRPONTSYD, SOFR, DGS2, DGS10, BAMLH0A0HYM2.",
               "- Breadth: cesta fija AAPL, MSFT, NVDA, AMZN, META, GOOGL, GOOG, AVGO, TSLA, AMD; "
               "no es el Nasdaq 100 completo y tiene sesgo de supervivencia histórico.", "",
               "## Cobertura", ""]
    summary += [f"- {key}: {'disponible' if value else 'faltante/no verificado'}" for key, value in available.items()]
    summary += ["", "## Advertencias", "",
        "FRED CSV sin FRED_API_KEY contiene revisiones actuales: release_date desconocida, "
        "asof_safe=false. Solo se usa para snapshot actual, nunca en backtest histórico. "
        "Con clave se consultan vintages ALFRED y se asigna disponibilidad al día siguiente de release_date.",
        "Net liquidity = WALCL - WDTGAL - 1000*RRPONTSYD, en millones USD; es una descripción, "
        "no una identidad económica ni una señal. Expanding/contracting depende solo del signo "
        "del cambio de cuatro semanas.",
        "Turnover no se calcula sin shares outstanding fechadas. News score usa un léxico fijo "
        "de titulares ingleses; neutral/sin palabras reconocidas queda sin score.",
        "La amplitud requiere al menos 8 de 10 componentes observables y conserva la cesta fija "
        "para reproducibilidad; cambios corporativos no están ajustados explícitamente."]
    summary += [f"- {issue}" for issue in issues] if issues else ["- Sin fallos de fuente adicionales."]
    (OUT / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Sources/coverage/limitations: {OUT / 'summary.md'}; warnings={len(issues)}")


if __name__ == "__main__":
    main()
