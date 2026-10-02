"""Phase 7 Pattern Engine validation: causality, states, persistence, descriptive metrics.

Live mode (default) needs Alpaca credentials in .env and uses ALPACA_DATA_FEED.
External context (Cboe VIX, breadth basket, FRED/ALFRED, Alpaca news) is optional:
a failed source is logged and left missing, never filled. ``--synthetic`` runs
the same checks offline on deterministic synthetic series; its numbers describe
the synthetic data only, not markets.

No returns, PnL, hit rates or signals are computed.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import traceback
import warnings

import numpy as np
import pandas as pd

from src.patterns import QuantContextConfig, describe_patterns, run_pattern_engine
from src.patterns.checks import check_results, prefix_check
from src.patterns.diagnostics import diagnose_prefix, format_report
from src.patterns.evidence import bar_close_times, complete_bars
from src.security import redact_secrets

ASSETS = ("ARM", "NVDA", "AMD", "AVGO", "QQQ")
BENCHMARK = "QQQ"
PRIMARY = ("1Day", "4Hour")
OUT = Path("phase7_results")
LOG = Path("phase7_validation.log")
QUANT = {"1Day": QuantContextConfig(), "4Hour": QuantContextConfig(min_train=150, refit_every=42)}


def truncate(frame: pd.DataFrame | None, stamp: pd.Timestamp, timeframe: str | None = None,
             column: str | None = None) -> pd.DataFrame | None:
    """Information set at bar-close ``stamp``: rows closed / available by then."""
    if frame is None or frame.empty:
        return frame
    if column is not None:
        return frame.loc[pd.to_datetime(frame[column], utc=True) <= stamp]
    base = complete_bars(frame)
    closes = bar_close_times(base, timeframe)
    return base.loc[(closes <= stamp).to_numpy()].reset_index(drop=True)


def run_one(ticker, timeframe, bars, benchmark, context, external):
    return run_pattern_engine(ticker, timeframe, bars, benchmark=benchmark, context_bars=context,
                              quant_config=QUANT.get(timeframe, QuantContextConfig()),
                              external_inputs=external)


def validate(ticker, timeframe, bars, benchmark, context, external, lines, cuts=(0.5, 0.75, 0.9),
             debug_prefix=False):
    run = run_one(ticker, timeframe, bars, benchmark, context, external)
    stats = check_results(run)
    n = len(run.evidence)
    positions = sorted({int(n*c) for c in cuts if 0 < int(n*c) < n-1})
    closes = run.evidence.close_time

    def prefix_run(stamp):
        close = closes.loc[run.evidence.timestamp.eq(stamp)].iloc[0]
        ext = None
        if external:
            ext = {k: truncate(v, close, column="available_at") for k, v in external.items()}
        ctx = {tf: truncate(frame, close, tf) for tf, frame in context.items()}
        return run_one(ticker, timeframe, truncate(bars, close, timeframe),
                       truncate(benchmark, close, timeframe) if benchmark is not None else None,
                       ctx, ext)

    if not debug_prefix:
        reports = prefix_check(run, prefix_run, positions)
    else:
        last = {}

        def remembered(stamp):
            last["run"] = prefix_run(stamp)
            return last["run"]
        try:
            reports = prefix_check(run, remembered, positions)
        except AssertionError:
            # Diagnosis only: the failure is reported and re-raised unchanged.
            lines.append(format_report(diagnose_prefix(run, last["run"])))
            raise
    lines.append(f"PASS {ticker} {timeframe}: bars={n}, results={stats['results']}, "
                 f"transitions={stats['transitions_checked']}, prefix_cuts="
                 + ",".join(str(r["T"]) for r in reports)
                 + f"; gaps kept/detected={run.meta.get('gaps_kept')}/{run.meta.get('gaps_detected')}")
    return run, stats


def synthetic_inputs(n=600):
    """Deterministic regime-switching walks labelled at New York midnight."""
    stamps = pd.bdate_range("2023-01-02", periods=n).tz_localize("America/New_York").tz_convert("UTC")
    data = {}
    for k, name in enumerate(("SYN_A", "SYN_B", "SYN_C", "SYN_D", "SYN_QQQ")):
        rng = np.random.default_rng(100+k)
        vol = np.where((np.arange(n)//120) % 2 == 0, 0.01, 0.025)
        drift = np.sin(np.arange(n)/60)*0.002
        close = 100*np.exp(np.cumsum(drift + vol*rng.standard_normal(n)))
        opens = close*np.exp(vol*0.2*rng.standard_normal(n))
        spread = np.abs(vol*rng.standard_normal(n))*close
        data[name] = pd.DataFrame({"timestamp": stamps, "open": opens,
            "high": np.maximum(opens, close)+spread, "low": np.minimum(opens, close)-spread,
            "close": close, "volume": rng.uniform(5e5, 2e6, n)})
    return data


def synthetic_intraday(daily: pd.DataFrame, start: str, seed: int, timeframe: str) -> pd.DataFrame:
    """Regular-session 15-minute walk from ``start`` aggregated like Alpaca data.

    Starts late on purpose, as live 4Hour/1Hour downloads do (400 and 90 days):
    early prefix cuts see no intraday context at all while the full run does.
    """
    from src.data.alpaca_client import normalize_bars
    days = [d for d in daily.timestamp.dt.tz_convert("America/New_York").dt.date
            if pd.Timestamp(d) >= pd.Timestamp(start)]
    stamps = pd.DatetimeIndex([s for d in days for s in pd.date_range(
        pd.Timestamp(f"{d} 09:30", tz="America/New_York"), periods=26, freq="15min")]).tz_convert("UTC")
    rng = np.random.default_rng(seed)
    close = 100*np.exp(np.cumsum(0.003*rng.standard_normal(len(stamps))))
    opens = np.r_[close[0], close[:-1]]
    spread = np.abs(0.002*rng.standard_normal(len(stamps)))*close
    raw = pd.DataFrame({"symbol": "X", "timestamp": stamps, "open": opens,
                        "high": np.maximum(opens, close)+spread, "low": np.minimum(opens, close)-spread,
                        "close": close, "volume": rng.uniform(1e4, 1e5, len(stamps))})
    end = (daily.timestamp.iloc[-1] + pd.Timedelta(days=1)).isoformat()
    return normalize_bars(raw.set_index(["symbol", "timestamp"]), start, end, timeframe, "regular")


def weekly_from_daily(daily: pd.DataFrame, end: str) -> pd.DataFrame:
    from src.data.alpaca_client import normalize_bars
    raw = daily.assign(symbol="X").set_index(["symbol", "timestamp"])
    start = daily.timestamp.iloc[0].tz_convert("America/New_York").normalize()
    return normalize_bars(raw, start, end, "1Week", "regular")


def load_live(start_daily: str, start_intraday: str, end: str, lines: list[str]):
    from src.data.alpaca_client import data_feed, download_historical_bars
    lines.append(f"feed={data_feed()}")
    daily, weekly, h4, h1 = {}, {}, {}, {}
    for ticker in ASSETS:
        daily[ticker] = download_historical_bars(ticker, start_daily, end, "1Day", "regular")
        weekly[ticker] = weekly_from_daily(daily[ticker], end)
        h4[ticker] = download_historical_bars(ticker, start_intraday, end, "4Hour", "regular")
        h1[ticker] = download_historical_bars(ticker, (date.today()-timedelta(days=90)).isoformat(),
                                              end, "1Hour", "regular")
        lines.append(f"{ticker}: 1Day={len(daily[ticker])} 1Week complete="
                     f"{int(weekly[ticker].is_complete.sum())}/{len(weekly[ticker])} "
                     f"4Hour complete={int(h4[ticker].is_complete.sum())}/{len(h4[ticker])} "
                     f"1Hour={len(h1[ticker])}")
    return daily, weekly, h4, h1


def load_external(start: str, end: str, daily: dict, lines: list[str]) -> dict:
    """Each source optional; failures are recorded and the family stays missing."""
    external = {}
    try:
        from src.context.sentiment import fetch_vix
        external["vix"] = fetch_vix()
        lines.append(f"VIX rows={len(external['vix'])}")
    except Exception as error:
        lines.append(f"VIX unavailable: {redact_secrets(str(error))}")
    try:
        from src.context.breadth import BREADTH_BASKET, calculate_breadth
        from src.data.alpaca_client import download_historical_bars
        members = {t: daily.get(t) if t in daily else
                   download_historical_bars(t, start, end, "1Day", "regular") for t in BREADTH_BASKET}
        external["breadth"] = calculate_breadth(members)
        lines.append("breadth basket available (fixed 10 names, survivorship bias)")
    except Exception as error:
        lines.append(f"breadth unavailable: {redact_secrets(str(error))}")
    try:
        from src.context.liquidity import fetch_fred_series
        parts = [fetch_fred_series(name, start) for name in ("fed_assets", "tga", "rrp")]
        macro = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        external["macro_observations"] = macro
        lines.append(f"macro rows={len(macro)} asof_safe={bool(macro.asof_safe.all())} "
                     "(non-safe rows are excluded by the engine)")
    except Exception as error:
        lines.append(f"macro unavailable: {redact_secrets(str(error))}")
    return external


def write_outputs(runs, out: Path, lines: list[str], label: str):
    out.mkdir(parents=True, exist_ok=True)
    events = pd.concat([run.events for run in runs], ignore_index=True)
    events.to_csv(out/"pattern_events.csv", index=False)
    results = [r for run in runs for r in run.results]
    metrics = describe_patterns(events, results)
    for name, frame in metrics.items():
        if frame is not None and not frame.empty:
            frame.to_csv(out/f"metrics_{name}.csv")
    summary = metrics.get("summary", pd.DataFrame())
    overlap = metrics.get("overlap", pd.DataFrame())
    text = [f"# Fase 7: Pattern Engine — {label}", "",
            "Métricas descriptivas. No hay retornos, PnL, win rate ni señales.", "",
            f"Episodios: {len(events)}; abiertos al final de la muestra: {int(events.is_open.sum())}.", "",
            "## Por patrón (todos los tickers y timeframes)", ""]
    if not events.empty:
        by = events.groupby("pattern").agg(episodes=("pattern", "count"),
            developing=("reached_developing", "mean"), confirmed=("confirmed_at", lambda s: s.notna().mean()),
            invalidated=("state", lambda s: s.eq("invalidated").mean()),
            expired=("state", lambda s: s.eq("expired").mean()),
            mean_bars=("bars_active", "mean"), score_max_median=("score_max", "median"))
        text += ["| patrón | episodios | →developing | confirmados | invalidados | expirados | barras medias | score máx. mediano |",
                 "|---|---|---|---|---|---|---|---|"]
        text += [f"| {p} | {int(r.episodes)} | {r.developing:.0%} | {r.confirmed:.0%} | {r.invalidated:.0%} | "
                 f"{r.expired:.0%} | {r.mean_bars:.1f} | {r.score_max_median:.2f} |" for p, r in by.iterrows()]
    text += ["", "## Solapamiento (posibles redundancias)", ""]
    if overlap is not None and not overlap.empty:
        top = overlap.loc[overlap.potential_redundancy].head(10)
        text += [f"- {r.pattern_a} / {r.pattern_b}: jaccard {r.jaccard:.2f}, "
                 f"{r.share_of_a:.0%} de A, {r.share_of_b:.0%} de B" for r in top.itertuples()] \
            or ["- Ninguna pareja supera jaccard 0.5 ni 80% de cobertura."]
    text += ["", "## Registro", ""] + [f"- {line}" for line in lines]
    (out/"summary.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    if not summary.empty:
        summary.to_csv(out/"metrics_by_ticker.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="offline synthetic run")
    parser.add_argument("--debug-prefix", action="store_true",
                        help="on a prefix failure, report the first divergent bar, field and input")
    args = parser.parse_args()
    lines: list[str] = []
    runs = []
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        if args.synthetic:
            data = synthetic_inputs()
            end = (data["SYN_A"].timestamp.iloc[-1] + pd.Timedelta(days=1)).isoformat()
            benchmark = data["SYN_QQQ"]
            late = data["SYN_A"].timestamp.iloc[int(len(data["SYN_A"])*0.8)].strftime("%Y-%m-%d")
            for k, ticker in enumerate(("SYN_A", "SYN_B", "SYN_C", "SYN_D")):
                weekly = weekly_from_daily(data[ticker], end)
                context = {"1Week": weekly}
                if ticker == "SYN_A":  # intraday context only for the last 20% of the sample
                    context["4Hour"] = synthetic_intraday(data[ticker], late, 300+k, "4Hour")
                    context["1Hour"] = synthetic_intraday(data[ticker], late, 400+k, "1Hour")
                run, _ = validate(ticker, "1Day", data[ticker], benchmark, context, None, lines,
                                  debug_prefix=args.debug_prefix)
                runs.append(run)
            out, label = OUT/"synthetic", "validación SINTÉTICA (sin datos de mercado)"
        else:
            end = date.today().isoformat()  # current session excluded
            daily, weekly, h4, h1 = load_live("2022-01-01", (date.today()-timedelta(days=400)).isoformat(),
                                              end, lines)
            external = load_external("2022-01-01", end, daily, lines)
            for ticker in ASSETS:
                benchmark = None if ticker == BENCHMARK else daily[BENCHMARK]
                context = {"1Week": weekly[ticker], "4Hour": h4[ticker], "1Hour": h1[ticker]}
                run, _ = validate(ticker, "1Day", daily[ticker], benchmark, context, external, lines,
                                  debug_prefix=args.debug_prefix)
                runs.append(run)
                benchmark = None if ticker == BENCHMARK else h4[BENCHMARK]
                context = {"1Week": weekly[ticker], "1Day": daily[ticker], "1Hour": h1[ticker]}
                run, _ = validate(ticker, "4Hour", h4[ticker], benchmark, context, external, lines,
                                  debug_prefix=args.debug_prefix)
                runs.append(run)
            out, label = OUT, "validación con datos Alpaca"
        write_outputs(runs, out, lines, label)
        lines.append(f"PASS: all pattern validations complete ({len(runs)} runs); outputs in {out}")
    except Exception:
        lines.append("FAIL: " + redact_secrets(traceback.format_exc()))
        raise
    finally:
        LOG.write_text("\n".join(lines)+"\n", encoding="utf-8")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
