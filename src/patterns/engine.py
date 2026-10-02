"""Phase 7 Pattern Engine: causal composite patterns with explicit evidence.

Pipeline per ticker and primary timeframe:
1. complete bars only (partial intraday blocks / weeks dropped);
2. quant context (walk-forward refits, filtered outputs) and external context
   (as-of ``available_at`` <= bar close), both optional;
3. evidence frame (features, as-of structure, Phase 6.1 breakouts, gaps);
4. other timeframes joined by close time as descriptive context; only the
   designated higher timeframe enters scoring, as one structure item;
5. detectors run their state machines bar by bar.

No signals, recommendations, returns or PnL are produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

from .base import PatternResult, clean
from .breakout import BreakoutDetector, FailedBreakoutDetector
from .config import HIGHER_TIMEFRAME, TIMEFRAME_PREFIX
from .divergence import DivergenceDetector
from .evidence import MTF_COLUMNS, attach_empty_timeframe, attach_timeframe, bar_close_times, \
    build_evidence, complete_bars
from .exhaustion import ExhaustionDetector
from .external_context import build_external_context
from .mean_reversion import MeanReversionDetector
from .quant_context import QuantContextConfig, build_quant_context
from .regime_transition import RegimeTransitionDetector
from .trend_continuation import TrendContinuationDetector

DIRECTIONAL = ("bullish", "bearish")
DEFAULT_DETECTORS = (
    (ExhaustionDetector, DIRECTIONAL),
    (BreakoutDetector, DIRECTIONAL),
    (FailedBreakoutDetector, DIRECTIONAL),
    (MeanReversionDetector, DIRECTIONAL),
    (TrendContinuationDetector, DIRECTIONAL),
    (RegimeTransitionDetector, ("to_high_vol", "to_low_vol")),
    (DivergenceDetector, DIRECTIONAL),
)
ACTIVE_STATES = ("candidate", "developing", "confirmed")
EVENT_COLUMNS = ["ticker", "timeframe", "pattern", "pattern_family", "direction", "subtype", "state",
                 "first_detected_at", "confirmed_at", "invalidated_at", "expired_at", "bars_active",
                 "is_open", "reached_developing", "score", "score_max", "score_at_detection",
                 "score_structure", "score_volatility", "score_regime", "score_context",
                 "score_momentum", "regime", "volatility", "level_asof_timestamp",
                 "key_evidence_for", "key_evidence_against", "final_evidence_for",
                 "final_evidence_against", "details"]


@dataclass
class PatternRun:
    ticker: str
    timeframe: str
    results: list[PatternResult]
    evidence: pd.DataFrame
    events: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def at(self, timestamp) -> list[PatternResult]:
        stamp = pd.Timestamp(timestamp)
        return [r for r in self.results if r.timestamp == stamp]


def _regime_label(row: dict) -> str:
    values = [v for v in (clean(row.get("hmm_p_high")), clean(row.get("markov_p_high"))) if v is not None]
    if not values:
        return "unavailable"
    if all(v >= 0.5 for v in values):
        return "high_vol"
    if all(v < 0.5 for v in values):
        return "low_vol"
    return "mixed"


def _volatility_label(row: dict) -> str:
    if clean(row.get("atr")) is None:
        return "unavailable"
    expansion = clean(row.get("atr_expansion"))
    if clean(row.get("volatility_expansion")) or (expansion is not None and expansion >= 1.25):
        return "expanding"
    if clean(row.get("volatility_compression")):
        return "compressed"
    return "normal"


def _pick(row: dict, keys: tuple[str, ...]) -> dict:
    return {k: clean(row.get(k)) for k in keys}


def bar_contexts(row: dict, mtf: list[str]) -> dict:
    regime = {"label": _regime_label(row), "hmm2_p_high": clean(row.get("hmm_p_high")),
              "markov2_p_high": clean(row.get("markov_p_high")),
              **_pick(row, ("cp_rv20_recent", "cp_slope_recent", "hurst", "permutation_entropy",
                            "quant_fit_index", "hmm_fit_converged", "markov_fit_converged",
                            "kalman_fit_converged"))}
    volatility = {"label": _volatility_label(row),
                  **_pick(row, ("realized_volatility_20", "rv20_percentile", "garch_vol",
                                "garch_vol_ratio", "atr", "atr_expansion", "volatility_compression",
                                "volatility_expansion", "compression_duration",
                                "garch_fit_converged"))}
    structure = _pick(row, ("structural_state", "last_high", "last_low", "last_high_label",
                            "last_low_label", "nearest_support", "nearest_resistance",
                            "support_dist_atr", "resistance_dist_atr", "levels_asof_timestamp",
                            "breakout_state", "breakout_direction", "breakout_age",
                            "fake_breakout_state", "htf_structural_state"))
    if "vix" in row or "news_quality" in row:
        external = _pick(row, ("vix", "vix_zscore", "vix_available_at", "breadth_sma50",
                               "breadth_available_at", "macro_liquidity_state", "macro_status",
                               "news_count_7d", "news_sentiment_7d", "news_quality"))
    else:
        external = {"status": "missing"}
    frames = {}
    for timeframe in mtf:
        prefix = TIMEFRAME_PREFIX[timeframe]
        frames[timeframe] = {c: clean(row.get(f"{prefix}_{c}")) for c in MTF_COLUMNS + ["close_time"]}
    return {"regime": regime, "volatility": volatility, "structure": structure,
            "external": external, "mtf": frames}


def _rows_with_events(frame: pd.DataFrame, events: pd.DataFrame) -> list[dict]:
    rows = frame.to_dict("records")
    for row in rows:
        row["events_here"] = []
    if not events.empty:
        for event in events.to_dict("records"):
            rows[int(event["bar_index"])]["events_here"].append(event)
    return rows


def run_pattern_engine(ticker: str, timeframe: str, bars: pd.DataFrame, *,
                       benchmark: pd.DataFrame | None = None,
                       context_bars: dict[str, pd.DataFrame] | None = None,
                       quant_config: QuantContextConfig | None = None,
                       external_inputs: dict | None = None,
                       detectors=DEFAULT_DETECTORS, use_quant: bool = True) -> PatternRun:
    """Run every detector on one ticker/timeframe. Inputs may extend past the
    last primary bar: every join is as-of, so later rows are ignored."""
    bars = complete_bars(bars)
    quant = build_quant_context(bars, quant_config or QuantContextConfig()) if use_quant else None
    external = None
    if external_inputs:
        external = build_external_context(bar_close_times(bars, timeframe), ticker, **external_inputs)
    bundle = build_evidence(bars, timeframe, benchmark=benchmark, quant=quant, external=external)
    frame = bundle.frame
    attached, with_data = [], []
    for other_tf, other_bars in (context_bars or {}).items():
        if other_tf == timeframe:
            continue
        # A requested timeframe keeps its columns (and its mtf_context key) even
        # with no complete bars yet: whether it has data later must not change
        # the shape of results stored for earlier bars.
        if other_bars is None or complete_bars(other_bars).empty:
            frame = attach_empty_timeframe(frame, other_tf)
        else:
            other = build_evidence(other_bars, other_tf, light=True).frame
            frame = attach_timeframe(frame, other, other_tf)
            with_data.append(other_tf)
        attached.append(other_tf)
    higher = HIGHER_TIMEFRAME.get(timeframe)
    column = f"{TIMEFRAME_PREFIX[higher]}_structural_state" if higher else None
    frame["htf_structural_state"] = frame[column] if column in frame else np.nan
    rows = _rows_with_events(frame, bundle.events)
    contexts = [bar_contexts(row, attached) for row in rows]
    results: list[PatternResult] = []
    for detector, directions in detectors:
        for direction in directions:
            results += detector(direction, ticker, timeframe).run(rows, contexts)
    meta = {**bundle.meta, "context_timeframes": attached, "context_timeframes_with_data": with_data,
            "higher_timeframe": higher, "higher_timeframe_available": higher in with_data}
    return PatternRun(ticker, timeframe, results, frame, episode_table(results), meta)


def _first(items):
    return items[0] if items else None


def episode_table(results: list[PatternResult]) -> pd.DataFrame:
    """One row per pattern episode, with its final known state.

    ``is_open`` episodes have not closed by the last bar; their final state can
    still change with later data and must be treated as provisional.
    """
    episodes: dict[tuple, list[PatternResult]] = {}
    for result in results:
        if result.episode_id is not None:
            episodes.setdefault((result.pattern_name, result.episode_id), []).append(result)
    rows = []
    for (_, _), items in sorted(episodes.items(), key=lambda kv: (kv[1][0].timestamp, kv[0][0])):
        first, last = items[0], items[-1]
        closed = last.invalidated_at is not None or last.expired_at is not None
        rows.append({
            "ticker": first.ticker, "timeframe": first.timeframe, "pattern": first.pattern_name,
            "pattern_family": first.pattern_family, "direction": first.direction,
            "subtype": last.subtype, "state": last.state,
            "first_detected_at": first.first_detected_at, "confirmed_at": last.confirmed_at,
            "invalidated_at": last.invalidated_at, "expired_at": last.expired_at,
            "bars_active": last.bars_active, "is_open": not closed,
            "reached_developing": any(r.state == "developing" for r in items),
            "score": last.score, "score_max": max(r.score for r in items),
            "score_at_detection": first.score, "score_structure": last.score_structure,
            "score_volatility": last.score_volatility, "score_regime": last.score_regime,
            "score_context": last.score_context, "score_momentum": last.score_momentum,
            "regime": first.regime_context.get("label"), "volatility": first.volatility_context.get("label"),
            "level_asof_timestamp": last.details.get("level_asof_timestamp"),
            "key_evidence_for": ";".join(first.evidence_for),
            "key_evidence_against": ";".join(first.evidence_against),
            "final_evidence_for": ";".join(last.evidence_for),
            "final_evidence_against": ";".join(last.evidence_against),
            "details": json.dumps(last.details, default=str, sort_keys=True)})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def describe_patterns(events: pd.DataFrame, results: list[PatternResult]) -> dict[str, pd.DataFrame]:
    """Descriptive statistics only: counts, durations, transitions, overlap.
    No returns, hit rates or profitability are computed."""
    out: dict[str, pd.DataFrame] = {}
    if events.empty:
        return {"summary": pd.DataFrame(), "overlap": pd.DataFrame()}
    frame = events.copy()
    first = pd.to_datetime(frame.first_detected_at, utc=True)
    keys = ["ticker", "timeframe", "pattern"]
    spans = {}
    for (ticker, timeframe), group in pd.DataFrame(
            [(r.ticker, r.timeframe, r.timestamp) for r in results],
            columns=["ticker", "timeframe", "timestamp"]).groupby(["ticker", "timeframe"]):
        spans[(ticker, timeframe)] = max((group.timestamp.max()-group.timestamp.min()).days/365.25, 1/365.25)
    grouped = frame.assign(first=first).groupby(keys)
    summary = grouped.agg(episodes=("state", "size"),
                          mean_bars_active=("bars_active", "mean"),
                          reached_developing=("reached_developing", "mean"),
                          confirmed=("confirmed_at", lambda s: s.notna().mean()),
                          invalidated=("state", lambda s: s.eq("invalidated").mean()),
                          expired=("state", lambda s: s.eq("expired").mean()),
                          still_open=("is_open", "mean"),
                          score_max_p25=("score_max", lambda s: s.quantile(.25)),
                          score_max_median=("score_max", "median"),
                          score_max_p75=("score_max", lambda s: s.quantile(.75))).reset_index()
    developing = frame.loc[frame.reached_developing]
    confirmed_given_dev = developing.groupby(keys).confirmed_at.apply(lambda s: s.notna().mean())
    summary = summary.merge(confirmed_given_dev.rename("confirmed_given_developing").reset_index(),
                            on=keys, how="left")
    summary["episodes_per_year"] = [row.episodes/spans[(row.ticker, row.timeframe)]
                                    for row in summary.itertuples()]
    out["summary"] = summary
    out["by_family"] = frame.groupby(["pattern_family", "state"]).size().unstack(fill_value=0)
    out["score_distribution"] = frame.groupby("pattern").score_max.describe(
        percentiles=[.1, .25, .5, .75, .9])
    active = pd.DataFrame([(r.ticker, r.timeframe, r.timestamp, r.pattern_name) for r in results
                           if r.state in ACTIVE_STATES],
                          columns=["ticker", "timeframe", "timestamp", "pattern"])
    if active.empty:
        out["overlap"] = pd.DataFrame()
        return out
    active["bar"] = list(zip(active.ticker, active.timeframe, active.timestamp))
    sets = active.groupby("pattern").bar.apply(set)
    names = sorted(sets.index)
    records = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            both = len(sets[a] & sets[b])
            if both:
                union = len(sets[a] | sets[b])
                records.append({"pattern_a": a, "pattern_b": b, "bars_together": both,
                                "jaccard": both/union, "share_of_a": both/len(sets[a]),
                                "share_of_b": both/len(sets[b])})
    overlap = pd.DataFrame(records, columns=["pattern_a", "pattern_b", "bars_together", "jaccard",
                                             "share_of_a", "share_of_b"])
    overlap["potential_redundancy"] = overlap.jaccard.ge(0.5) | overlap[["share_of_a", "share_of_b"]].max(axis=1).ge(0.8)
    out["overlap"] = overlap.sort_values("jaccard", ascending=False).reset_index(drop=True)
    simultaneous = active.groupby("bar").pattern.nunique()
    out["simultaneous"] = simultaneous.value_counts().sort_index().rename("bars").to_frame()
    return out
