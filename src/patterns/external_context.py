"""Per-bar External Context, joined strictly by ``available_at <= bar close``.

Every family is optional. A missing family yields NaN plus a status string,
never an artificial zero. Macro liquidity enters only from rows with
``asof_safe=True`` (ALFRED vintages); news sentiment only when the 7-day
window has enough scored headlines, otherwise it is left neutral (NaN).

Statuses are decided per bar from rows with ``available_at <= close`` only, so
appending later rows never changes an earlier bar. ``macro_status``:

* ``missing``: the macro family was not supplied to the engine (``None``:
  not requested or the download failed). Same for every bar.
* ``not_yet_available``: supplied, but no as-of-safe history usable at this
  close (no row available yet, or safe rows that do not yet form a series).
* ``excluded_not_asof_safe``: rows were available at this close, but none is
  ``asof_safe`` (current-revision CSV without vintages): excluded on purpose.
* ``asof_safe``: liquidity computed from as-of-safe rows known at this close.

An empty frame (e.g. a source truncated to T) is "supplied", not missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.context.liquidity import macro_liquidity_history
from src.context.sentiment import news_snapshot

from .config import THRESHOLDS

EXTERNAL_COLUMNS = ["vix", "vix_zscore", "vix_change_5d", "vix_available_at",
                    "breadth_sma50", "breadth_sma50_change5", "breadth_available_at",
                    "macro_liquidity_state", "macro_net_liquidity_4w_change", "macro_status",
                    "news_count_7d", "news_scored_7d", "news_sentiment_7d", "news_quality"]


def _asof(close_times: pd.Series, right: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    left = pd.DataFrame({"close_time": pd.to_datetime(close_times, utc=True).to_numpy(),
                         "_pos": np.arange(len(close_times))})
    right = right.sort_values("available_at")[["available_at"] + columns].copy()
    right["available_at"] = pd.to_datetime(right.available_at, utc=True)
    merged = pd.merge_asof(left.sort_values("close_time"), right, left_on="close_time",
                           right_on="available_at", direction="backward", allow_exact_matches=True)
    return merged.sort_values("_pos").reset_index(drop=True)


def build_external_context(close_times: pd.Series, ticker: str, *, vix: pd.DataFrame | None = None,
                           breadth: pd.DataFrame | None = None,
                           macro_observations: pd.DataFrame | None = None,
                           news: pd.DataFrame | None = None) -> pd.DataFrame:
    n = len(close_times)
    out = pd.DataFrame({c: pd.Series([np.nan]*n, dtype=object) for c in EXTERNAL_COLUMNS})
    out["macro_status"] = "missing"
    out["news_quality"] = "missing"
    if vix is not None and not vix.empty:
        merged = _asof(close_times, vix, ["vix", "vix_zscore_252d", "vix_change_5d"])
        out["vix"], out["vix_zscore"] = merged.vix, merged.vix_zscore_252d
        out["vix_change_5d"], out["vix_available_at"] = merged.vix_change_5d, merged.available_at
    if breadth is not None and not breadth.empty:
        frame = breadth.sort_values("available_at").copy()
        # Change over the breadth table's own prior rows: known at this row's availability.
        frame["breadth_sma50_change5"] = frame.breadth_sma50.diff(5)
        merged = _asof(close_times, frame, ["breadth_sma50", "breadth_sma50_change5"])
        out["breadth_sma50"], out["breadth_sma50_change5"] = merged.breadth_sma50, \
            merged.breadth_sma50_change5
        out["breadth_available_at"] = merged.available_at
    if macro_observations is not None:
        out["macro_status"] = _macro_status_and_values(out, close_times, macro_observations)
    if news is not None:  # supplied: per-bar coverage, even when no headline is known yet
        for i, stamp in enumerate(pd.to_datetime(close_times, utc=True)):
            snap = news_snapshot(news, ticker, stamp) if not news.empty else \
                {"news_count_7d": 0, "news_scored_7d": 0, "news_sentiment_7d": None}
            out.at[i, "news_count_7d"] = snap["news_count_7d"]
            out.at[i, "news_scored_7d"] = snap["news_scored_7d"]
            enough = snap["news_scored_7d"] >= THRESHOLDS["news_min_scored"]
            out.at[i, "news_quality"] = "ok" if enough else "insufficient_coverage"
            # Insufficient coverage: neutral (NaN), neither favours nor penalises.
            out.at[i, "news_sentiment_7d"] = snap["news_sentiment_7d"] if enough else np.nan
    return out


def _macro_status_and_values(out: pd.DataFrame, close_times: pd.Series,
                             observations: pd.DataFrame) -> list[str]:
    """Per-bar macro status from rows known at each close (see module docstring)."""
    stamps = pd.to_datetime(close_times, utc=True)
    if observations.empty:
        return ["not_yet_available"]*len(stamps)
    available = pd.to_datetime(observations.available_at, utc=True).reset_index(drop=True)
    safe_mask = observations.asof_safe.astype(bool).to_numpy()
    safe = observations.loc[safe_mask]
    if not safe_mask.all():
        out.attrs["macro_rows_excluded_not_asof_safe"] = int((~safe_mask).sum())
    statuses = []
    for i, stamp in enumerate(stamps):
        known = available.le(stamp).to_numpy()
        if not known.any():
            statuses.append("not_yet_available")
            continue
        if not (known & safe_mask).any():
            statuses.append("excluded_not_asof_safe")
            continue
        history = macro_liquidity_history(safe, stamp)
        if history.empty:
            statuses.append("not_yet_available")
            continue
        last = history.iloc[-1]
        out.at[i, "macro_liquidity_state"] = last.liquidity_state
        out.at[i, "macro_net_liquidity_4w_change"] = last.net_liquidity_4w_change
        statuses.append("asof_safe")
    return statuses
