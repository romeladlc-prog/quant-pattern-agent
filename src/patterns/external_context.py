"""Per-bar External Context, joined strictly by ``available_at <= bar close``.

Every family is optional. A missing family yields NaN plus a status string,
never an artificial zero. Macro liquidity enters only from rows with
``asof_safe=True`` (ALFRED vintages); news sentiment only when the 7-day
window has enough scored headlines, otherwise it is left neutral (NaN).
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
    if macro_observations is not None and not macro_observations.empty:
        safe = macro_observations.loc[macro_observations.asof_safe.astype(bool)]
        if safe.empty:
            out["macro_status"] = "excluded_not_asof_safe"
        else:
            out["macro_status"] = "asof_safe"
            for i, stamp in enumerate(pd.to_datetime(close_times, utc=True)):
                history = macro_liquidity_history(safe, stamp)
                if history.empty:
                    out.at[i, "macro_status"] = "not_yet_available"
                    continue
                last = history.iloc[-1]
                out.at[i, "macro_liquidity_state"] = last.liquidity_state
                out.at[i, "macro_net_liquidity_4w_change"] = last.net_liquidity_4w_change
            if len(safe) < len(macro_observations):
                out.attrs["macro_rows_excluded_not_asof_safe"] = int(len(macro_observations)-len(safe))
    if news is not None and not news.empty:
        for i, stamp in enumerate(pd.to_datetime(close_times, utc=True)):
            snap = news_snapshot(news, ticker, stamp)
            out.at[i, "news_count_7d"] = snap["news_count_7d"]
            out.at[i, "news_scored_7d"] = snap["news_scored_7d"]
            enough = snap["news_scored_7d"] >= THRESHOLDS["news_min_scored"]
            out.at[i, "news_quality"] = "ok" if enough else "insufficient_coverage"
            # Insufficient coverage: neutral (NaN), neither favours nor penalises.
            out.at[i, "news_sentiment_7d"] = snap["news_sentiment_7d"] if enough else np.nan
    return out
