"""Macro and security liquidity descriptors with explicit availability."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import os

import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    unit: str
    multiplier_to_million_usd: float | None = None


FRED_SERIES = {
    "fed_assets": FredSeries("WALCL", "USD millions", 1),
    "reserve_balances": FredSeries("WRESBAL", "USD millions", 1),
    "tga": FredSeries("WDTGAL", "USD millions", 1),
    "rrp": FredSeries("RRPONTSYD", "USD billions", 1000),
    "sofr": FredSeries("SOFR", "percent"),
    "treasury_2y": FredSeries("DGS2", "percent"),
    "treasury_10y": FredSeries("DGS10", "percent"),
    "credit_spread": FredSeries("BAMLH0A0HYM2", "percentage points"),
}


def fetch_fred_series(name: str, start: str, api_key: str | None = None,
                      timeout: int = 30) -> pd.DataFrame:
    """ALFRED vintages if keyed; otherwise current-revision CSV, current use only.

    release_date is a *date*, not an intraday publication time. Vintages become
    available at 00:00 UTC the following day, a conservative daily convention.
    """
    spec = FRED_SERIES[name]
    key = api_key or os.getenv("FRED_API_KEY")
    if key:
        response = requests.get("https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": spec.series_id, "api_key": key, "file_type": "json",
                    "observation_start": start, "realtime_start": start,
                    "realtime_end": datetime.now(timezone.utc).date().isoformat(),
                    "limit": 100000}, timeout=timeout)
        response.raise_for_status()
        observations = response.json()["observations"]
        rows = []
        for item in observations:
            value = pd.to_numeric(item["value"], errors="coerce")
            if pd.isna(value):
                continue
            release = pd.Timestamp(item["realtime_start"], tz="UTC")
            rows.append({"series": name, "observation_date": pd.Timestamp(item["date"], tz="UTC"),
                         "release_date": release, "available_at": release + pd.Timedelta(days=1),
                         "value": float(value), "unit": spec.unit, "source": spec.series_id,
                         "asof_safe": True})
        return pd.DataFrame(rows)
    response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                            params={"id": spec.series_id, "cosd": start}, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty:
        return pd.DataFrame()
    value = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    fetched = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame({"series": name,
        "observation_date": pd.to_datetime(frame.iloc[:, 0], utc=True),
        "release_date": pd.NaT, "available_at": fetched, "value": value,
        "unit": spec.unit, "source": spec.series_id, "asof_safe": False}).dropna(subset=["value"])


def macro_asof(observations: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Latest vintage per observation known at as_of, then latest observation."""
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone aware")
    eligible = observations.loc[observations.available_at.le(as_of)].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(["series", "observation_date", "available_at"])
    vintage = eligible.drop_duplicates(["series", "observation_date"], keep="last")
    return vintage.sort_values("observation_date").drop_duplicates("series", keep="last")


def macro_liquidity_history(observations: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Weekly descriptive net liquidity in USD millions, as known at as_of."""
    eligible = observations.loc[observations.available_at.le(pd.Timestamp(as_of))].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible = eligible.sort_values("available_at").drop_duplicates(
        ["series", "observation_date"], keep="last")
    eligible["millions"] = eligible.apply(lambda row: row.value *
        (FRED_SERIES[row.series].multiplier_to_million_usd or np.nan), axis=1)
    levels = eligible.pivot(index="observation_date", columns="series", values="millions")
    if not {"fed_assets", "tga", "rrp"}.issubset(levels.columns):
        return pd.DataFrame()
    # H.4.1 is Wednesday-level weekly; use only its observation dates. RRP is
    # daily and selected from the most recent date no later than that Wednesday.
    weekly = levels.reindex(levels.index[levels.fed_assets.notna()]).copy()
    for name in ("tga", "rrp"):
        weekly[name] = levels[name].reindex(weekly.index, method="ffill")
    weekly["net_liquidity"] = weekly.fed_assets - weekly.tga - weekly.rrp
    weekly["net_liquidity_1w_change"] = weekly.net_liquidity.diff(1)
    weekly["net_liquidity_4w_change"] = weekly.net_liquidity.diff(4)
    mean = weekly.net_liquidity.rolling(52, min_periods=26).mean()
    std = weekly.net_liquidity.rolling(52, min_periods=26).std()
    weekly["net_liquidity_zscore_52w"] = (weekly.net_liquidity - mean) / std.replace(0, np.nan)
    weekly["net_liquidity_slope_4w"] = weekly.net_liquidity.diff(4) / 4
    weekly["liquidity_state"] = np.select(
        [weekly.net_liquidity_4w_change > 0, weekly.net_liquidity_4w_change < 0],
        ["expanding", "contracting"], default="neutral")
    weekly.loc[weekly.net_liquidity_4w_change.isna(), "liquidity_state"] = "unknown"
    return weekly


def asset_liquidity(bars: pd.DataFrame, shares_outstanding: float | None = None) -> pd.DataFrame:
    """Causal OHLCV descriptors. Turnover needs point-in-time share count."""
    frame = bars.sort_values("timestamp").set_index("timestamp").copy()
    close, volume = frame.close.astype(float), frame.volume.astype(float)
    out = pd.DataFrame(index=frame.index)
    out["dollar_volume"] = close * volume
    baseline = volume.shift(1).rolling(20, min_periods=20).mean()
    out["relative_volume"] = volume / baseline.replace(0, np.nan)
    out["turnover"] = volume / shares_outstanding if shares_outstanding and shares_outstanding > 0 else np.nan
    returns = close.pct_change(fill_method=None).abs()
    out["amihud_illiquidity"] = returns / out.dollar_volume.replace(0, np.nan)
    out["volatility_per_volume"] = np.log(close).diff().abs() / volume.replace(0, np.nan)
    out["overnight_gap"] = frame.open.astype(float) / close.shift(1) - 1
    out["available_at"] = (frame.index.tz_convert("America/New_York").normalize() +
                           pd.Timedelta(hours=16, minutes=30)).tz_convert("UTC")
    return out
