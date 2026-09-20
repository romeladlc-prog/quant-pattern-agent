"""Point-in-time composition of separate external context families."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .liquidity import macro_asof, macro_liquidity_history
from .sentiment import news_snapshot


@dataclass(frozen=True)
class MarketContextSnapshot:
    ticker: str
    as_of: pd.Timestamp
    volatility_context: dict
    macro_liquidity_context: dict
    asset_liquidity_context: dict
    breadth_context: dict
    news_sentiment_context: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _latest(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    if frame.empty:
        return None
    eligible = frame.loc[frame.available_at.le(as_of)]
    return eligible.sort_values("available_at").iloc[-1] if not eligible.empty else None


def _value(value):
    if value is None or pd.isna(value):
        return None
    return value


def make_snapshot(ticker: str, as_of: pd.Timestamp, *, vix: pd.DataFrame,
                  macro_observations: pd.DataFrame, asset: pd.DataFrame,
                  breadth: pd.DataFrame, news: pd.DataFrame) -> MarketContextSnapshot:
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("Snapshot as_of must be timezone-aware")
    as_of = as_of.tz_convert("UTC")
    vix_row = _latest(vix, as_of)
    asset_row = _latest(asset, as_of)
    breadth_row = _latest(breadth, as_of)
    macro = macro_asof(macro_observations, as_of) if not macro_observations.empty else pd.DataFrame()
    history = macro_liquidity_history(macro_observations, as_of) if not macro.empty else pd.DataFrame()
    macro_values = {row.series: row for row in macro.itertuples()} if not macro.empty else {}
    net = history.iloc[-1] if not history.empty else None
    macro_context = {"asof_safe": bool(macro.asof_safe.all()) if not macro.empty else False,
                     "series_observation_dates": {key: str(row.observation_date) for key, row in macro_values.items()},
                     "series_release_dates": {key: str(row.release_date) if pd.notna(row.release_date) else None
                                              for key, row in macro_values.items()},
                     "net_liquidity_observation_date": str(history.index[-1]) if not history.empty else None,
                     "net_liquidity_usd_millions": _value(net.net_liquidity) if net is not None else None,
                     "net_liquidity_1w_change": _value(net.net_liquidity_1w_change) if net is not None else None,
                     "net_liquidity_4w_change": _value(net.net_liquidity_4w_change) if net is not None else None,
                     "net_liquidity_zscore_52w": _value(net.net_liquidity_zscore_52w) if net is not None else None,
                     "net_liquidity_slope_4w": _value(net.net_liquidity_slope_4w) if net is not None else None,
                     "liquidity_state": net.liquidity_state if net is not None else "unknown"}
    for name in ("fed_assets", "reserve_balances", "tga", "rrp", "sofr", "treasury_2y",
                 "treasury_10y", "credit_spread"):
        macro_context[name] = _value(macro_values[name].value) if name in macro_values else None
    if macro_context["treasury_2y"] is not None and macro_context["treasury_10y"] is not None:
        macro_context["spread_2s10s_percentage_points"] = (
            macro_context["treasury_10y"] - macro_context["treasury_2y"])
    else:
        macro_context["spread_2s10s_percentage_points"] = None
    volatility = ({"vix": _value(vix_row.vix), "vix_change_1d": _value(vix_row.vix_change_1d),
                   "vix_change_5d": _value(vix_row.vix_change_5d),
                   "vix_zscore_252d": _value(vix_row.vix_zscore_252d),
                   "observation_date": str(vix_row.observation_date),
                   "available_at": str(vix_row.available_at)} if vix_row is not None else {})
    asset_context = ({key: _value(asset_row[key]) for key in ("dollar_volume", "relative_volume",
                      "turnover", "amihud_illiquidity", "volatility_per_volume", "overnight_gap")}
                     if asset_row is not None else {})
    if asset_row is not None:
        asset_context["available_at"] = str(asset_row.available_at)
    market_breadth = ({key: _value(breadth_row[key]) for key in ("members_available",
                       "breadth_sma20", "breadth_sma50", "breadth_sma200", "advances", "declines",
                       "advance_decline_ratio", "new_highs_252d", "new_lows_252d")}
                      if breadth_row is not None else {})
    if breadth_row is not None:
        market_breadth["available_at"] = str(breadth_row.available_at)
    news_context = news_snapshot(news, ticker, as_of) if not news.empty else {}
    return MarketContextSnapshot(ticker, as_of, volatility, macro_context,
                                 asset_context, market_breadth, news_context)
