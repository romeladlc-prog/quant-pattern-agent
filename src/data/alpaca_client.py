"""Alpaca RAW responses and UTC-normalized US equity bars."""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Union

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

DateLike = Union[str, date, datetime, pd.Timestamp]
SUPPORTED_TIMEFRAMES = {"15Min", "1Hour", "4Hour", "1Day", "1Week"}
SESSIONS = {"regular", "extended"}
NY = "America/New_York"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _as_utc(value: DateLike, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} no es una fecha válida: {value!r}") from error
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _bounds(start: DateLike, end: DateLike) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = _as_utc(start, "start"), _as_utc(end, "end")
    if first >= last:
        raise ValueError("La fecha inicial debe ser anterior a la fecha final.")
    return first, last


def _settings() -> tuple[str, str]:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    names = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL")
    values = [os.getenv(name, "").strip() for name in names]
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise ValueError("Faltan variables de entorno requeridas: " + ", ".join(missing))
    return values[0], values[1]


def _alpaca_timeframe(value: str) -> TimeFrame:
    return {"15Min": TimeFrame(15, TimeFrameUnit.Minute), "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day, "1Week": TimeFrame.Week}[value]


def download_raw_bars(symbol: str, start: DateLike, end: DateLike,
                      timeframe: str = "1Day") -> pd.DataFrame:
    """Return Alpaca response.df unchanged, including its index and inclusive end."""
    if not symbol or not symbol.strip():
        raise ValueError("El símbolo no puede estar vacío.")
    if timeframe not in {"15Min", "1Hour", "1Day", "1Week"}:
        raise ValueError("RAW solo admite 15Min, 1Hour, 1Day o 1Week.")
    first, last = _bounds(start, end)
    key, secret = _settings()
    request = StockBarsRequest(symbol_or_symbols=symbol.strip().upper(),
                               timeframe=_alpaca_timeframe(timeframe),
                               start=first.to_pydatetime(), end=last.to_pydatetime())
    try:
        return StockHistoricalDataClient(key, secret).get_stock_bars(request).df
    except Exception as error:
        raise RuntimeError(f"No se pudieron descargar datos de Alpaca para {symbol.strip().upper()}: {error}") from error


def _frame(raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Copy RAW into a sorted UTC frame with strict [start, end) labels."""
    bars = raw.reset_index().copy()
    if "symbol" in bars:
        bars = bars.drop(columns="symbol")
    if bars.empty:
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                             **{name: pd.Series(dtype="float64") for name in COLUMNS[1:]}})
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.loc[bars["timestamp"].ge(start) & bars["timestamp"].lt(end)]
    return bars.sort_values("timestamp").reset_index(drop=True)


def _session_filter(bars: pd.DataFrame, session: str) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    clock = bars["timestamp"].dt.tz_convert(NY).dt.time
    opening, closing = ((time(9, 30), time(16)) if session == "regular"
                        else (time(4), time(20)))
    return bars.loc[clock.ge(opening) & clock.lt(closing)].reset_index(drop=True)


def _aggregate(window: pd.DataFrame, timestamp: pd.Timestamp) -> dict:
    return {"timestamp": timestamp, "open": window["open"].iloc[0],
            "high": window["high"].max(), "low": window["low"].min(),
            "close": window["close"].iloc[-1], "volume": window["volume"].sum()}


def _intraday_aggregate(bars: pd.DataFrame, session: str, hours: int) -> pd.DataFrame:
    """Regular uses 15m bars anchored 09:30 ET; extended uses 1h bars at 04:00 ET.

    Regular 4H windows: 09:30-13:30, 13:30-16:00. Extended 4H windows:
    04-08, 08-12, 12-16, 16-20. A missing source slot makes a block partial.
    """
    if bars.empty:
        return pd.DataFrame(columns=COLUMNS + ["is_complete"])
    step = pd.Timedelta(minutes=15 if session == "regular" else 60)
    opening = time(9, 30) if session == "regular" else time(4)
    closing = time(16) if session == "regular" else time(20)
    local = bars["timestamp"].dt.tz_convert(NY)
    records = []
    for day, group in bars.groupby(local.dt.date, sort=True):
        day_open = pd.Timestamp(datetime.combine(day, opening), tz=NY)
        day_close = pd.Timestamp(datetime.combine(day, closing), tz=NY)
        slots = ((group["timestamp"].dt.tz_convert(NY) - day_open) // pd.Timedelta(hours=hours)).astype(int)
        for slot, window in group.groupby(slots, sort=True):
            left = day_open + slot * pd.Timedelta(hours=hours)
            right = min(left + pd.Timedelta(hours=hours), day_close)
            expected = pd.date_range(left, right, freq=step, inclusive="left").tz_convert("UTC")
            label = left.tz_convert("UTC")
            if label < bars["timestamp"].iloc[0]:
                label = window["timestamp"].iloc[0]
            record = _aggregate(window, label)
            record["is_complete"] = (right - left == pd.Timedelta(hours=hours) and
                window["timestamp"].reset_index(drop=True).equals(
                    pd.Series(expected, name="timestamp")))
            records.append(record)
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def _weekly(bars: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Aggregate only fully closed 1Day bars by New York trading week."""
    if bars.empty:
        return pd.DataFrame(columns=COLUMNS + ["source_last_timestamp", "source_last_session_close"])
    local_dates = bars["timestamp"].dt.tz_convert(NY).dt.date
    closes = local_dates.map(lambda day: pd.Timestamp(datetime.combine(day, time(16)), tz=NY).tz_convert("UTC"))
    bars = bars.loc[closes.le(end)].reset_index(drop=True)
    if bars.empty:
        return pd.DataFrame(columns=COLUMNS + ["source_last_timestamp", "source_last_session_close"])
    local_dates = bars["timestamp"].dt.tz_convert(NY).dt.date
    monday = local_dates.map(lambda day: day - timedelta(days=day.weekday()))
    records = []
    for _, window in bars.groupby(monday, sort=True):
        record = _aggregate(window, window["timestamp"].iloc[0])
        record["source_last_timestamp"] = window["timestamp"].iloc[-1]
        last_day = window["timestamp"].iloc[-1].tz_convert(NY).date()
        record["source_last_session_close"] = pd.Timestamp(
            datetime.combine(last_day, time(16)), tz=NY).tz_convert("UTC")
        records.append(record)
    return pd.DataFrame(records).reset_index(drop=True)


def normalize_bars(raw: pd.DataFrame, start: DateLike, end: DateLike,
                   timeframe: str = "1Day", session: str = "regular") -> pd.DataFrame:
    """Create a new analytical frame; never mutate RAW.

    Input source: 1Day for 1Week; 15Min for regular 1Hour/4Hour;
    1Hour for extended 4Hour; otherwise the requested timeframe.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES or session not in SESSIONS:
        raise ValueError("Timeframe o sesión no soportado.")
    if timeframe in {"1Day", "1Week"} and session != "regular":
        raise ValueError("1Day y 1Week solo admiten session='regular'.")
    first, last = _bounds(start, end)
    bars = _frame(raw, first, last)
    if timeframe == "1Week":
        return _weekly(bars, last)
    if timeframe == "1Day":
        return bars
    bars = _session_filter(bars, session)
    if timeframe == "15Min" or (timeframe == "1Hour" and session == "extended"):
        return bars
    return _intraday_aggregate(bars, session, 1 if timeframe == "1Hour" else 4)


def download_historical_bars(symbol: str, start: DateLike, end: DateLike,
                             timeframe: str = "1Day", session: str = "regular") -> pd.DataFrame:
    """Download NORMALIZED [start, end) bars; regular is the model default."""
    if timeframe not in SUPPORTED_TIMEFRAMES or session not in SESSIONS:
        raise ValueError("Timeframe o sesión no soportado.")
    if timeframe in {"1Day", "1Week"} and session != "regular":
        raise ValueError("1Day y 1Week solo admiten session='regular'.")
    source = ("1Day" if timeframe == "1Week" else
              "15Min" if session == "regular" and timeframe in {"1Hour", "4Hour"} else
              "1Hour" if timeframe == "4Hour" else timeframe)
    raw = download_raw_bars(symbol, start, end, source)
    return normalize_bars(raw, start, end, timeframe, session)
