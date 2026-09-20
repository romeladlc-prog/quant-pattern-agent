"""Validate immutable Alpaca RAW and session-aware NORMALIZED ARM data."""
from __future__ import annotations

import sys
from datetime import time

import pandas as pd

from src.data.alpaca_client import COLUMNS, download_raw_bars, normalize_bars

SYMBOL = "ARM"
RANGES = {
    "15Min": ("2025-01-01", "2025-02-01"),
    "1Hour": ("2025-01-01", "2025-04-01"),
    "4Hour": ("2025-01-01", "2025-04-01"),
    "1Day": ("2024-01-01", "2025-04-01"),
    "1Week": ("2022-01-01", "2025-04-01"),
}


def source_timeframe(timeframe: str, session: str) -> str:
    if timeframe == "1Week":
        return "1Day"
    if session == "regular" and timeframe in {"1Hour", "4Hour"}:
        return "15Min"
    return "1Hour" if timeframe == "4Hour" else timeframe


def check_bars(bars: pd.DataFrame) -> list[str]:
    problems = []
    if any(column not in bars for column in COLUMNS):
        return ["missing OHLCV columns"]
    if bars.empty:
        return ["empty dataset"]
    if bars[COLUMNS].isna().any().any():
        problems.append("NaN in timestamp or OHLCV")
    if bars["timestamp"].duplicated().any():
        problems.append("duplicate timestamps")
    if not bars["timestamp"].is_monotonic_increasing:
        problems.append("timestamps out of order")
    if bars["high"].lt(bars[["open", "close", "low"]].max(axis=1)).any():
        problems.append("high below open, close or low")
    if bars["low"].gt(bars[["open", "close", "high"]].min(axis=1)).any():
        problems.append("low above open, close or high")
    if bars["volume"].lt(0).any():
        problems.append("negative volume")
    if str(bars["timestamp"].dt.tz) != "UTC":
        problems.append("timestamps are not UTC-aware")
    return problems


def check_normalized(bars: pd.DataFrame, timeframe: str, session: str,
                     start: str, end: str) -> list[str]:
    problems = check_bars(bars)
    if bars.empty:
        return problems
    first, last = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if not bars["timestamp"].ge(first).all() or not bars["timestamp"].lt(last).all():
        problems.append("timestamp outside [start, end)")
    if timeframe in {"15Min", "1Hour", "4Hour"} and session == "regular":
        clock = bars["timestamp"].dt.tz_convert("America/New_York").dt.time
        if not (clock.ge(time(9, 30)) & clock.lt(time(16))).all():
            problems.append("regular bar starts outside 09:30-16:00 ET")
    if timeframe in {"15Min", "1Hour", "4Hour"} and session == "extended":
        clock = bars["timestamp"].dt.tz_convert("America/New_York").dt.time
        if not (clock.ge(time(4)) & clock.lt(time(20))).all():
            problems.append("extended bar starts outside 04:00-20:00 ET")
    if timeframe == "1Week":
        if "source_last_session_close" not in bars:
            problems.append("missing weekly source provenance")
        elif not bars["source_last_session_close"].le(last).all():
            problems.append("weekly bar uses a day whose session closes after end")
    if timeframe == "4Hour":
        if "is_complete" not in bars or bars["is_complete"].isna().any():
            problems.append("missing 4Hour completeness flag")
    return problems


def check_aggregates(weekly: pd.DataFrame, daily: pd.DataFrame,
                     four_hour: pd.DataFrame, quarter_hour: pd.DataFrame) -> list[str]:
    """Independently compare weekly inputs and regular 4H OHLCV totals."""
    problems = []
    if not weekly.empty:
        local = daily["timestamp"].dt.tz_convert("America/New_York").dt.date
        monday = local.map(lambda day: day - pd.Timedelta(days=day.weekday()))
        groups = list(daily.groupby(monday, sort=True))
        if len(groups) != len(weekly):
            problems.append("weekly group count differs from clipped daily source")
        else:
            for (_, group), row in zip(groups, weekly.itertuples(index=False)):
                expected = (group["open"].iloc[0], group["high"].max(),
                            group["low"].min(), group["close"].iloc[-1], group["volume"].sum())
                actual = (row.open, row.high, row.low, row.close, row.volume)
                if not all(abs(a - b) < 1e-8 for a, b in zip(actual, expected)):
                    problems.append("weekly OHLCV differs from daily source")
                    break
    if not four_hour.empty:
        local = quarter_hour["timestamp"].dt.tz_convert("America/New_York")
        opening = local.dt.normalize() + pd.Timedelta(hours=9, minutes=30)
        slots = ((local - opening) // pd.Timedelta(hours=4)).astype(int)
        keys = list(zip(local.dt.date, slots))
        groups = list(quarter_hour.groupby(pd.Series(keys), sort=True))
        if len(groups) != len(four_hour):
            problems.append("4Hour group count differs from regular 15Min source")
        else:
            for (_, group), row in zip(groups, four_hour.itertuples(index=False)):
                expected = (group["open"].iloc[0], group["high"].max(),
                            group["low"].min(), group["close"].iloc[-1], group["volume"].sum())
                actual = (row.open, row.high, row.low, row.close, row.volume)
                if not all(abs(a - b) < 1e-8 for a, b in zip(actual, expected)):
                    problems.append("4Hour OHLCV differs from regular 15Min source")
                    break
    return problems


def main() -> int:
    rows = []
    cache = {}
    normalized = {}
    reported_raw = set()

    def raw(source: str, start: str, end: str) -> pd.DataFrame:
        key = (source, start, end)
        if key not in cache:
            cache[key] = download_raw_bars(SYMBOL, start, end, source)
        return cache[key]

    def add(dataset: str, timeframe: str, session: str, bars: pd.DataFrame,
            problems: list[str]) -> None:
        first = str(bars["timestamp"].iloc[0]) if not bars.empty else "-"
        last = str(bars["timestamp"].iloc[-1]) if not bars.empty else "-"
        rows.append([dataset, timeframe, session, len(bars), first, last, problems])
        print(f"\n{dataset} {timeframe} {session}: {len(bars)} rows; {', '.join(problems) if problems else 'OK'}")
        if "is_complete" in bars:
            print(f"  Blocks complete={int(bars['is_complete'].sum())}, partial={int((~bars['is_complete']).sum())}")
        print(bars[COLUMNS].tail(5).to_string(index=False))

    for timeframe, (start, end) in RANGES.items():
        sessions = ("regular", "extended") if timeframe in {"15Min", "1Hour", "4Hour"} else ("regular",)
        for session in sessions:
            source = source_timeframe(timeframe, session)
            try:
                source_raw = raw(source, start, end)
                key = (source, start, end)
                if key not in reported_raw:
                    raw_frame = source_raw.reset_index().copy()
                    if "symbol" in raw_frame:
                        raw_frame = raw_frame.drop(columns="symbol")
                    raw_frame["timestamp"] = pd.to_datetime(raw_frame["timestamp"], utc=True)
                    problems = check_bars(raw_frame)
                    end_count = int(raw_frame["timestamp"].ge(pd.Timestamp(end, tz="UTC")).sum())
                    print(f"RAW {source}: {end_count} bars at/after requested end (preserved)")
                    add("RAW", source, f"all [{start}, {end}]", raw_frame, problems)
                    reported_raw.add(key)
                bars = normalize_bars(source_raw, start, end, timeframe, session)
                normalized[(timeframe, session)] = bars
                add("NORMALIZED", timeframe, session, bars,
                    check_normalized(bars, timeframe, session, start, end))
            except Exception as error:
                rows.append(["NORMALIZED", timeframe, session, 0, "-", "-", [str(error)]])
                print(f"ERROR {timeframe} {session}: {error}")

    # Native weekly bars remain available only as RAW; models use daily-built weeks.
    try:
        start, end = RANGES["1Week"]
        weekly_raw = raw("1Week", start, end).reset_index().copy()
        if "symbol" in weekly_raw:
            weekly_raw = weekly_raw.drop(columns="symbol")
        weekly_raw["timestamp"] = pd.to_datetime(weekly_raw["timestamp"], utc=True)
        add("RAW", "1Week", f"all [{start}, {end}]", weekly_raw, check_bars(weekly_raw))
    except Exception as error:
        rows.append(["RAW", "1Week", "native", 0, "-", "-", [str(error)]])

    needed = [("1Week", "regular"), ("1Day", "regular"),
              ("4Hour", "regular"), ("15Min", "regular")]
    if all(key in normalized for key in needed):
        daily_for_week = normalize_bars(
            raw("1Day", *RANGES["1Week"]), *RANGES["1Week"], "1Day")
        quarter_for_four = normalize_bars(
            raw("15Min", *RANGES["4Hour"]), *RANGES["4Hour"], "15Min", "regular")
        issues = check_aggregates(normalized[("1Week", "regular")], daily_for_week,
                                  normalized[("4Hour", "regular")], quarter_for_four)
        for row in rows:
            if row[0] == "NORMALIZED" and row[1] in {"1Week", "4Hour"}:
                row[6].extend(issue for issue in issues if row[1] in issue)
        cutoff = pd.Timestamp("2025-03-31T15:00:00Z")  # 11:00 ET, before that day's close.
        mid_session = normalize_bars(raw("1Day", *RANGES["1Week"]),
                                     RANGES["1Week"][0], cutoff, "1Week")
        if not mid_session.empty and mid_session["source_last_session_close"].gt(cutoff).any():
            next(row for row in rows if row[0] == "NORMALIZED" and row[1] == "1Week")[6].append(
                "mid-session cutoff leaked a later daily close")

    print("\nDATASET | TIMEFRAME | SESSION | ROWS | FIRST | LAST | VALID")
    for dataset, timeframe, session, count, first, last, problems in rows:
        print(f"{dataset} | {timeframe} | {session} | {count} | {first} | {last} | {'OK' if not problems else 'FAIL'}")
        for issue in problems:
            print(f"  - {issue}")
    return int(any(row[6] for row in rows))


if __name__ == "__main__":
    sys.exit(main())
