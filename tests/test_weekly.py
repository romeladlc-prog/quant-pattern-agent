"""1Week NORMALIZED bars from 1Day carry is_complete and keep [start, end)."""
import pandas as pd

from src.data.alpaca_client import normalize_bars


def daily_raw(days):
    stamps = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") for d in days]).tz_convert("UTC")
    frame = pd.DataFrame({"timestamp": stamps, "open": range(1, len(days)+1),
                          "high": [x+1 for x in range(1, len(days)+1)],
                          "low": [x-.5 for x in range(1, len(days)+1)],
                          "close": range(1, len(days)+1), "volume": [100]*len(days)})
    return frame.assign(symbol="T").set_index(["symbol", "timestamp"])


# Monday 2025-03-03 .. Friday 2025-03-14: two full trading weeks.
DAYS = pd.bdate_range("2025-03-03", "2025-03-14").strftime("%Y-%m-%d").tolist()


def test_complete_weeks():
    weekly = normalize_bars(daily_raw(DAYS), "2025-03-03", "2025-03-15", "1Week")
    assert weekly.is_complete.tolist() == [True, True]
    assert weekly.close.tolist() == [5, 10] and weekly.volume.tolist() == [500, 500]


def test_end_mid_week_marks_last_week_incomplete():
    # end = Wednesday 2025-03-12 00:00 UTC: Mon and Tue of week two are closed.
    weekly = normalize_bars(daily_raw(DAYS), "2025-03-03", "2025-03-12", "1Week")
    assert weekly.is_complete.tolist() == [True, False]
    assert weekly.close.iloc[-1] == 7  # Tuesday close; no later day leaks in
    assert weekly.source_last_timestamp.iloc[-1] < pd.Timestamp("2025-03-12", tz="UTC")


def test_end_before_friday_close_is_incomplete():
    # Friday's bar is labelled at its open, but its session closes at 16:00 ET.
    end = pd.Timestamp("2025-03-14 15:59", tz="America/New_York")
    weekly = normalize_bars(daily_raw(DAYS), "2025-03-03", end, "1Week")
    assert weekly.is_complete.tolist() == [True, False]
    assert weekly.close.iloc[-1] == 9  # Friday excluded until it closes
    end = pd.Timestamp("2025-03-14 16:00", tz="America/New_York")
    weekly = normalize_bars(daily_raw(DAYS), "2025-03-03", end, "1Week")
    assert weekly.is_complete.tolist() == [True, True]


def test_start_mid_week_marks_first_week_partial():
    weekly = normalize_bars(daily_raw(DAYS), "2025-03-05", "2025-03-15", "1Week")
    assert weekly.is_complete.tolist() == [False, True]
    assert weekly.open.iloc[0] == 3  # first day inside [start, end)


def test_empty_weekly_has_is_complete_column():
    weekly = normalize_bars(daily_raw(DAYS), "2025-03-03", "2025-03-03 12:00", "1Week")
    assert weekly.empty and "is_complete" in weekly
