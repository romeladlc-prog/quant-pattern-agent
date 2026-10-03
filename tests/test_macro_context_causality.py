"""Macro (and news) status must depend only on rows known at each bar close.

Real-data failure (PR #4, ARM 1Day, T=2025-03-25): without FRED_API_KEY the
FRED CSV rows are current revisions with ``asof_safe=False`` and
``available_at`` = download time. Truncated to T they vanish, so the prefix run
got ``macro_status="missing"`` while the full run, which still held the rows,
stamped every past bar ``"excluded_not_asof_safe"``.
"""
import pandas as pd
import pytest

from src.patterns.checks import prefix_check
from src.patterns.engine import run_pattern_engine
from src.patterns.external_context import build_external_context
from synthetic import random_walk

NY = "America/New_York"
CLOSES = pd.Series((pd.bdate_range("2025-03-03", periods=40) + pd.Timedelta(hours=16))
                   .tz_localize(NY).tz_convert("UTC"))
FETCHED = pd.Timestamp("2026-10-03 13:40", tz="UTC")  # after every close


def macro_rows(series, dates, available_at, safe, value=1.0):
    dates = pd.DatetimeIndex(dates)
    available = available_at if isinstance(available_at, pd.Timestamp) else pd.DatetimeIndex(available_at)
    return pd.DataFrame({"series": series, "observation_date": dates,
                         "release_date": pd.NaT if not safe else dates, "available_at": available,
                         "value": value, "unit": "x", "source": series, "asof_safe": safe})


def fred_csv():
    """Current-revision CSV as fetched without an API key: available only at download time."""
    dates = pd.date_range("2024-10-02", "2025-04-30", freq="W-WED", tz="UTC")
    return pd.concat([macro_rows(name, dates, FETCHED, False) for name in ("fed_assets", "tga", "rrp")],
                     ignore_index=True)


def alfred_vintages():
    """As-of-safe weekly vintages, each available the day after its observation."""
    dates = pd.date_range("2025-02-19", "2025-04-30", freq="W-WED", tz="UTC")
    parts = [macro_rows(name, dates, dates + pd.Timedelta(days=1), True,
                        value=pd.Series(range(len(dates)), dtype=float)*(k+1) + 100)
             for k, name in enumerate(("fed_assets", "tga", "rrp"))]
    return pd.concat(parts, ignore_index=True)


def assert_prefix_invariant(macro, cuts=(1, 3, 10, 25, 40)):
    full = build_external_context(CLOSES, "T", macro_observations=macro)
    for k in cuts:
        close = CLOSES.iloc[k-1]
        known = None if macro is None else macro.loc[pd.to_datetime(macro.available_at, utc=True) <= close]
        prefix = build_external_context(CLOSES.iloc[:k], "T", macro_observations=known)
        pd.testing.assert_frame_equal(prefix, full.iloc[:k])
    return full


def test_fred_csv_rows_known_only_later_do_not_change_past_status():
    """Fails on PR #4 before the fix: prefix 'missing', full 'excluded_not_asof_safe'."""
    full = assert_prefix_invariant(fred_csv())
    assert set(full.macro_status) == {"not_yet_available"}
    assert full.macro_liquidity_state.isna().all()


def test_no_macro_data_is_missing_everywhere():
    full = assert_prefix_invariant(None)
    assert set(full.macro_status) == {"missing"}


def test_empty_supplied_macro_is_not_yet_available():
    empty = fred_csv().iloc[:0]
    assert set(build_external_context(CLOSES, "T", macro_observations=empty).macro_status) == \
        {"not_yet_available"}


def test_unsafe_rows_already_known_are_excluded():
    rows = macro_rows("fed_assets", ["2025-01-01"], [pd.Timestamp("2025-01-02", tz="UTC")], False)
    full = assert_prefix_invariant(rows)
    assert set(full.macro_status) == {"excluded_not_asof_safe"}


def test_rows_available_after_t_only_change_later_bars():
    rows = macro_rows("fed_assets", ["2025-03-07"], [CLOSES.iloc[10] + pd.Timedelta(hours=1)], False)
    full = assert_prefix_invariant(rows, cuts=(1, 10, 11, 12, 40))
    assert set(full.macro_status.iloc[:11]) == {"not_yet_available"}
    assert set(full.macro_status.iloc[11:]) == {"excluded_not_asof_safe"}


def test_asof_safe_vintages_are_used_from_their_availability():
    full = assert_prefix_invariant(alfred_vintages())
    assert set(full.macro_status) == {"asof_safe"}
    # 4-week change needs five weekly vintages: the first bars are still unknown.
    assert full.macro_liquidity_state.iloc[0] == "unknown"
    assert full.macro_liquidity_state.iloc[-1] in {"expanding", "contracting", "neutral"}


def test_safe_vintages_arriving_after_unsafe_rows():
    unsafe = macro_rows("fed_assets", ["2024-12-25"], [pd.Timestamp("2024-12-26", tz="UTC")], False)
    safe = alfred_vintages()
    safe = safe.assign(available_at=pd.to_datetime(safe.available_at, utc=True) + pd.Timedelta(days=21))
    full = assert_prefix_invariant(pd.concat([unsafe, safe], ignore_index=True))
    statuses = full.macro_status.tolist()
    assert statuses[0] == "excluded_not_asof_safe" and statuses[-1] == "asof_safe"


@pytest.mark.parametrize("news_rows", [0, 1])
def test_news_status_does_not_depend_on_future_rows(news_rows):
    news = pd.DataFrame({"ticker": "T", "published_at": [CLOSES.iloc[30] - pd.Timedelta(hours=2)]*news_rows,
                         "available_at": [CLOSES.iloc[30] - pd.Timedelta(hours=1)]*news_rows,
                         "headline": "x", "summary": "", "source": "", "score": 0.5,
                         "scoring_method": "v1"})
    full = build_external_context(CLOSES, "T", news=news)
    prefix = build_external_context(CLOSES.iloc[:10], "T", news=news.iloc[:0])
    pd.testing.assert_frame_equal(prefix, full.iloc[:10])
    assert set(full.news_quality) == {"insufficient_coverage"}


def test_engine_results_invariant_with_fred_csv_macro():
    bars = random_walk(120, seed=11)
    bars["timestamp"] = pd.bdate_range("2025-01-02", periods=120).tz_localize(NY).tz_convert("UTC")
    macro = fred_csv()
    full = run_pattern_engine("S", "1Day", bars, use_quant=False,
                              external_inputs={"macro_observations": macro})

    def prefix(stamp):
        close = full.evidence.close_time.loc[full.evidence.timestamp.eq(stamp)].iloc[0]
        known = macro.loc[pd.to_datetime(macro.available_at, utc=True) <= close]
        return run_pattern_engine("S", "1Day", bars.loc[bars.timestamp <= stamp], use_quant=False,
                                  external_inputs={"macro_observations": known})
    reports = prefix_check(full, prefix, [60, 100])
    assert [r["compared"] for r in reports] == [61*14, 101*14]
    assert {r.external_context["macro_status"] for r in full.results} == {"not_yet_available"}
