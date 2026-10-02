"""Prefix causality: adding future bars never changes a stored past result."""
import pandas as pd
import pytest

from src.data.alpaca_client import normalize_bars
from src.patterns.checks import check_results, normalized, prefix_check
from src.patterns.engine import run_pattern_engine
from src.patterns.evidence import bar_close_times, breakout_recency
from src.patterns.external_context import build_external_context
from src.patterns.quant_context import MODELS_USED, QuantContextConfig, build_quant_context
from src.models.model_status import model_status
from synthetic import random_walk

NY = "America/New_York"


def daily(n=320, seed=5):
    bars = random_walk(n, seed)
    bars["timestamp"] = pd.bdate_range("2024-01-01", periods=n).tz_localize(NY).tz_convert("UTC")
    return bars


def weekly(bars, end):
    raw = bars.assign(symbol="S").set_index(["symbol", "timestamp"])
    return normalize_bars(raw, "2024-01-01", end, "1Week", "regular")


def truncated(frame, stamp, timeframe):
    closes = bar_close_times(frame, timeframe)
    return frame.loc[(closes <= stamp).to_numpy()].reset_index(drop=True)


@pytest.mark.parametrize("cut", [120, 200, 290])
def test_engine_prefix_invariance_structure_context(cut):
    bars, bench = daily(), daily(seed=9)
    end = "2025-06-01"
    wk = weekly(bars, end)
    full = run_pattern_engine("S", "1Day", bars, benchmark=bench, context_bars={"1Week": wk},
                              use_quant=False)
    close = full.evidence.close_time.iloc[cut-1]
    prefix = run_pattern_engine("S", "1Day", bars.iloc[:cut], benchmark=bench.iloc[:cut],
                                context_bars={"1Week": truncated(wk, close, "1Week")}, use_quant=False)
    mine, theirs = normalized(prefix.results), normalized(full.results)
    assert mine and all(theirs[key] == value for key, value in mine.items())
    # Future context rows are ignored too: same prefix, full weekly/benchmark inputs.
    loose = run_pattern_engine("S", "1Day", bars.iloc[:cut], benchmark=bench,
                               context_bars={"1Week": wk}, use_quant=False)
    assert normalized(loose.results) == mine


def test_engine_prefix_invariance_with_quant_models():
    bars = daily(260)
    cfg = QuantContextConfig(min_train=150, refit_every=40, cp_window=60, complexity_window=50)
    full = run_pattern_engine("S", "1Day", bars, quant_config=cfg)
    reports = prefix_check(full, lambda stamp: run_pattern_engine(
        "S", "1Day", bars.loc[bars.timestamp <= stamp], quant_config=cfg), [170, 235])
    assert [r["compared"] for r in reports] == [171*14, 236*14]
    check_results(full)


def test_quant_context_deterministic_and_prefix_equal():
    bars = daily(260)
    cfg = QuantContextConfig(min_train=150, refit_every=40, change_points=False, complexity=False)
    a, b = build_quant_context(bars, cfg), build_quant_context(bars, cfg)
    pd.testing.assert_frame_equal(a, b)
    prefix = build_quant_context(bars.iloc[:210], cfg)
    pd.testing.assert_frame_equal(prefix, a.iloc[:210], check_dtype=False)
    assert a.markov_p_high.notna().any() and a.hmm_p_high.notna().any()


def test_quant_context_uses_only_validated_two_state_models():
    for name in MODELS_USED:
        assert not model_status(name).experimental, name
    assert "hmm3" not in MODELS_USED and "markov3" not in MODELS_USED


def test_external_context_respects_available_at():
    closes = pd.Series([pd.Timestamp("2025-03-03 16:00", tz=NY).tz_convert("UTC"),
                        pd.Timestamp("2025-03-04 16:00", tz=NY).tz_convert("UTC")])
    vix = pd.DataFrame({"observation_date": pd.to_datetime(["2025-02-28", "2025-03-03"], utc=True),
                        "vix": [15.0, 30.0], "vix_zscore_252d": [0.1, 3.0], "vix_change_5d": [0.0, 15.0],
                        "available_at": [pd.Timestamp("2025-02-28 20:00", tz=NY).tz_convert("UTC"),
                                         pd.Timestamp("2025-03-03 20:00", tz=NY).tz_convert("UTC")]})
    out = build_external_context(closes, "T", vix=vix)
    # Monday's 16:00 close cannot see Monday's VIX (available 20:00); Tuesday can.
    assert out.vix.tolist() == [15.0, 30.0]
    assert (pd.to_datetime(out.vix_available_at, utc=True) <= closes).all()


def test_missing_external_context_has_no_artificial_values():
    closes = pd.Series(pd.date_range("2025-01-06 21:00", periods=3, freq="D", tz="UTC"))
    out = build_external_context(closes, "T")
    assert out.vix.isna().all() and set(out.macro_status) == {"missing"}
    assert set(out.news_quality) == {"missing"}


def test_stale_breakout_is_not_current():
    events = pd.DataFrame([{"bar_index": 2, "state": "breakout_confirmed", "direction": "up"}])
    recency = breakout_recency(12, events, max_age=5)
    assert recency.breakout_state.iloc[2:8].eq("breakout_confirmed").all()
    assert recency.breakout_state.iloc[8:].eq("inactive").all()


def test_partial_weekly_context_is_never_used():
    bars = daily(78)  # 15 full weeks plus Monday-Wednesday
    end = bars.timestamp.iloc[-1] + pd.Timedelta(hours=20)  # Wednesday closed, week not over
    wk = weekly(bars, end.isoformat())
    assert wk.is_complete.iloc[:-1].all() and not wk.is_complete.iloc[-1]
    run = run_pattern_engine("S", "1Day", bars, context_bars={"1Week": wk}, use_quant=False)
    used = pd.to_datetime(run.evidence.w1_close_time.dropna(), utc=True)
    complete = pd.to_datetime(wk.loc[wk.is_complete, "source_last_session_close"], utc=True)
    assert set(used) <= set(complete)
    assert (used <= run.evidence.close_time.loc[used.index]).all()
    partial = wk.loc[~wk.is_complete, "source_last_session_close"]
    assert not set(pd.to_datetime(partial, utc=True)) & set(used)


def test_breakout_episodes_use_levels_known_before_the_event():
    run = run_pattern_engine("S", "1Day", daily(), use_quant=False)
    breakouts = run.events.loc[run.events.pattern_family.isin(["breakout", "failed_breakout"])]
    assert not breakouts.empty
    asof = pd.to_datetime(breakouts.level_asof_timestamp, utc=True)
    assert (asof < pd.to_datetime(breakouts.first_detected_at, utc=True)).all()
    check_results(run)
