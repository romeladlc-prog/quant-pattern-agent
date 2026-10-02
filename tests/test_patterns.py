"""Phase 7 synthetic scenarios: every pattern family, evidence for/against, context rules."""
import json

import pandas as pd
import pytest

from pattern_rows import event, make_rows, run, span, states
from src.patterns.base import COMPONENT_WEIGHTS
from src.patterns.breakout import BreakoutDetector, FailedBreakoutDetector
from src.patterns.config import GAP_FILTER, MEAN_REVERSION_PERSISTENT_CAP, PATTERN_CATALOG
from src.patterns.divergence import DivergenceDetector
from src.patterns.engine import run_pattern_engine
from src.patterns.evidence import filtered_gaps
from src.patterns.exhaustion import ExhaustionDetector
from src.patterns.external_context import build_external_context
from src.patterns.mean_reversion import MeanReversionDetector
from src.patterns import regime_transition
from src.patterns.regime_transition import RegimeTransitionDetector
from src.patterns.trend_continuation import TrendContinuationDetector
from src.structure.swings import atr
from synthetic import anchored, bars_from_close

EXH_UP = dict(zscore_20=2.2, accel_up_recent=True, slope_change=-0.01, exhaustion_candidate=True,
              volatility_expansion=True, relative_volume=1.8, resistance_dist_atr=0.5)
EXH_DOWN = dict(zscore_20=-2.2, accel_down_recent=True, slope_change=0.01, exhaustion_candidate=True,
                volatility_expansion=True, relative_volume=1.8, support_dist_atr=0.5)


def exhaustion_rows(flags, extreme_key, extreme_value, revert_z, n=22):
    o = span({}, 10, 15, **flags)
    o[10][extreme_key] = extreme_value
    span(o, 11, 15, zscore_20=flags["zscore_20"]*0.7)
    span(o, 15, n, **{**flags, "zscore_20": revert_z})
    return make_rows(n, o)


def test_bearish_exhaustion_lifecycle_and_evidence_against():
    rows = exhaustion_rows(EXH_UP, "high", 103.0, -0.1)
    for row in rows:
        row.update(htf_structural_state="uptrend", relative_return_20=0.05, ratio_slope=0.01)
    results = run(ExhaustionDetector, "bearish", rows)
    assert states(results)[9:16] == ["inactive", "candidate", "candidate", "developing",
                                     "developing", "developing", "confirmed"]
    first = results[10]
    assert first.pattern_name == "bearish_exhaustion"
    assert {"extended_vs_mean", "prior_acceleration", "deceleration"} <= set(first.required_conditions_met)
    assert {"higher_tf_trend_intact", "rs_still_strong"} <= set(first.evidence_against)
    assert {"near_opposing_level", "volatility_expanding"} <= set(first.evidence_for)
    assert first.first_detected_at == rows[10]["timestamp"]
    assert results[15].confirmed_at == rows[15]["timestamp"] and results[15].bars_active == 6


def test_bullish_exhaustion_mirror():
    rows = exhaustion_rows(EXH_DOWN, "low", 97.0, 0.1)
    results = run(ExhaustionDetector, "bullish", rows)
    assert results[10].state == "candidate" and results[15].state == "confirmed"
    assert run(ExhaustionDetector, "bearish", rows)[10].state == "inactive"


def test_exhaustion_requires_convergence_not_one_flag():
    rows = make_rows(15, span({}, 5, 15, zscore_20=3.0))  # extended only
    assert set(states(run(ExhaustionDetector, "bearish", rows))) == {"inactive"}


def breakout_rows(expansion=True):
    rows = make_rows(12, span({}, 5, 12, close=101.5, high=102.0,
                              volatility_expansion=expansion, relative_volume=1.4,
                              relative_return_20=0.02, rolling_slope_20=0.3))
    event(rows, 5, "breakout_candidate", "up")
    event(rows, 6, "breakout_confirmed", "up", asof_offset=2)  # same level as the candidate
    return rows


def test_breakout_confirmed_with_expansion_and_weak():
    results = run(BreakoutDetector, "bullish", breakout_rows())
    assert states(results)[4:7] == ["inactive", "candidate", "confirmed"]
    assert results[6].subtype == "breakout_with_expansion"
    assert results[5].subtype == "breakout_under_confirmation"
    assert results[6].details["level_asof_timestamp"] < results[6].details["breakout_timestamp"]
    weak = run(BreakoutDetector, "bullish", breakout_rows(expansion=False))
    assert weak[6].state == "confirmed" and weak[6].subtype == "weak_breakout"


def test_breakout_requires_level_known_before_event():
    rows = make_rows(10, span({}, 5, 10, close=101.5))
    event(rows, 5, "breakout_candidate", "up", asof_offset=0)  # level as-of the same bar
    assert set(states(run(BreakoutDetector, "bullish", rows))) == {"inactive"}


def test_breakout_invalidated_by_fake_and_failed_breakout_recorded():
    o = span({}, 5, 7, close=101.5)
    span(o, 7, 12, close=99.5, low=98.5)
    o[8] = {"close": 99.0}
    o[9] = {"close": 98.8}
    rows = make_rows(12, o)
    event(rows, 5, "breakout_candidate", "up")
    event(rows, 7, "fake_breakout_up", "up", failure_bars=2, asof_offset=3)
    up = run(BreakoutDetector, "bullish", rows)
    assert up[7].state == "invalidated"
    failed = run(FailedBreakoutDetector, "bearish", rows)
    assert states(failed)[6:10] == ["inactive", "candidate", "developing", "confirmed"]
    d = failed[9].details
    assert d["level"] == 100.0 and d["bars_to_failure"] == 2
    assert d["breakout_timestamp"] == rows[5]["timestamp"]
    assert d["failure_timestamp"] == rows[7]["timestamp"]
    assert d["level_asof_timestamp"] == rows[4]["timestamp"]
    assert {"breakout_distance_atr", "relative_volume", "volatility_context"} <= set(d)


def test_failed_breakout_invalidated_when_level_reclaimed():
    rows = make_rows(10, span({}, 7, 10, close=99.5))
    rows[8]["close"] = 100.5  # back above the level by > 0.25 ATR
    event(rows, 7, "fake_breakout_up", "up", failure_bars=2)
    assert run(FailedBreakoutDetector, "bearish", rows)[8].state == "invalidated"


def test_bullish_mean_reversion_confirms_on_reversion():
    o = span({}, 10, 13, zscore_20=-2.5, slope_change=0.01, support_dist_atr=0.5)
    span(o, 13, 16, zscore_20=-0.8, slope_change=0.01, support_dist_atr=0.5)
    results = run(MeanReversionDetector, "bullish", make_rows(16, o))
    assert results[10].state == "candidate" and results[13].state == "confirmed"


def test_mean_reversion_capped_when_trend_and_regime_persistent():
    o = span({}, 10, 20, zscore_20=-2.5, slope_change=0.01, support_dist_atr=0.5,
             structural_state="downtrend", hurst=0.7)
    span(o, 14, 20, zscore_20=-0.5)
    results = run(MeanReversionDetector, "bullish", make_rows(20, o))
    assert all(r.score <= MEAN_REVERSION_PERSISTENT_CAP for r in results[10:])
    assert "capped_trend_and_regime_persistent" in results[10].notes
    assert "confirmed" not in states(results)
    assert {"trend_persistent_structure", "regime_persistent"} <= set(results[10].evidence_against)


def test_hurst_and_entropy_never_activate_alone():
    rows = make_rows(15, span({}, 0, 15, zscore_20=-1.0, hurst=0.3, permutation_entropy=0.99))
    for direction in ("bullish", "bearish"):
        assert set(states(run(MeanReversionDetector, direction, rows))) == {"inactive"}


def test_mean_reversion_requires_exhaustion_evidence():
    rows = make_rows(15, span({}, 5, 15, zscore_20=-2.6, slope_change=-0.02,
                              acceleration_state="accelerating_down"))
    assert set(states(run(MeanReversionDetector, "bullish", rows))) == {"inactive"}


@pytest.mark.parametrize("direction, state, slope, z, confirm_close, break_close", [
    ("bullish", "uptrend", 0.5, -0.5, 106.0, 94.0),
    ("bearish", "downtrend", -0.5, 0.3, 94.0, 106.0)])
def test_trend_continuation(direction, state, slope, z, confirm_close, break_close):
    base = dict(structural_state=state, rolling_slope_20=slope, zscore_20=z, last_high=105.0, last_low=95.0)
    rows = make_rows(12, span({}, 5, 12, **base))
    rows[9]["close"] = confirm_close
    results = run(TrendContinuationDetector, direction, rows)
    assert results[5].state == "candidate" and results[9].state == "confirmed"
    rows = make_rows(12, span({}, 5, 12, **base))
    rows[8]["close"] = break_close
    assert run(TrendContinuationDetector, direction, rows)[8].state == "invalidated"


def regime_rows():
    o = span({}, 0, 5, hmm_p_high=0.2, hmm_p_high_lag=0.2, markov_p_high=0.2, markov_p_high_lag=0.2)
    span(o, 5, 12, hmm_p_high=0.7, hmm_p_high_lag=0.2, markov_p_high=0.65, markov_p_high_lag=0.2,
         cp_rv20_recent=True)
    return make_rows(12, o)


def test_regime_transition_candidate_then_confirmed():
    results = run(RegimeTransitionDetector, "to_high_vol", regime_rows())
    assert results[5].state == "candidate" and results[5].subtype == "regime_transition_candidate"
    assert results[8].state == "confirmed" and results[8].subtype == "regime_transition_confirmed"
    assert results[8].details["probabilities_at_detection"]["hmm2_p_high"] == 0.7
    assert set(states(run(RegimeTransitionDetector, "to_low_vol", regime_rows()))) == {"inactive"}


def test_regime_transition_requires_validated_models(monkeypatch):
    monkeypatch.setattr(regime_transition, "validated_models", lambda purpose: ["hmm3", "markov3"])
    assert set(states(run(RegimeTransitionDetector, "to_high_vol", regime_rows()))) == {"inactive"}


def divergence_rows(n=40, bearish=True):
    key, sign = ("high", 1) if bearish else ("low", -1)
    o = {8: {key: 100 + 5*sign, "momentum_20": 0.10*sign},
         18: {key: 100 + 6*sign, "momentum_20": 0.05*sign},
         12: {("low" if bearish else "high"): 100 - 3*sign}}
    o[20] = {f"new_swing_{key}": True, f"last_{key}_label": "HH" if bearish else "LL",
             f"last_{key}_index": 18, f"prev_{key}_index": 8}
    span(o, 21, n, **{f"last_{key}_label": "HH" if bearish else "LL", f"last_{key}_index": 18,
                      f"prev_{key}_index": 8})
    return make_rows(n, o)


@pytest.mark.parametrize("bearish", [True, False])
def test_divergence_alone_never_confirms(bearish):
    direction = "bearish" if bearish else "bullish"
    results = run(DivergenceDetector, direction, divergence_rows(bearish=bearish))
    assert results[20].state == "candidate"
    assert "momentum_divergence" in results[20].evidence_for
    assert "confirmed" not in states(results)
    assert "expired" in states(results)


@pytest.mark.parametrize("bearish", [True, False])
def test_divergence_confirms_with_neckline_break(bearish):
    rows = divergence_rows(bearish=bearish)
    rows[23]["close"] = 96.5 if bearish else 103.5
    results = run(DivergenceDetector, "bearish" if bearish else "bullish", rows)
    assert results[23].state == "confirmed"


def test_missing_context_is_excluded_not_zero():
    rows = exhaustion_rows(EXH_UP, "high", 103.0, -0.1)
    result = run(ExhaustionDetector, "bearish", rows)[10]
    assert result.score_context is None and result.external_context == {"status": "missing"}
    components = {c: getattr(result, f"score_{c}") for c in COMPONENT_WEIGHTS}
    available = {c: v for c, v in components.items() if v is not None}
    expected = sum(COMPONENT_WEIGHTS[c]*v for c, v in available.items())/sum(
        COMPONENT_WEIGHTS[c] for c in available)
    assert result.score == pytest.approx(expected, abs=1e-6)
    assert any(note.startswith("not_evaluable:") and "vix_extreme" in note for note in result.notes)


def test_unavailable_macro_and_news_coverage():
    closes = pd.Series(pd.date_range("2025-03-03 21:00", periods=5, freq="D", tz="UTC"))
    unsafe = pd.DataFrame({"series": "fed_assets", "observation_date": closes - pd.Timedelta(days=30),
                           "release_date": pd.NaT, "available_at": closes - pd.Timedelta(days=20),
                           "value": 1.0, "unit": "x", "source": "WALCL", "asof_safe": False})
    out = build_external_context(closes, "T", macro_observations=unsafe)
    assert set(out.macro_status) == {"excluded_not_asof_safe"}
    assert out.macro_liquidity_state.isna().all()
    news = pd.DataFrame({"ticker": "T", "published_at": closes - pd.Timedelta(hours=2),
                         "available_at": closes - pd.Timedelta(hours=1), "headline": "x",
                         "summary": "", "source": "", "score": [0.5, None, None, None, None],
                         "scoring_method": "v1"})
    out = build_external_context(closes, "T", news=news)
    assert set(out.news_quality) == {"insufficient_coverage"}
    assert out.news_sentiment_7d.isna().all()


def test_gaps_are_auxiliary_low_weight_and_filtered():
    for family, catalog in PATTERN_CATALOG.items():
        assert not any("gap" in name for name in catalog["required"]), family
    bars = bars_from_close([100, 100.3, 103, 103.1, 108, 108.2, 108.3] * 5,
                           opens=[100, 100.3, 103.6, 103.1, 108.9, 108.2, 108.3] * 5)
    gaps, meta = filtered_gaps(bars, atr(bars))
    assert meta["gaps_detected"] == meta["gaps_kept"] + meta["gaps_discarded"]
    assert meta["gaps_discarded"] > 0
    rows = exhaustion_rows(EXH_UP, "high", 103.0, -0.1)
    for row in rows:
        row["gap_up_recent"] = True
    result = run(ExhaustionDetector, "bearish", rows)[10]
    assert "exhaustion_gap" in result.optional_conditions_met
    assert GAP_FILTER["weight"] < 1


def test_engine_breakout_uses_asof_level_end_to_end():
    bars = anchored({0: 100, 5: 110, 10: 100, 15: 110, 20: 100, 25: 116, 30: 105, 40: 110.6, 45: 100})
    result = run_pattern_engine("SYN", "1Day", bars, use_quant=False)
    up = result.events.loc[result.events.pattern.eq("bullish_breakout")]
    assert not up.empty
    first = up.iloc[0]
    details = json.loads(first.details)
    assert details["level"] == pytest.approx(110.2)
    assert pd.Timestamp(first.level_asof_timestamp) < pd.Timestamp(first.first_detected_at)


def test_scores_bounded_and_named_as_convergence():
    rows = exhaustion_rows(EXH_UP, "high", 103.0, -0.1)
    for result in run(ExhaustionDetector, "bearish", rows):
        assert 0 <= result.score <= 1 and result.score_kind == "convergence_score"
