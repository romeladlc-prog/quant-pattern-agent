"""Regression tests for the Phase 7 live prefix failure (ARM 1Day, T=2025-03-25).

Root cause: context timeframes with no closed bar by T (live 4Hour/1Hour
downloads start 400/90 days back) were skipped in the prefix run but attached
in the full run, so ``mtf_context`` gained "4Hour"/"1Hour" keys for every past
bar once future intraday data existed. The fixture below reproduces that
layout offline: daily history from the start, intraday context only at the end.
"""
import logging
import warnings

import numpy as np
import pandas as pd

from src.patterns.checks import normalized, prefix_check
from src.patterns.diagnostics import diagnose_prefix, format_report
from src.patterns.engine import run_pattern_engine
from src.patterns.evidence import bar_close_times
from src.patterns.quant_context import QuantContextConfig, build_quant_context
from validate_patterns import synthetic_intraday, weekly_from_daily

NY = "America/New_York"


def daily(n=300, seed=5, df=None):
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(df, n) if df else rng.standard_normal(n)
    close = 100*np.exp(np.cumsum(0.02*shocks))
    opens = close*np.exp(0.004*rng.standard_normal(n))
    spread = np.abs(0.01*rng.standard_normal(n))*close
    stamps = pd.bdate_range("2024-01-01", periods=n).tz_localize(NY).tz_convert("UTC")
    return pd.DataFrame({"timestamp": stamps, "open": opens, "high": np.maximum(opens, close)+spread,
                         "low": np.minimum(opens, close)-spread, "close": close,
                         "volume": rng.uniform(5e5, 2e6, n)})


def late_context(bars):
    """Weekly over the whole sample; 4Hour/1Hour only for the last ~15%."""
    end = (bars.timestamp.iloc[-1] + pd.Timedelta(days=1)).isoformat()
    late = bars.timestamp.iloc[int(len(bars)*0.85)].strftime("%Y-%m-%d")
    return {"1Week": weekly_from_daily(bars, end),
            "4Hour": synthetic_intraday(bars, late, 1, "4Hour"),
            "1Hour": synthetic_intraday(bars, late, 2, "1Hour")}


def truncate(frame, stamp, timeframe):
    complete = frame.loc[frame.is_complete.astype(bool)] if "is_complete" in frame else frame
    complete = complete.reset_index(drop=True)
    return complete.loc[(bar_close_times(complete, timeframe) <= stamp).to_numpy()].reset_index(drop=True)


def prefix_runner(bars, context, **kwargs):
    full = run_pattern_engine("S", "1Day", bars, context_bars=context, **kwargs)

    def run(stamp):
        close = full.evidence.close_time.loc[full.evidence.timestamp.eq(stamp)].iloc[0]
        return run_pattern_engine("S", "1Day", truncate(bars, close, "1Day"),
                                  context_bars={tf: truncate(f, close, tf) for tf, f in context.items()},
                                  **kwargs)
    return full, run


def test_late_intraday_context_passes_prefix_check():
    bars = daily()
    context = late_context(bars)
    full, run = prefix_runner(bars, context, use_quant=False)
    # 150: no intraday bar closed yet (the live failure); 280: partial intraday history.
    reports = prefix_check(full, run, [150, 280])
    assert [r["compared"] for r in reports] == [151*14, 281*14]


def test_mtf_context_schema_does_not_depend_on_future_data():
    bars = daily()
    context = late_context(bars)
    full, run = prefix_runner(bars, context, use_quant=False)
    prefix = run(full.evidence.timestamp.iloc[150])
    assert prefix.meta["context_timeframes"] == ["1Week", "4Hour", "1Hour"]
    assert prefix.meta["context_timeframes_with_data"] == ["1Week"]
    for result in prefix.results:
        assert set(result.mtf_context) == {"1Week", "4Hour", "1Hour"}
        assert all(v is None for v in result.mtf_context["1Hour"].values())
    assert set(c for c in full.evidence if c.startswith(("h4_", "h1_"))) == \
        set(c for c in prefix.evidence if c.startswith(("h4_", "h1_")))


def test_debug_prefix_names_the_first_divergent_input():
    """The pre-fix behaviour (timeframe dropped in the prefix run) is diagnosed
    as a schema divergence of the 4Hour context at bar 0."""
    bars = daily()
    context = late_context(bars)
    full = run_pattern_engine("S", "1Day", bars, context_bars=context, use_quant=False)
    old_prefix = run_pattern_engine("S", "1Day", bars.iloc[:151],
                                    context_bars={"1Week": context["1Week"]}, use_quant=False)
    report = diagnose_prefix(full, old_prefix)
    assert report["differing_results"] == report["compared"] == 151*14
    assert report["first_bar"] == 0
    first = report["first_input"]
    assert first["column"].startswith("h4_") and first["kind"].startswith("schema")
    assert first["source"] == "4Hour context (h4_*)"
    fields = report["fields_at_first_bar"]["bullish_exhaustion"]
    assert fields == ["mtf_context.1Hour (only in full)", "mtf_context.4Hour (only in full)"]
    assert "first divergent input: h4_" in format_report(report)
    fixed = run_pattern_engine("S", "1Day", bars.iloc[:151], use_quant=False,
                               context_bars={tf: f.iloc[:0] for tf, f in context.items()} |
                               {"1Week": context["1Week"]})
    assert diagnose_prefix(full, fixed)["differing_results"] == 0


QUANT = QuantContextConfig(min_train=150, refit_every=40, change_points=False, complexity=False)


def test_hmm_em_non_convergence_is_flagged_and_prefix_stable(caplog):
    # Student-t(2.5) returns, seed 1: the selected HMM2 at refit 169 has an EM
    # log-likelihood decrease (hmmlearn "Model is not converging").
    bars = daily(300, seed=1, df=2.5)
    with caplog.at_level(logging.WARNING), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quant = build_quant_context(bars, QUANT)
    assert not [r for r in caplog.records if r.name.startswith("hmmlearn")]
    bad = [d for d in quant.attrs["fit_diagnostics"] if not d["converged"]]
    assert [(d["refit"], d["model"]) for d in bad] == [(169, "hmm2")]
    assert any(seed["loglik_decreases"] > 0 for seed in bad[0]["seeds"])
    segment = quant.loc[169:208, "hmm_fit_converged"]
    assert segment.eq(False).all() and quant.loc[209:, "hmm_fit_converged"].eq(True).all()
    assert quant.loc[:168, "hmm_fit_converged"].isna().all()
    # The flagged fit gives the same values with data up to T as with the full sample.
    for cut in (190, 230):
        prefix = build_quant_context(bars.iloc[:cut], QUANT)
        pd.testing.assert_frame_equal(prefix, quant.iloc[:cut], check_dtype=False)
        assert prefix.attrs["fit_diagnostics"] == [d for d in quant.attrs["fit_diagnostics"]
                                                   if d["refit"] < cut]


def test_non_converged_fit_is_visible_in_pattern_results():
    bars = daily(300, seed=1, df=2.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = run_pattern_engine("S", "1Day", bars, quant_config=QUANT)
        reports = prefix_check(full, lambda stamp: run_pattern_engine(
            "S", "1Day", bars.loc[bars.timestamp <= stamp], quant_config=QUANT), [190])
    assert reports[0]["compared"] == 191*14
    flagged = [r for r in full.results if 169 <= r.bar_index <= 208]
    assert flagged and all(r.regime_context["hmm_fit_converged"] is False for r in flagged)
    assert all(any(n.startswith("model_not_converged:") and "hmm2" in n for n in r.notes)
               for r in flagged)
    clean = [r for r in full.results if r.bar_index >= 209]
    assert all(not any(n.startswith("model_not_converged") for n in r.notes) for r in clean)
    assert normalized(full.results)


def test_hmm2_fit_is_bit_reproducible():
    """Second live-like divergence (hidden behind the first): sklearn KMeans
    inside hmmlearn's init is not bit-reproducible with several OpenMP threads,
    so the same training rows gave HMM2 parameters differing in the last bits
    and prefix/full p_high differing by ~1e-14 from the refit bar on."""
    from src.features.statistical_features import calculate_statistical_features
    from src.models.regime_models import fit_hmm_regimes
    # Same rows as the live-like NVDA 1Day run where it showed at refit 586.
    features = calculate_statistical_features(daily(650, seed=1, df=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = [fit_hmm_regimes(features.iloc[:587], features.iloc[:end], 2) for end in (601, 649)*4]
    reference = fits[0]
    for fit in fits[1:]:
        assert np.array_equal(fit["model"].covars_, reference["model"].covars_)
        assert np.array_equal(fit["model"].means_, reference["model"].means_)
        rows = len(reference["probabilities"])
        assert np.array_equal(fit["probabilities"].to_numpy()[:rows], reference["probabilities"].to_numpy())
