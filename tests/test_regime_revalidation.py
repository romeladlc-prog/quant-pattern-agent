"""Offline tests for the Phase 4.1 regime revalidation (no Alpaca, no FRED)."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src.features.statistical_features import calculate_statistical_features
from src.models.regime_models import fit_hmm_regimes, fit_markov_regimes
from src.validation.regime_revalidation import (FIT_COLUMNS, LEGACY_LABEL, STATUSES, TOLERANCES,
                                                RevalidationConfig, cut_plan, fit_once,
                                                reproducibility_row)
from src.validation.walk_forward import WalkForwardConfig, walk_forward_splits
from validate_regime_reproducibility import run, synthetic_bars

NY = "America/New_York"


def bars(n=300, seed=1, df=None):
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(df, n) if df else rng.standard_normal(n)
    close = 100*np.exp(np.cumsum(0.02*shocks))
    stamps = pd.bdate_range("2024-01-01", periods=n).tz_localize(NY).tz_convert("UTC")
    return pd.DataFrame({"timestamp": stamps, "open": close, "high": close*1.01, "low": close*.99,
                         "close": close, "volume": 1e6})


@pytest.fixture(scope="module")
def features():
    return calculate_statistical_features(bars(320, seed=7))


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def assert_same_fit(a, b, params):
    np.testing.assert_allclose(a["probabilities"].to_numpy(), b["probabilities"].to_numpy(),
                               rtol=0, atol=TOLERANCES["probability_atol"])
    np.testing.assert_allclose(params(a), params(b), rtol=0, atol=TOLERANCES["parameter_atol"])
    assert a["states"].equals(b["states"])


def test_hmm2_same_input_and_configuration_is_reproducible():
    # Rows where several OpenMP threads in KMeans gave different fits (Phase 7.1).
    features = calculate_statistical_features(bars(650, seed=1, df=3))
    fits = [quiet(fit_hmm_regimes, features.iloc[:587], features.iloc[:601], 2) for _ in range(6)]
    for fit in fits[1:]:
        assert_same_fit(fits[0], fit, lambda f: np.r_[f["model"].means_.ravel(), f["model"].covars_.ravel(),
                                                      f["model"].transmat_.ravel()])
        assert fit["train_loglik"] == pytest.approx(fits[0]["train_loglik"], rel=TOLERANCES["loglik_rtol"])
        assert fit["fit_quality"] == fits[0]["fit_quality"]


def test_markov2_same_input_and_rng_is_reproducible(features):
    fits = [quiet(fit_markov_regimes, features.iloc[:250], features, 2, random_state=42)
            for _ in range(3)]
    for fit in fits[1:]:
        assert_same_fit(fits[0], fit, lambda f: np.asarray(f["result"].params, float))
        assert fit["result"].llf == pytest.approx(fits[0]["result"].llf, rel=TOLERANCES["loglik_rtol"])
        assert fit["converged"] == fits[0]["converged"]


def test_repeats_are_reported_reproducible(features):
    cut = {"cut": "walk_forward", "fold": 0, "train_stop": 250, "scope_stop": 260}
    for model in ("HMM", "Markov"):
        runs = [fit_once("S", model, 2, features, cut, 42, f"repeat_{k}") for k in range(3)]
        row = reproducibility_row(runs)
        assert row["reproducible"] and row["max_abs_probability_diff"] <= TOLERANCES["probability_atol"]


def test_hmm_strict_convergence_flag():
    """Student-t(2.5), seed 1, first 170 rows: hmmlearn says converged, but the
    selected EM run lowered its log-likelihood (``Model is not converging``)."""
    feats = calculate_statistical_features(bars(300, seed=1, df=2.5))
    cut = {"cut": "walk_forward", "fold": 0, "train_stop": 170, "scope_stop": 171}
    row = fit_once("S", "HMM", 2, feats, cut, 42, "repeat_0").row
    assert row["status"] == "not_converged"
    assert row["monitor_converged"] is True and row["loglik_decreases"] >= 1
    assert not row["hit_max_iter"]
    clean = fit_once("S", "HMM", 2, feats, {**cut, "train_stop": 210, "scope_stop": 211}, 42, "r").row
    assert clean["status"] == "converged" and clean["loglik_decreases"] == 0


def test_failed_fit_is_recorded_not_replaced(features):
    cut = {"cut": "walk_forward", "fold": 0, "train_stop": 10, "scope_stop": 11}  # no valid rows
    runs = [fit_once("S", model, 2, features, cut, 42, "repeat_0") for model in ("HMM", "Markov")]
    for fit in runs:
        assert fit.row["status"] == "fit_failed" and fit.row["error"]
        assert fit.p_high is None and fit.row["p_high_final"] is None
    assert reproducibility_row([runs[0], runs[0]])["reproducible"]


def test_cut_plan_uses_the_phase4_1_training_rows(features):
    cfg = RevalidationConfig(min_train=150, step=40)
    plan = cut_plan(features.index, cfg)
    splits = list(walk_forward_splits(features.index, WalkForwardConfig(150, 40, 1)))
    walk = [c for c in plan if c["cut"] == "walk_forward"]
    assert [(c["train_stop"], c["scope_stop"]-1) for c in walk] == [(s.stop, p) for s, p in splits]
    assert all(s.start == 0 for s, _ in splits)  # expanding, as Phase 4.1
    legacy = [c for c in plan if c["cut"] == "legacy_last_cut"]
    assert legacy == [{"cut": "legacy_last_cut", "fold": -1,
                       "train_stop": max(150, len(features)-40), "scope_stop": len(features)}]
    for cut in walk:
        row = fit_once("S", "Markov", 2, features, cut, 42, "r").row
        assert row["train_end"] == features.index[cut["train_stop"]-1] < row["test_date"]
        assert row["test_date"] == features.index[cut["scope_stop"]-1]


def test_fold_results_do_not_use_rows_after_the_test_date(features):
    cut = {"cut": "walk_forward", "fold": 1, "train_stop": 220, "scope_stop": 221}
    truncated = features.iloc[:221]
    for model in ("HMM", "Markov"):
        full = fit_once("S", model, 2, features, cut, 42, "r")
        prefix = fit_once("S", model, 2, truncated, cut, 42, "r")
        assert full.row == prefix.row
        pd.testing.assert_series_equal(full.p_high, prefix.p_high, rtol=0,
                                       atol=TOLERANCES["probability_atol"])
        assert full.p_high.index.max() == features.index[220]


def test_output_schema_and_legacy_marking(tmp_path):
    legacy = tmp_path/"regimes.csv"
    pd.DataFrame([{"asset": "SYN_ARM", "model": "HMM2", "converged": True, "stable": True}]).to_csv(
        legacy, index=False)
    cfg = RevalidationConfig(min_train=150, step=60, repeats=2)
    out = tmp_path/"out"
    tables = run(synthetic_bars(260, ("ARM",)), cfg, out, "test", {"source": "synthetic"}, legacy)
    expected = {"regime_revalidation.csv", "regime_reproducibility.csv", "regime_seed_sensitivity.csv",
                "regime_fold_stability.csv", "convergence_summary.csv", "summary.md",
                "run_metadata.json", f"{LEGACY_LABEL}_regimes.csv"}
    assert expected <= {p.name for p in out.iterdir()}
    fits = pd.read_csv(out/"regime_revalidation.csv")
    assert list(fits.columns) == FIT_COLUMNS
    assert set(fits.status) <= set(STATUSES) and set(fits.model) == {"HMM2", "Markov2"}
    assert set(fits.run) == {"repeat_0", "repeat_1", "alt_seed"}
    # 2 walk-forward folds + legacy cut, 2 models, 2 repeats + alt seed.
    assert len(fits) == 3*2*3 and tables["reproducibility"].reproducible.all()
    meta = json.loads((out/"run_metadata.json").read_text())
    assert meta["config"]["tolerances"] == TOLERANCES and meta["hmm_openmp_threads"] == 1
    assert pd.read_csv(out/f"{LEGACY_LABEL}_regimes.csv").label.eq(LEGACY_LABEL).all()
    assert legacy.read_text().startswith("asset,model,converged,stable")  # original untouched
    assert "legacy_pre_revalidation" in (out/"summary.md").read_text()
