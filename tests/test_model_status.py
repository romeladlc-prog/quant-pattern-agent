"""Phase 4.1 decisions are machine-readable; 3-state fits run but are flagged."""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.models.model_status import allowed_models, model_status, validated_models


def test_decisions_are_encoded():
    assert model_status("autoreg").benchmark_only and model_status("arima").benchmark_only
    assert model_status("garch").role == "primary_volatility"
    assert model_status("egarch").role == "complementary_volatility"
    assert model_status("har_rv").role == "benchmark_volatility"
    for name in ("hmm2", "markov2"):
        assert model_status(name).stable_for_current_use
    for name in ("hmm3", "markov3"):
        status = model_status(name)
        assert status.experimental and status.warning == "unstable_across_assets"
        assert not status.stable_for_current_use


def test_kalman_purpose_is_explicit():
    assert validated_models("level") == ["kalman_local_level"]
    assert validated_models("slope_description") == []
    assert allowed_models("slope_description") == ["kalman_local_linear_trend"]
    assert model_status("kalman_local_linear_trend").warning


def test_future_layers_cannot_pick_benchmarks_or_experimental():
    assert validated_models("volatility_forecast") == ["garch", "egarch"]
    assert "hmm3" not in allowed_models("regime_description")
    assert set(validated_models("regime_description")) == {"hmm2", "markov2"}
    with pytest.raises(KeyError):
        model_status("lstm")


def regime_features(n=300, seed=1):
    rng = np.random.default_rng(seed)
    returns = np.r_[rng.normal(0, .01, n//2), rng.normal(0, .03, n - n//2)]
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame({"log_return": returns}, index=index)
    frame["realized_volatility_20"] = np.sqrt((frame.log_return**2).rolling(20).sum())
    frame["rolling_slope_20"] = frame.log_return.cumsum().rolling(20).mean().diff()
    return frame.dropna()


def test_three_state_hmm_runs_with_experimental_status():
    pytest.importorskip("hmmlearn")
    from src.models.regime_models import fit_hmm_regimes
    features = regime_features()
    with pytest.warns(UserWarning, match="experimental"):
        fit = fit_hmm_regimes(features, features, n_states=3)
    assert fit["model_status"]["experimental"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        two = fit_hmm_regimes(features, features, n_states=2)
    assert two["model_status"]["stable_for_current_use"]


def test_fit_outputs_carry_status():
    from src.models.state_space_models import fit_state_specification
    from src.models.time_series_models import fit_autoreg
    from src.models.volatility_models import fit_arch_families
    features = regime_features()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ar = fit_autoreg(features.log_return, max_lags=3)
        arch = fit_arch_families(features.log_return)
        level = fit_state_specification(100+features.log_return.cumsum(), "local level")
    assert ar["benchmark_only"] and ar["model_status"]["benchmark_only"]
    assert arch["GARCH"]["model_status"]["role"] == "primary_volatility"
    assert level["model_status"]["name"] == "kalman_local_level"
