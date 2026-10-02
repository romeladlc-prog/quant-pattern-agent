"""Validated role of each Phase 4 model, from the Phase 4.1 audit.

Metadata only: it does not block fitting any model. Later layers must read it
before treating a model output as a validated component. Evidence and limits
are in docs/MODEL_DECISIONS.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelStatus:
    name: str
    family: str
    role: str
    benchmark_only: bool = False
    stable_for_current_use: bool = False
    experimental: bool = False
    warning: str | None = None
    validated_for: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return asdict(self)


MODEL_STATUS = {status.name: status for status in (
    ModelStatus("autoreg", "return", "benchmark_return", benchmark_only=True,
                warning="no_stable_edge_over_zero_return"),
    ModelStatus("arima", "return", "benchmark_return", benchmark_only=True,
                warning="no_stable_edge_over_zero_return"),
    ModelStatus("garch", "volatility", "primary_volatility",
                stable_for_current_use=True, validated_for=("volatility_forecast",)),
    ModelStatus("egarch", "volatility", "complementary_volatility",
                stable_for_current_use=True, validated_for=("volatility_forecast",)),
    ModelStatus("gjr_garch", "volatility", "possibly_redundant_volatility",
                warning="close_to_garch_fit_failure_avgo"),
    ModelStatus("har_rv", "volatility", "benchmark_volatility", benchmark_only=True),
    ModelStatus("kalman_local_level", "state_space", "stable_level_state",
                stable_for_current_use=True, validated_for=("level",)),
    ModelStatus("kalman_local_linear_trend", "state_space", "slope_description",
                warning="boundary_estimates_57pct_less_stable",
                validated_for=("slope_description",)),
    ModelStatus("hmm2", "regime", "regime_description", stable_for_current_use=True,
                validated_for=("regime_description",)),
    ModelStatus("markov2", "regime", "regime_description", stable_for_current_use=True,
                validated_for=("regime_description",)),
    ModelStatus("hmm3", "regime", "regime_description", experimental=True,
                warning="unstable_across_assets"),
    ModelStatus("markov3", "regime", "regime_description", experimental=True,
                warning="unstable_across_assets"),
    ModelStatus("pelt", "change_point", "retrospective_description",
                warning="retrospective_not_causal",
                validated_for=("rv20_change_points", "kalman_slope_change_points")),
    ModelStatus("hurst", "complexity", "descriptive_feature",
                validated_for=("descriptive_feature",)),
    ModelStatus("shannon_entropy", "complexity", "descriptive_feature",
                validated_for=("descriptive_feature",)),
    ModelStatus("permutation_entropy", "complexity", "descriptive_feature",
                validated_for=("descriptive_feature",)),
)}


def model_status(name: str) -> ModelStatus:
    try:
        return MODEL_STATUS[name]
    except KeyError as error:
        raise KeyError(f"Modelo sin estado registrado: {name}") from error


def validated_models(purpose: str) -> list[str]:
    """Stable, non-experimental, non-benchmark models validated for a purpose."""
    return [s.name for s in MODEL_STATUS.values()
            if purpose in s.validated_for and s.stable_for_current_use
            and not s.experimental and not s.benchmark_only]


def allowed_models(purpose: str) -> list[str]:
    """Models usable for a purpose, including those with a documented warning."""
    return [s.name for s in MODEL_STATUS.values()
            if purpose in s.validated_for and not s.experimental]
