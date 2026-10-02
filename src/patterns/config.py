"""Transparent, hand-set Pattern Engine parameters (Phase 7).

Nothing here was optimized on historical outcomes. Weights express how much
each piece of evidence counts toward a *convergence* score: the share of
weighted supporting evidence present, net of contradicting evidence. The
score is not a probability, an expected return, or a buy/sell signal.
"""
from __future__ import annotations

from dataclasses import dataclass

SCORE_NAME = "convergence_score"
COMPONENTS = ("structure", "volatility", "regime", "context", "momentum")

# pattern_score = sum(W_c * score_c) / sum(W_c) over components with evidence.
COMPONENT_WEIGHTS = {"structure": 0.30, "momentum": 0.25, "volatility": 0.20,
                     "regime": 0.15, "context": 0.10}

STATES = ("inactive", "candidate", "developing", "confirmed", "invalidated", "expired")

BREAKOUT_MAX_AGE_BARS = 5  # Phase 6.1 recency rule, reused for the per-bar state


@dataclass(frozen=True)
class StateMachineConfig:
    enter_score: float = 0.40      # score needed (with all required evidence) to open
    maintain_score: float = 0.25   # below this, a pending bar counts as a lapse
    confirm_score: float = 0.35    # minimum score on the confirming bar
    developing_after_bars: int = 2  # candidate -> developing after this many bars
    grace_bars: int = 2            # consecutive lapses tolerated before expiry
    max_pending_bars: int = 10     # candidate/developing older than this expires
    confirmed_ttl_bars: int = 5    # a confirmed episode closes after this many bars
    cooldown_bars: int = 3         # bars after a closed episode before a new one


# Event-driven families (breakout, failed breakout, regime transition,
# divergence) open on their causal event and confirm by their own causal rule,
# so their enter/confirm score floors are 0; the score still reports evidence.
STATE_MACHINES = {
    "exhaustion": StateMachineConfig(max_pending_bars=10),
    "breakout": StateMachineConfig(enter_score=0.0, confirm_score=0.0, max_pending_bars=3,
                                   confirmed_ttl_bars=10, developing_after_bars=1),
    "failed_breakout": StateMachineConfig(enter_score=0.0, confirm_score=0.0, max_pending_bars=6,
                                          confirmed_ttl_bars=10, developing_after_bars=1),
    "mean_reversion": StateMachineConfig(max_pending_bars=10),
    "trend_continuation": StateMachineConfig(max_pending_bars=15),
    "regime_transition": StateMachineConfig(enter_score=0.0, confirm_score=0.0,
                                            max_pending_bars=10, developing_after_bars=1),
    "divergence": StateMachineConfig(enter_score=0.0, confirm_score=0.0, max_pending_bars=15),
}

# Event evidence stays "present" for this many bars after the event (the
# Phase 6.1 breakout recency); episodes still open only on the event bar.
EVENT_EVIDENCE_BARS = BREAKOUT_MAX_AGE_BARS

THRESHOLDS = {
    "zscore_extended": 1.5,        # 20-bar close z-score for an extended move
    "zscore_extreme": 2.0,         # mean-reversion stretch
    "near_level_atr": 1.0,         # level within this many ATR(14)
    "inside_level_atr": 0.25,      # close this close to a level is "back at the level"
    "relative_volume_high": 1.5,
    "relative_volume_ok": 1.0,
    "atr_expansion": 1.25,
    "garch_ratio_high": 1.2,
    "garch_ratio_low": 0.8,
    "rv_percentile_high": 0.8,
    "hurst_persistent": 0.55,
    "hurst_antipersistent": 0.45,
    "permutation_entropy_high": 0.95,
    "regime_probability": 0.5,
    "regime_confirm_probability": 0.6,
    "regime_revert_probability": 0.4,
    "regime_confirm_bars": 3,
    "regime_shift": 0.2,           # |p_high - p_high_lag| for a moving regime
    "regime_low_probability": 0.3,
    "vix_zscore_high": 1.0,
    "vix_zscore_low": -1.0,
    "breadth_high": 0.6,
    "breadth_low": 0.4,
    "news_min_scored": 3,          # scored headlines in 7 days to use sentiment
    "recent_bars": 10,
    "structure_change_bars": 10,
}

# Phase 6 decision: gaps are noisy with IEX. Auxiliary evidence only, never
# required, low weight, and only gaps passing both size filters count.
GAP_FILTER = {"min_atr": 0.5, "min_pct": 0.005, "recent_bars": 3, "weight": 0.25}

# Evidence catalog: names a detector may emit. Weight 1.0 unless listed.
# "required" gates the state machine; "optional" and "contradictory" only score.
PATTERN_CATALOG = {
    "exhaustion": {
        "required": ["extended_vs_mean", "prior_acceleration", "deceleration"],
        "optional": ["near_opposing_level", "slope_deteriorating", "rs_weakening",
                     "structure_exhaustion_flag", "volatility_expanding", "relative_volume_high",
                     "garch_vol_rising", "change_point_recent", "high_vol_regime_rising",
                     "vix_extreme", "breadth_diverging", "exhaustion_gap"],
        "contradictory": ["higher_tf_trend_intact", "rs_still_strong", "active_breakout_with_move",
                          "low_vol_regime_stable", "hurst_persistent"],
    },
    "breakout": {
        "required": ["causal_breakout_event", "level_known_before_event"],
        "optional": ["strong_level", "prior_compression", "structure_aligned", "higher_tf_aligned",
                     "volatility_expansion", "relative_volume_ok", "rs_aligned", "slope_aligned",
                     "regime_stable", "vix_supportive", "breadth_supportive", "breakout_gap"],
        "contradictory": ["low_volume", "rs_against", "momentum_decelerating", "close_near_level",
                          "higher_tf_against", "vix_spike"],
    },
    "failed_breakout": {
        "required": ["causal_fake_breakout_event"],
        "optional": ["back_in_range", "strong_level", "momentum_deteriorating", "rs_weak",
                     "volume_on_failure", "volatility_expansion", "high_vol_regime_rising",
                     "vix_rising"],
        "contradictory": ["close_reclaims_level", "rs_strong", "higher_tf_aligned_with_breakout"],
    },
    "mean_reversion": {
        "required": ["extreme_zscore", "exhaustion_evidence"],
        "optional": ["near_opposing_level", "deceleration", "volatility_elevated",
                     "hurst_antipersistent", "entropy_high", "breadth_extreme", "vix_extreme"],
        "contradictory": ["trend_persistent_structure", "regime_persistent",
                          "higher_tf_trend_aligned", "still_accelerating"],
    },
    "trend_continuation": {
        "required": ["structure_trend", "slope_aligned", "pullback"],
        "optional": ["compression_recent", "higher_tf_aligned", "kalman_aligned", "rs_aligned",
                     "volatility_compatible", "regime_stable", "breadth_supportive", "vix_supportive"],
        "contradictory": ["rs_against", "volatility_expanding", "regime_shifting",
                          "higher_tf_against", "structure_warning"],
    },
    "regime_transition": {
        "required": ["validated_regime_models", "regime_probability_cross"],
        "optional": ["other_model_agrees", "change_point_rv20", "change_point_slope",
                     "rv20_moving", "garch_vol_moving", "structure_changed", "kalman_turn",
                     "vix_moving"],
        "contradictory": ["models_disagree", "rv20_against"],
    },
    "divergence": {
        "required": ["price_new_extreme", "any_indicator_divergence"],
        "optional": ["momentum_divergence", "slope_divergence", "rs_divergence",
                     "volume_divergence", "breadth_divergence", "near_opposing_level",
                     "deceleration", "volatility_expanding"],
        "contradictory": ["structure_trend_strong", "active_breakout_with_move", "rs_strong"],
    },
}

EVIDENCE_WEIGHTS = {
    "exhaustion": {"exhaustion_gap": GAP_FILTER["weight"], "hurst_persistent": 0.5,
                   "structure_exhaustion_flag": 0.5},
    "breakout": {"breakout_gap": GAP_FILTER["weight"], "level_known_before_event": 0.5},
    "mean_reversion": {"hurst_antipersistent": 0.5, "entropy_high": 0.5,
                       "trend_persistent_structure": 1.5},
    "divergence": {"any_indicator_divergence": 0.5},
}

# Mean reversion is capped when structure and regime both favour persistence.
MEAN_REVERSION_PERSISTENT_CAP = 0.4

# Higher timeframe used as one piece of structure evidence; others are descriptive.
HIGHER_TIMEFRAME = {"1Hour": "4Hour", "4Hour": "1Day", "1Day": "1Week", "1Week": None}
TIMEFRAME_PREFIX = {"1Week": "w1", "1Day": "d1", "4Hour": "h4", "1Hour": "h1"}


def weight(pattern: str, name: str) -> float:
    return EVIDENCE_WEIGHTS.get(pattern, {}).get(name, 1.0)
