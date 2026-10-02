"""Trend continuation candidates: a pullback inside an intact swing structure.

``bullish_continuation``: HH/HL structure, positive slope, price pulled back
below the last confirmed swing high but above the last swing low.
Confirmation is resumption beyond the swing extreme known at detection;
invalidation is a close beyond the opposite swing or a structure break.
Descriptive only: no entry is suggested.
"""
from __future__ import annotations

from .base import History, Observation, PatternDetector, clean, num
from .common import TREND, any_true, breadth_supportive, ge, higher_tf_is, regime_shift, \
    regime_stable_low, vix_supportive, volatility_expanding
from .config import THRESHOLDS


class TrendContinuationDetector(PatternDetector):
    family = "trend_continuation"

    def observe(self, h: History) -> Observation:
        r, d = h.now, self.sign
        last_high, last_low = num(r, "last_high"), num(r, "last_low")
        trigger, stop = (last_high, last_low) if d > 0 else (last_low, last_high)
        z, slope = num(r, "zscore_20"), num(r, "rolling_slope_20")
        trend = r.get("structural_state") == TREND[d]
        slope_aligned = None if slope is None else slope*d > 0
        pullback = None if trigger is None or stop is None or z is None else \
            (r["close"]-trigger)*d < 0 and (r["close"]-stop)*d > 0 and z*d <= 0.5
        kalman = num(r, "kalman_level_change")
        rs20 = num(r, "relative_return_20")
        expanding = volatility_expanding(r)
        garch = num(r, "garch_vol_ratio")
        compatible = None if expanding is None and garch is None else \
            not (expanding is True or (garch is not None and garch > THRESHOLDS["garch_ratio_high"]))
        shift = regime_shift(r)
        last_label = r.get("last_high_label") if d > 0 else r.get("last_low_label")
        warning = None if last_label is None else last_label == ("LH" if d > 0 else "HL")
        required = {"structure_trend": trend, "slope_aligned": slope_aligned, "pullback": pullback}
        support = [
            self.ev("structure_trend", "structure", trend),
            self.ev("slope_aligned", "momentum", slope_aligned),
            self.ev("pullback", "structure", pullback),
            self.ev("compression_recent", "volatility",
                    any_true(clean(r.get("compression_recent")), clean(r.get("volatility_compression")))),
            self.ev("higher_tf_aligned", "structure", higher_tf_is(r, TREND[d])),
            self.ev("kalman_aligned", "momentum", None if kalman is None else kalman*d > 0),
            self.ev("rs_aligned", "momentum", None if rs20 is None else rs20*d > 0),
            self.ev("volatility_compatible", "volatility", compatible),
            self.ev("regime_stable", "regime", regime_stable_low(r)),
            self.ev("breadth_supportive", "context", breadth_supportive(r, d)),
            self.ev("vix_supportive", "context", vix_supportive(r, d)),
        ]
        against = [
            self.ev("rs_against", "momentum", None if rs20 is None else rs20*d < 0),
            self.ev("volatility_expanding", "volatility", expanding),
            self.ev("regime_shifting", "regime", ge(shift, 1.5*THRESHOLDS["regime_shift"])),
            self.ev("higher_tf_against", "structure", higher_tf_is(r, TREND[-d])),
            self.ev("structure_warning", "structure", warning),
        ]
        return Observation(required, support, against, {"trigger": trigger, "stop": stop})

    def maintained(self, h, ep, obs, score):
        return h.now.get("structural_state") == TREND[self.sign] and score >= self.machine.maintain_score

    def on_start(self, h, ep, obs):
        ep.memory["trigger"], ep.memory["stop"] = obs.details["trigger"], obs.details["stop"]

    def confirmed(self, h, ep, obs):
        return (h.now["close"] - ep.memory["trigger"])*self.sign > 0

    def invalidated(self, h, ep, obs):
        broke = (h.now["close"] - ep.memory["stop"])*self.sign < 0
        return broke or h.now.get("structural_state") == TREND[-self.sign]
