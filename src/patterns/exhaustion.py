"""Exhaustion candidates: convergent evidence that an extended move is tiring.

``bearish_exhaustion`` describes an advance losing force; ``bullish_exhaustion``
a decline losing force. No RSI: the required core is an extended close
(z-score), recent acceleration in the move, and deceleration now.
"""
from __future__ import annotations

from .base import Episode, History, Observation, PatternDetector, clean, num
from .common import TREND, any_true, change_point_recent, ge, gt, high_vol_rising, \
    higher_tf_is, le, lt, near_opposing_level, regime_stable_low, volatility_expanding
from .config import THRESHOLDS


class ExhaustionDetector(PatternDetector):
    family = "exhaustion"

    @property
    def move(self) -> int:
        """Direction of the move being exhausted: up for bearish exhaustion."""
        return -self.sign

    def observe(self, h: History) -> Observation:
        r, m = h.now, self.move
        z = num(r, "zscore_20")
        slope_change = num(r, "slope_change")
        extended = None if z is None else z*m >= THRESHOLDS["zscore_extended"]
        prior = clean(r.get("accel_up_recent" if m > 0 else "accel_down_recent"))
        decel = None if slope_change is None else slope_change*m < 0
        then = h.ago(3)
        slope_now, slope_then = num(r, "rolling_slope_20"), num(then, "rolling_slope_20") if then else None
        deteriorating = None if slope_now is None or slope_then is None else (slope_now-slope_then)*m < 0
        rs5, rs_slope = num(r, "relative_return_5"), num(r, "ratio_slope")
        rs_weak = any_true(lt(None if rs5 is None else rs5*m, 0), lt(None if rs_slope is None else rs_slope*m, 0))
        rs20 = num(r, "relative_return_20")
        rs_strong = None if rs20 is None or rs_slope is None else rs20*m > 0 and rs_slope*m > 0
        vix_z = num(r, "vix_zscore")
        vix_extreme = None if vix_z is None else (vix_z <= THRESHOLDS["vix_zscore_low"] if m > 0
                                                   else vix_z >= THRESHOLDS["vix_zscore_high"])
        breadth_change = num(r, "breadth_sma50_change5")
        breadth_div = None if breadth_change is None else breadth_change*m < 0
        gap = clean(r.get("gap_up_recent" if m > 0 else "gap_down_recent"))
        breakout_with_move = r.get("breakout_state") in ("breakout_candidate", "breakout_confirmed") \
            and r.get("breakout_direction") == ("up" if m > 0 else "down")
        required = {"extended_vs_mean": extended, "prior_acceleration": prior, "deceleration": decel}
        support = [
            self.ev("extended_vs_mean", "momentum", extended),
            self.ev("prior_acceleration", "momentum", prior),
            self.ev("deceleration", "momentum", decel),
            self.ev("slope_deteriorating", "momentum", deteriorating),
            self.ev("rs_weakening", "momentum", rs_weak),
            self.ev("structure_exhaustion_flag", "structure", clean(r.get("exhaustion_candidate"))),
            self.ev("near_opposing_level", "structure", near_opposing_level(r, m)),
            self.ev("volatility_expanding", "volatility", volatility_expanding(r)),
            self.ev("relative_volume_high", "volatility",
                    ge(num(r, "relative_volume"), THRESHOLDS["relative_volume_high"])),
            self.ev("garch_vol_rising", "volatility",
                    ge(num(r, "garch_vol_ratio"), THRESHOLDS["garch_ratio_high"])),
            self.ev("change_point_recent", "regime", change_point_recent(r)),
            self.ev("high_vol_regime_rising", "regime", high_vol_rising(r)),
            self.ev("vix_extreme", "context", vix_extreme),
            self.ev("breadth_diverging", "context", breadth_div),
            self.ev("exhaustion_gap", "structure", None if gap is None else bool(gap)),
        ]
        against = [
            self.ev("higher_tf_trend_intact", "structure", higher_tf_is(r, TREND[m])),
            self.ev("rs_still_strong", "momentum", rs_strong),
            self.ev("active_breakout_with_move", "structure", breakout_with_move),
            self.ev("low_vol_regime_stable", "regime", regime_stable_low(r)),
            self.ev("hurst_persistent", "regime", gt(num(r, "hurst"), THRESHOLDS["hurst_persistent"])),
        ]
        return Observation(required, support, against)

    def maintained(self, h, ep, obs, score):
        z = num(h.now, "zscore_20")
        return z is not None and z*self.move > 0 and score >= self.machine.maintain_score

    def on_start(self, h: History, ep: Episode, obs: Observation) -> None:
        window = h.window(5)
        key = "high" if self.move > 0 else "low"
        values = [w[key] for w in window]
        ep.memory["move_extreme"] = max(values) if self.move > 0 else min(values)

    def confirmed(self, h, ep, obs):
        # Reversal confirmed: the close is back at or through its 20-bar mean.
        return le(None if num(h.now, "zscore_20") is None else num(h.now, "zscore_20")*self.move, 0) is True

    def invalidated(self, h, ep, obs):
        r, atr = h.now, num(h.now, "atr")
        if atr is None:
            return False
        extreme = ep.memory["move_extreme"]
        return (r["close"] - extreme)*self.move > 0.5*atr
