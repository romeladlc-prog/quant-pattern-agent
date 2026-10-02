"""Mean-reversion candidates: an extreme stretch plus evidence of exhaustion.

``bullish_mean_reversion``: close stretched far below its mean;
``bearish_mean_reversion``: stretched far above. Hurst and entropy are
auxiliary optional evidence only and can never open the pattern alone.
When structure *and* regime both favour persistence in the stretched
direction, the score is capped and confirmation is blocked.
"""
from __future__ import annotations

from .base import History, Observation, PatternDetector, clean, num
from .common import TREND, any_true, ge, gt, higher_tf_is, lt, near_opposing_level
from .config import MEAN_REVERSION_PERSISTENT_CAP, THRESHOLDS


class MeanReversionDetector(PatternDetector):
    family = "mean_reversion"

    @property
    def move(self) -> int:
        return -self.sign  # stretched direction

    def _persistent(self, r: dict) -> tuple[bool | None, bool | None]:
        m = self.move
        structure = r.get("structural_state") == TREND[m]
        hurst = gt(num(r, "hurst"), THRESHOLDS["hurst_persistent"])
        kalman = num(r, "kalman_level_change")
        kalman_aligned = None if kalman is None else kalman*m > 0
        regime = None if hurst is None and kalman_aligned is None else \
            bool(hurst is True or (kalman_aligned is True and clean(r.get("accel_up_recent" if m > 0
                                                                          else "accel_down_recent"))))
        return structure, regime

    def observe(self, h: History) -> Observation:
        r, m = h.now, self.move
        z = num(r, "zscore_20")
        slope_change = num(r, "slope_change")
        decel = None if slope_change is None else slope_change*m < 0
        near = near_opposing_level(r, m)
        exhaustion = any_true(decel, clean(r.get("exhaustion_candidate")), near)
        extreme = None if z is None else z*m >= THRESHOLDS["zscore_extreme"]
        structure_persistent, regime_persistent = self._persistent(r)
        breadth = num(r, "breadth_sma50")
        breadth_extreme = None if breadth is None else (breadth <= 0.2 if m < 0 else breadth >= 0.8)
        vix_z = num(r, "vix_zscore")
        vix_extreme = None if vix_z is None else (vix_z >= 1.5 if m < 0 else vix_z <= THRESHOLDS["vix_zscore_low"])
        accelerating = r.get("acceleration_state") == ("accelerating_up" if m > 0 else "accelerating_down")
        required = {"extreme_zscore": extreme, "exhaustion_evidence": exhaustion}
        support = [
            self.ev("extreme_zscore", "momentum", extreme),
            self.ev("exhaustion_evidence", "momentum", exhaustion),
            self.ev("near_opposing_level", "structure", near),
            self.ev("deceleration", "momentum", decel),
            self.ev("volatility_elevated", "volatility",
                    ge(num(r, "rv20_percentile"), THRESHOLDS["rv_percentile_high"])),
            self.ev("hurst_antipersistent", "regime", lt(num(r, "hurst"), THRESHOLDS["hurst_antipersistent"])),
            self.ev("entropy_high", "regime",
                    ge(num(r, "permutation_entropy"), THRESHOLDS["permutation_entropy_high"])),
            self.ev("breadth_extreme", "context", breadth_extreme),
            self.ev("vix_extreme", "context", vix_extreme),
        ]
        against = [
            self.ev("trend_persistent_structure", "structure", structure_persistent),
            self.ev("regime_persistent", "regime", regime_persistent),
            self.ev("higher_tf_trend_aligned", "structure", higher_tf_is(r, TREND[m])),
            self.ev("still_accelerating", "momentum", accelerating),
        ]
        blocked = structure_persistent is True and regime_persistent is True
        return Observation(required, support, against, {"persistence_block": blocked})

    def adjust_score(self, h, obs, score):
        if obs.details.get("persistence_block"):
            return min(score, MEAN_REVERSION_PERSISTENT_CAP), ["capped_trend_and_regime_persistent"]
        return score, []

    def on_start(self, h, ep, obs):
        ep.memory["z_start"] = num(h.now, "zscore_20")
        ep.memory["move_extreme"] = h.now["high"] if self.move > 0 else h.now["low"]

    def confirmed(self, h, ep, obs):
        if obs.details.get("persistence_block"):
            return False
        z = num(h.now, "zscore_20")
        return z is not None and z*self.move <= 1.0

    def invalidated(self, h, ep, obs):
        r, m = h.now, self.move
        z, atr = num(r, "zscore_20"), num(r, "atr")
        stretched = z is not None and z*m >= ep.memory["z_start"]*m + 1.0
        beyond = atr is not None and (r["close"] - ep.memory["move_extreme"])*m > atr
        return stretched or beyond
