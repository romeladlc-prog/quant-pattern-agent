"""Price/indicator divergences at confirmed swing pivots.

``bearish_divergence``: a new confirmed swing high is a higher high (HH) but
an indicator measured at the new pivot bar is weaker than at the previous
swing high. ``bullish_divergence``: lower low with a stronger indicator.
Evaluated on the bar where the new pivot is confirmed (``confirmed_at``),
comparing values at the two pivot bars, both already in the past.

A divergence alone never confirms: confirmation needs a close through the
neckline (the opposite extreme between the two pivots), which is structural,
non-divergence evidence.
"""
from __future__ import annotations

from .base import History, Observation, PatternDetector, num
from .common import TREND, any_true, higher_tf_is, near_opposing_level, volatility_expanding
from .config import EVENT_EVIDENCE_BARS

INDICATORS = {"momentum_divergence": ("momentum_20", "momentum"),
              "slope_divergence": ("slope_20_pct", "momentum"),
              "rs_divergence": ("price_benchmark_ratio", "momentum"),
              "volume_divergence": ("relative_volume", "momentum"),
              "breadth_divergence": ("breadth_sma50", "context")}


class DivergenceDetector(PatternDetector):
    family = "divergence"

    @property
    def move(self) -> int:
        return -self.sign  # bearish divergence forms on an up move (HH)

    def _pivots(self, r: dict) -> tuple[int | None, int | None]:
        key = "high" if self.move > 0 else "low"
        new, old = num(r, f"last_{key}_index"), num(r, f"prev_{key}_index")
        return (None if new is None else int(new)), (None if old is None else int(old))

    def observe(self, h: History) -> Observation:
        r, m = h.now, self.move
        fresh = bool(r.get("new_swing_high" if m > 0 else "new_swing_low"))
        label = r.get("last_high_label" if m > 0 else "last_low_label")
        new_i, old_i = self._pivots(r)
        # The latest confirmed pivot pair stays evaluable for EVENT_EVIDENCE_BARS
        # bars after the pivot's confirmation (pivot bar + right_bars).
        recent = new_i is not None and h.i - new_i <= EVENT_EVIDENCE_BARS + 2
        extreme = recent and label == ("HH" if m > 0 else "LL")
        new_row = h.at(new_i) if new_i is not None else None
        old_row = h.at(old_i) if old_i is not None else None
        divergences = {}
        for name, (column, _) in INDICATORS.items():
            a = num(new_row, column) if new_row and extreme else None
            b = num(old_row, column) if old_row and extreme else None
            divergences[name] = None if a is None or b is None else (a - b)*m < 0
        any_div = any_true(*divergences.values()) if extreme else False
        slope_change = num(r, "slope_change")
        breakout_with_move = r.get("breakout_state") in ("breakout_candidate", "breakout_confirmed") \
            and r.get("breakout_direction") == ("up" if m > 0 else "down")
        rs20 = num(r, "relative_return_20")
        strong = r.get("structural_state") == TREND[m] and higher_tf_is(r, TREND[m]) is True
        required = {"price_new_extreme": extreme, "any_indicator_divergence": any_div}
        support = [self.ev("price_new_extreme", "structure", extreme),
                   self.ev("any_indicator_divergence", "momentum", any_div)]
        support += [self.ev(name, INDICATORS[name][1], value) for name, value in divergences.items()]
        support += [
            self.ev("near_opposing_level", "structure", near_opposing_level(r, m)),
            self.ev("deceleration", "momentum", None if slope_change is None else slope_change*m < 0),
            self.ev("volatility_expanding", "volatility", volatility_expanding(r)),
        ]
        against = [
            self.ev("structure_trend_strong", "structure", strong),
            self.ev("active_breakout_with_move", "structure", breakout_with_move),
            self.ev("rs_strong", "momentum", None if rs20 is None else rs20*m > 0),
        ]
        details = {}
        if extreme and new_i is not None and old_i is not None:
            between = [h.at(k) for k in range(old_i, new_i+1)]
            neck_key = "low" if m > 0 else "high"
            values = [row[neck_key] for row in between]
            details = {"pivot_price": new_row["high" if m > 0 else "low"],
                       "neckline": min(values) if m > 0 else max(values),
                       "pivot_bar_timestamps": [old_row["timestamp"], new_row["timestamp"]],
                       "diverging": [k for k, v in divergences.items() if v]}
        details["_fresh_pivot"] = fresh
        return Observation(required, support, against, details)

    def entry(self, h, obs, score):
        return obs.details.get("_fresh_pivot") is True and super().entry(h, obs, score)

    def maintained(self, h, ep, obs, score):
        return True

    def on_start(self, h, ep, obs):
        ep.memory.update({k: obs.details[k] for k in ("pivot_price", "neckline",
                                                      "pivot_bar_timestamps", "diverging")})

    def confirmed(self, h, ep, obs):
        # Non-divergence evidence: a close through the neckline.
        return (h.now["close"] - ep.memory["neckline"])*self.move < 0

    def invalidated(self, h, ep, obs):
        atr = num(h.now, "atr")
        return atr is not None and (h.now["close"] - ep.memory["pivot_price"])*self.move > 0.25*atr

