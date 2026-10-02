"""Breakout and failed-breakout patterns built only on Phase 6.1 causal events.

Each episode starts from an event that ``detect_breakouts`` already dated at
the bar where it was observed, tested against a level known at the previous
bar (``level_asof_timestamp`` < event ``timestamp``). Full-sample levels are
never used. Descriptive subtypes, not signals:

* ``breakout_under_confirmation``: candidate outside the level, not yet confirmed;
* ``breakout_with_expansion``: confirmed with volatility expansion and volume;
* ``weak_breakout``: confirmed without expansion or volume;
* ``failed_breakout_risk``: price back near the level, or momentum and RS against.
"""
from __future__ import annotations

from .base import Episode, History, Observation, PatternDetector, clean, num
from .common import TREND, ge, high_vol_rising, higher_tf_is, lt, regime_stable_low, \
    volatility_expanding
from .config import EVENT_EVIDENCE_BARS, THRESHOLDS


def _events(row: dict, states: tuple[str, ...], direction: str) -> list[dict]:
    return [e for e in row.get("events_here", []) if e["state"] in states and e["direction"] == direction]


def _recent_event(h: History, states: tuple[str, ...], direction: str) -> tuple[dict | None, int | None]:
    """Latest matching event observed in the last EVENT_EVIDENCE_BARS bars (age 0 = this bar)."""
    for age in range(EVENT_EVIDENCE_BARS+1):
        row = h.ago(age)
        if row is None:
            break
        events = _events(row, states, direction)
        if events:
            return _pick(events), age
    return None, None


def _pick(events: list[dict]) -> dict:
    """Several levels crossed on one bar: the most-touched level, then the nearest."""
    return sorted(events, key=lambda e: (-(e.get("level_touch_count") or 0), e["distance_atr"]))[0]


class BreakoutDetector(PatternDetector):
    family = "breakout"

    @property
    def side(self) -> str:
        return "up" if self.sign > 0 else "down"

    def observe(self, h: History) -> Observation:
        r, d = h.now, self.sign
        event, age = _recent_event(h, ("breakout_candidate",), self.side)
        rv = num(r, "relative_volume")
        rs20, slope_change = num(r, "relative_return_20"), num(r, "slope_change")
        atr = num(r, "atr")
        level = event["level"] if event else None
        known_before = None if event is None else event["level_asof_timestamp"] < event["timestamp"]
        gap = clean(r.get("gap_up_recent" if d > 0 else "gap_down_recent"))
        vix_z = num(r, "vix_zscore")
        required = {"causal_breakout_event": event is not None, "level_known_before_event": known_before}
        support = [
            self.ev("causal_breakout_event", "structure", event is not None),
            self.ev("level_known_before_event", "structure", known_before),
            self.ev("strong_level", "structure",
                    None if event is None else (event.get("level_touch_count") or 0) >= 3),
            self.ev("prior_compression", "volatility", clean(r.get("compression_recent"))),
            self.ev("structure_aligned", "structure", r.get("structural_state") == TREND[d]),
            self.ev("higher_tf_aligned", "structure", higher_tf_is(r, TREND[d])),
            self.ev("volatility_expansion", "volatility", volatility_expanding(r)),
            self.ev("relative_volume_ok", "momentum", ge(rv, THRESHOLDS["relative_volume_ok"])),
            self.ev("rs_aligned", "momentum", None if rs20 is None else rs20*d > 0),
            self.ev("slope_aligned", "momentum",
                    None if num(r, "rolling_slope_20") is None else num(r, "rolling_slope_20")*d > 0),
            self.ev("regime_stable", "regime", regime_stable_low(r)),
            self.ev("vix_supportive", "context", None if vix_z is None else
                    (vix_z < THRESHOLDS["vix_zscore_high"] if d > 0 else vix_z > 0)),
            self.ev("breadth_supportive", "context", None if num(r, "breadth_sma50") is None else
                    (num(r, "breadth_sma50") >= 0.5 if d > 0 else num(r, "breadth_sma50") < 0.5)),
            self.ev("breakout_gap", "structure", None if gap is None else bool(gap)),
        ]
        near = None if level is None or atr is None else \
            abs(r["close"]-level) < THRESHOLDS["inside_level_atr"]*atr
        against = [
            self.ev("low_volume", "momentum", lt(rv, THRESHOLDS["relative_volume_ok"])),
            self.ev("rs_against", "momentum", None if rs20 is None else rs20*d < 0),
            self.ev("momentum_decelerating", "momentum", None if slope_change is None else slope_change*d < 0),
            self.ev("close_near_level", "structure", near),
            self.ev("higher_tf_against", "structure", higher_tf_is(r, TREND[-d])),
            self.ev("vix_spike", "context", None if vix_z is None else
                    (vix_z >= 2*THRESHOLDS["vix_zscore_high"] if d > 0 else False)),
        ]
        details = {"_event": event, "_event_age": age} if event else {}
        return Observation(required, support, against, details)

    def entry(self, h, obs, score):
        return obs.details.get("_event_age") == 0 and super().entry(h, obs, score)

    def maintained(self, h, ep, obs, score):
        return True  # event-driven: ends by confirmation, failure or the pending limit

    def on_start(self, h, ep, obs):
        e = obs.details["_event"]
        ep.memory.update({"level": e["level"], "level_asof_timestamp": e["level_asof_timestamp"],
                          "breakout_timestamp": e["timestamp"], "breakout_distance_atr": e["distance_atr"],
                          "relative_volume": e["relative_volume"], "level_type": e.get("level_type"),
                          "level_touch_count": e.get("level_touch_count")})

    def _same_level(self, event: dict, ep: Episode) -> bool:
        return event["level"] == ep.memory["level"] and \
            event["level_asof_timestamp"] == ep.memory["level_asof_timestamp"]

    def confirmed(self, h, ep, obs):
        return any(self._same_level(e, ep) for e in _events(h.now, ("breakout_confirmed",), self.side))

    def invalidated(self, h, ep, obs):
        fake = f"fake_breakout_{self.side}"
        if any(self._same_level(e, ep) for e in _events(h.now, (fake,), self.side)):
            return True
        return (h.now["close"] - ep.memory["level"])*self.sign <= 0

    def subtype(self, h, ep, obs):
        if ep is None:
            return None
        names = {e.name for e in obs.support + obs.against if e.value is True}
        r, atr = h.now, num(h.now, "atr")
        near = atr is not None and abs(r["close"]-ep.memory["level"]) < THRESHOLDS["inside_level_atr"]*atr
        risk = near or {"momentum_decelerating", "rs_against"} <= names
        if risk:
            return "failed_breakout_risk"
        if ep.state == "confirmed":
            expansion = "volatility_expansion" in names and "relative_volume_ok" in names
            return "breakout_with_expansion" if expansion else "weak_breakout"
        return "breakout_under_confirmation"


class FailedBreakoutDetector(PatternDetector):
    """``bullish_failed_breakout``: a failed *down* breakout (back above support);
    ``bearish_failed_breakout``: a failed *up* breakout (back below resistance)."""

    family = "failed_breakout"

    @property
    def broken_side(self) -> str:
        return "down" if self.sign > 0 else "up"

    @property
    def broke(self) -> int:
        return -self.sign  # direction of the original, failed breakout

    def observe(self, h: History) -> Observation:
        r, b = h.now, self.broke
        event, age = _recent_event(h, (f"fake_breakout_{self.broken_side}",), self.broken_side)
        atr = num(r, "atr")
        level = event["level"] if event else None
        back = None if level is None or atr is None else \
            (level - r["close"])*b >= THRESHOLDS["inside_level_atr"]*atr
        reclaim = None if level is None else (r["close"]-level)*b > 0
        rs5, rs20 = num(r, "relative_return_5"), num(r, "relative_return_20")
        slope_change = num(r, "slope_change")
        vix_change = num(r, "vix_change_5d")
        required = {"causal_fake_breakout_event": event is not None}
        support = [
            self.ev("causal_fake_breakout_event", "structure", event is not None),
            self.ev("back_in_range", "structure", back),
            self.ev("strong_level", "structure",
                    None if event is None else (event.get("level_touch_count") or 0) >= 3),
            self.ev("momentum_deteriorating", "momentum", None if slope_change is None else slope_change*b < 0),
            self.ev("rs_weak", "momentum", None if rs5 is None else rs5*b < 0),
            self.ev("volume_on_failure", "momentum", ge(num(r, "relative_volume"), THRESHOLDS["relative_volume_ok"])),
            self.ev("volatility_expansion", "volatility", volatility_expanding(r)),
            self.ev("high_vol_regime_rising", "regime", high_vol_rising(r)),
            self.ev("vix_rising", "context", None if vix_change is None else vix_change*b > 0),
        ]
        against = [
            self.ev("close_reclaims_level", "structure", reclaim),
            self.ev("rs_strong", "momentum", None if rs20 is None else rs20*b > 0),
            self.ev("higher_tf_aligned_with_breakout", "structure", higher_tf_is(r, TREND[b])),
        ]
        return Observation(required, support, against,
                           {"_event": event, "_event_age": age} if event else {})

    def entry(self, h, obs, score):
        return obs.details.get("_event_age") == 0 and super().entry(h, obs, score)

    def maintained(self, h, ep, obs, score):
        return True

    def on_start(self, h, ep, obs):
        e = obs.details["_event"]
        failure_bars = int(e["failure_bars"]) if e.get("failure_bars") is not None else None
        breakout_at = h.at(h.i - failure_bars) if failure_bars is not None else None
        ep.memory.update({
            "level": e["level"], "level_asof_timestamp": e["level_asof_timestamp"],
            "breakout_timestamp": breakout_at["timestamp"] if breakout_at else None,
            "failure_timestamp": e["timestamp"], "bars_to_failure": failure_bars,
            "breakout_distance_atr": e["distance_atr"], "relative_volume": e["relative_volume"],
            "volatility_context": "expanding" if volatility_expanding(h.now) else "not_expanding",
            "_inside_closes": 0})

    def confirmed(self, h, ep, obs):
        inside = (h.now["close"] - ep.memory["level"])*self.broke < 0
        if h.i == ep.start:
            return False
        ep.memory["_inside_closes"] = ep.memory["_inside_closes"]+1 if inside else 0
        return ep.memory["_inside_closes"] >= 2

    def invalidated(self, h, ep, obs):
        atr = num(h.now, "atr")
        return atr is not None and \
            (h.now["close"] - ep.memory["level"])*self.broke > THRESHOLDS["inside_level_atr"]*atr
