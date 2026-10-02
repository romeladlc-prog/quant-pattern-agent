"""Regime transitions from validated two-state models only (HMM2, Markov2).

``to_high_vol_regime_transition`` / ``to_low_vol_regime_transition``. A
candidate opens when either model's filtered high-volatility probability
crosses 0.5 versus ``lag_bars`` earlier (same parameter set). It is
confirmed when both models give the new regime >= 0.6 for
``regime_confirm_bars`` consecutive bars, and invalidated when both fall
back below 0.4. Three-state models are never consulted.
"""
from __future__ import annotations

from src.models.model_status import validated_models

from .base import History, Observation, PatternDetector, clean, num
from .common import regime_values
from .config import THRESHOLDS

REQUIRED_MODELS = ("hmm2", "markov2")


class RegimeTransitionDetector(PatternDetector):
    family = "regime_transition"

    @property
    def name(self) -> str:
        return f"{self.direction}_regime_transition"

    def _new(self, p: float | None) -> float | None:
        """Probability of the destination regime."""
        return None if p is None else (p if self.sign > 0 else 1-p)

    def observe(self, h: History) -> Observation:
        r = h.now
        hmm, hmm_lag, mk, mk_lag = regime_values(r)
        validated = set(validated_models("regime_description"))
        models_ok = set(REQUIRED_MODELS) <= validated and (hmm is not None or mk is not None)
        threshold = THRESHOLDS["regime_probability"]
        crossings = [self._new(a) >= threshold > self._new(b)
                     for a, b in ((hmm, hmm_lag), (mk, mk_lag)) if a is not None and b is not None]
        cross = None if not crossings else any(crossings)
        agree = None if hmm is None or mk is None else \
            self._new(hmm) >= threshold and self._new(mk) >= threshold
        disagree = None if hmm is None or mk is None else \
            min(self._new(hmm), self._new(mk)) < THRESHOLDS["regime_low_probability"] and \
            max(self._new(hmm), self._new(mk)) >= threshold
        rv_now, before = num(r, "realized_volatility_20"), h.ago(5)
        rv_then = num(before, "realized_volatility_20") if before else None
        rv_move = None if rv_now is None or rv_then is None else (rv_now-rv_then)*self.sign > 0
        garch = num(r, "garch_vol_ratio")
        garch_move = None if garch is None else (garch >= THRESHOLDS["garch_ratio_high"] if self.sign > 0
                                                 else garch <= THRESHOLDS["garch_ratio_low"])
        k_now, k_lag = num(r, "kalman_level_change"), num(r, "kalman_level_change_lag")
        kalman_turn = None if k_now is None or k_lag is None else (k_now > 0) != (k_lag > 0)
        vix_change = num(r, "vix_change_5d")
        required = {"validated_regime_models": models_ok, "regime_probability_cross": cross}
        support = [
            self.ev("validated_regime_models", "regime", models_ok),
            self.ev("regime_probability_cross", "regime", cross),
            self.ev("other_model_agrees", "regime", agree),
            self.ev("change_point_rv20", "regime", clean(r.get("cp_rv20_recent"))),
            self.ev("change_point_slope", "regime", clean(r.get("cp_slope_recent"))),
            self.ev("rv20_moving", "volatility", rv_move),
            self.ev("garch_vol_moving", "volatility", garch_move),
            self.ev("structure_changed", "structure", clean(r.get("structure_changed_recent"))),
            self.ev("kalman_turn", "momentum", kalman_turn),
            self.ev("vix_moving", "context", None if vix_change is None else vix_change*self.sign > 0),
        ]
        against = [
            self.ev("models_disagree", "regime", disagree),
            self.ev("rv20_against", "volatility", None if rv_move is None else not rv_move),
        ]
        probabilities = {"hmm2_p_high": hmm, "hmm2_p_high_lag": hmm_lag,
                         "markov2_p_high": mk, "markov2_p_high_lag": mk_lag}
        return Observation(required, support, against, {"regime_probabilities": probabilities})

    def maintained(self, h, ep, obs, score):
        return True

    def on_start(self, h, ep, obs):
        ep.memory["_streak"] = 0
        ep.memory["probabilities_at_detection"] = obs.details["regime_probabilities"]

    def confirmed(self, h, ep, obs):
        hmm, _, mk, _ = regime_values(h.now)
        both = hmm is not None and mk is not None and \
            min(self._new(hmm), self._new(mk)) >= THRESHOLDS["regime_confirm_probability"]
        ep.memory["_streak"] = ep.memory["_streak"]+1 if both else 0
        return ep.memory["_streak"] >= THRESHOLDS["regime_confirm_bars"]

    def invalidated(self, h, ep, obs):
        hmm, _, mk, _ = regime_values(h.now)
        values = [self._new(v) for v in (hmm, mk) if v is not None]
        return bool(values) and all(v < THRESHOLDS["regime_revert_probability"] for v in values)

    def subtype(self, h, ep, obs):
        if ep is None:
            return None
        return "regime_transition_confirmed" if ep.state == "confirmed" else "regime_transition_candidate"
