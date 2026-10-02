"""Pattern results, evidence scoring and the shared deterministic state machine.

Causality is structural: a detector sees bar i through ``History``, which
exposes rows 0..i only, and the state machine walks bars in order keeping
only past state. Adding later bars therefore cannot change a past result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import pandas as pd

from .config import COMPONENT_WEIGHTS, COMPONENTS, PATTERN_CATALOG, SCORE_NAME, \
    STATE_MACHINES, StateMachineConfig, weight


def clean(value):
    """NaN/NaT -> None and numpy scalars -> Python, for stable comparisons."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if hasattr(value, "item") and not isinstance(value, pd.Timestamp):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return value
        if isinstance(value, float) and math.isnan(value):
            return None
    return value


def deep_clean(value):
    if isinstance(value, dict):
        return {k: deep_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [deep_clean(v) for v in value]
    return clean(value)


def num(row: dict, key: str) -> float | None:
    value = clean(row.get(key))
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def both(*values) -> bool:
    return all(v is not None for v in values)


class History:
    """Read-only view of evidence rows 0..i. Never exposes row i+1 or later."""

    def __init__(self, rows: list[dict], i: int):
        self._rows, self.i = rows, i

    @property
    def now(self) -> dict:
        return self._rows[self.i]

    def ago(self, k: int) -> dict | None:
        return self._rows[self.i-k] if 0 <= k <= self.i else None

    def at(self, position: int) -> dict | None:
        return self._rows[position] if 0 <= position <= self.i else None

    def window(self, k: int) -> list[dict]:
        return self._rows[max(0, self.i-k+1):self.i+1]


@dataclass(frozen=True)
class Evidence:
    name: str
    component: str
    weight: float
    value: bool | None  # None: not evaluable (missing input), never counted as zero
    detail: object = None


@dataclass
class Observation:
    required: dict[str, bool | None]
    support: list[Evidence]
    against: list[Evidence]
    details: dict = field(default_factory=dict)

    @property
    def required_ok(self) -> bool:
        return bool(self.required) and all(v is True for v in self.required.values())


@dataclass
class PatternResult:
    ticker: str
    timestamp: pd.Timestamp
    timeframe: str
    pattern_name: str
    pattern_family: str
    direction: str
    state: str
    score: float
    score_structure: float | None
    score_volatility: float | None
    score_regime: float | None
    score_context: float | None
    score_momentum: float | None
    evidence_for: list[str]
    evidence_against: list[str]
    required_conditions_met: list[str]
    required_conditions_missing: list[str]
    optional_conditions_met: list[str]
    first_detected_at: pd.Timestamp | None
    confirmed_at: pd.Timestamp | None
    invalidated_at: pd.Timestamp | None
    expired_at: pd.Timestamp | None
    bars_active: int
    regime_context: dict
    volatility_context: dict
    structure_context: dict
    external_context: dict
    notes: list[str]
    subtype: str | None = None
    episode_id: int | None = None
    bar_index: int = -1
    details: dict = field(default_factory=dict)
    mtf_context: dict = field(default_factory=dict)
    score_kind: str = SCORE_NAME

    def as_dict(self) -> dict:
        return asdict(self)


def component_scores(support: list[Evidence], against: list[Evidence]) -> dict[str, float | None]:
    """score_c = S_c / (E_c + A_c).

    S_c: weight of supporting evidence present; E_c: weight of supporting
    evidence that could be evaluated; A_c: weight of contradicting evidence
    present. No evaluable evidence in a component -> None (excluded, not 0).
    """
    scores = {}
    for component in COMPONENTS:
        s = sum(e.weight for e in support if e.component == component and e.value is True)
        e_ = sum(e.weight for e in support if e.component == component and e.value is not None)
        a = sum(e.weight for e in against if e.component == component and e.value is True)
        scores[component] = None if e_ + a == 0 else s / (e_ + a)
    return scores


def pattern_score(scores: dict[str, float | None]) -> float:
    """Weighted mean of available component scores, in [0, 1]."""
    available = {c: s for c, s in scores.items() if s is not None}
    total = sum(COMPONENT_WEIGHTS[c] for c in available)
    if not total:
        return 0.0
    return round(min(1.0, max(0.0, sum(COMPONENT_WEIGHTS[c]*s for c, s in available.items())/total)), 6)


@dataclass
class Episode:
    episode_id: int
    start: int
    first_detected_at: pd.Timestamp
    state: str = "candidate"
    confirmed_index: int | None = None
    confirmed_at: pd.Timestamp | None = None
    invalidated_at: pd.Timestamp | None = None
    expired_at: pd.Timestamp | None = None
    lapses: int = 0
    closed: bool = False
    memory: dict = field(default_factory=dict)
    subtype: str | None = None


class PatternDetector:
    """Base detector. Subclasses implement ``observe`` and the episode hooks.

    Transitions per bar (in this order, once an episode is open):
      1. invalidated: ``invalidated`` is true (any open state, incl. confirmed);
      2. confirmed: pending and ``confirmed`` is true and score >= confirm_score;
      3. pending checks: a lapse (``maintained`` false) increments a counter;
         more than ``grace_bars`` lapses or more than ``max_pending_bars``
         bars without confirmation -> expired;
         otherwise candidate -> developing after ``developing_after_bars``;
      4. confirmed episodes close after ``confirmed_ttl_bars`` (state stays
         confirmed; ``expired_at`` records when it aged out).
    A new episode opens only when no episode is open, the cooldown has passed,
    every required condition is true and score >= enter_score.
    """

    family: str = ""

    def __init__(self, direction: str, ticker: str, timeframe: str,
                 machine: StateMachineConfig | None = None):
        self.direction = direction
        self.ticker, self.timeframe = ticker, timeframe
        self.machine = machine or STATE_MACHINES[self.family]

    @property
    def name(self) -> str:
        return f"{self.direction}_{self.family}"

    @property
    def sign(self) -> int:
        return 1 if self.direction in ("bullish", "to_high_vol") else -1

    # -- evidence helpers -------------------------------------------------
    def ev(self, name: str, component: str, value, detail=None) -> Evidence:
        catalog = PATTERN_CATALOG[self.family]
        if name not in catalog["optional"] + catalog["contradictory"] + catalog["required"]:
            raise KeyError(f"{self.family}: evidence {name!r} not in catalog")
        value = None if value is None else bool(value)
        return Evidence(name, component, weight(self.family, name), value, detail)

    # -- hooks --------------------------------------------------------------
    def observe(self, h: History) -> Observation:
        raise NotImplementedError

    def entry(self, h: History, obs: Observation, score: float) -> bool:
        return obs.required_ok and score >= self.machine.enter_score

    def maintained(self, h: History, ep: Episode, obs: Observation, score: float) -> bool:
        return obs.required_ok and score >= self.machine.maintain_score

    def on_start(self, h: History, ep: Episode, obs: Observation) -> None:
        pass

    def confirmed(self, h: History, ep: Episode, obs: Observation) -> bool:
        return False

    def invalidated(self, h: History, ep: Episode, obs: Observation) -> bool:
        return False

    def adjust_score(self, h: History, obs: Observation, score: float) -> tuple[float, list[str]]:
        return score, []

    def subtype(self, h: History, ep: Episode | None, obs: Observation) -> str | None:
        return None

    # -- driver -------------------------------------------------------------
    def run(self, rows: list[dict], contexts: list[dict]) -> list[PatternResult]:
        m = self.machine
        results: list[PatternResult] = []
        episode: Episode | None = None
        cooldown_until = -1
        next_id = 0
        for i in range(len(rows)):
            h = History(rows, i)
            stamp = h.now["timestamp"]
            obs = self.observe(h)
            scores = component_scores(obs.support, obs.against)
            score, notes = self.adjust_score(h, obs, pattern_score(scores))
            closed_now = False
            if episode is None:
                if i > cooldown_until and self.entry(h, obs, score):
                    episode = Episode(next_id, i, stamp)
                    next_id += 1
                    self.on_start(h, episode, obs)
            else:
                active = i - episode.start + 1
                pending = episode.state in ("candidate", "developing")
                if self.invalidated(h, episode, obs):
                    episode.state, episode.invalidated_at = "invalidated", stamp
                    closed_now = True
                elif pending and self.confirmed(h, episode, obs) and score >= m.confirm_score:
                    episode.state, episode.confirmed_at, episode.confirmed_index = "confirmed", stamp, i
                elif pending:
                    episode.lapses = 0 if self.maintained(h, episode, obs, score) else episode.lapses+1
                    if episode.lapses > m.grace_bars or active > m.max_pending_bars:
                        episode.state, episode.expired_at = "expired", stamp
                        closed_now = True
                    elif episode.state == "candidate" and active > m.developing_after_bars \
                            and episode.lapses == 0:
                        episode.state = "developing"
                elif episode.state == "confirmed" and i - episode.confirmed_index >= m.confirmed_ttl_bars:
                    episode.expired_at = stamp
                    closed_now = True
            if episode is not None:
                episode.subtype = self.subtype(h, episode, obs) or episode.subtype
            results.append(self._result(h, obs, scores, score, notes, episode, contexts[i]))
            if closed_now:
                episode.closed = True
                episode = None
                cooldown_until = i + m.cooldown_bars
        return results

    def _result(self, h, obs, scores, score, notes, ep, context) -> PatternResult:
        row = h.now
        support_true = [e.name for e in obs.support if e.value is True]
        against_true = [e.name for e in obs.against if e.value is True]
        required_met = [k for k, v in obs.required.items() if v is True]
        required_missing = [k for k, v in obs.required.items() if v is not True]
        optional = [e.name for e in obs.support if e.value is True and e.name not in obs.required]
        missing_inputs = sorted({e.name for e in obs.support + obs.against if e.value is None})
        notes = list(notes)
        if missing_inputs:
            notes.append("not_evaluable:" + ",".join(missing_inputs))
        unconverged = [model for model, flag in (
            ("hmm2", context["regime"].get("hmm_fit_converged")),
            ("markov2", context["regime"].get("markov_fit_converged")),
            ("kalman_local_level", context["regime"].get("kalman_fit_converged")),
            ("garch", context["volatility"].get("garch_fit_converged"))) if flag is False]
        if unconverged:
            notes.append("model_not_converged:" + ",".join(unconverged))
        details = {k: v for k, v in obs.details.items() if not k.startswith("_")}
        if ep is not None:
            details.update({k: v for k, v in ep.memory.items() if not k.startswith("_")})
        return PatternResult(
            ticker=self.ticker, timestamp=row["timestamp"], timeframe=self.timeframe,
            pattern_name=self.name, pattern_family=self.family, direction=self.direction,
            state=ep.state if ep is not None else "inactive", score=score,
            score_structure=_round(scores["structure"]), score_volatility=_round(scores["volatility"]),
            score_regime=_round(scores["regime"]), score_context=_round(scores["context"]),
            score_momentum=_round(scores["momentum"]),
            evidence_for=support_true, evidence_against=against_true,
            required_conditions_met=required_met, required_conditions_missing=required_missing,
            optional_conditions_met=optional,
            first_detected_at=ep.first_detected_at if ep else None,
            confirmed_at=ep.confirmed_at if ep else None,
            invalidated_at=ep.invalidated_at if ep else None,
            expired_at=ep.expired_at if ep else None,
            bars_active=(h.i - ep.start + 1) if ep else 0,
            regime_context=context["regime"], volatility_context=context["volatility"],
            structure_context=context["structure"], external_context=context["external"],
            notes=notes, subtype=ep.subtype if ep else None,
            episode_id=ep.episode_id if ep else None, bar_index=h.i,
            details=deep_clean(details), mtf_context=context["mtf"])


def _round(value):
    return None if value is None else round(value, 6)
