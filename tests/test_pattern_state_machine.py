"""Generic state machine: transitions, grace, expiry, TTL, invalidation, cooldown."""
import pytest

from pattern_rows import make_rows, span, states
from src.patterns.base import History, Observation, PatternDetector
from src.patterns.checks import ALLOWED
from src.patterns.config import STATES, StateMachineConfig
from src.patterns.engine import bar_contexts

MACHINE = StateMachineConfig(enter_score=0.4, maintain_score=0.25, confirm_score=0.35,
                             developing_after_bars=2, grace_bars=2, max_pending_bars=6,
                             confirmed_ttl_bars=3, cooldown_bars=2)


class Scripted(PatternDetector):
    """Evidence comes from row flags: req (required), conf, inv."""
    family = "exhaustion"

    def observe(self, h: History) -> Observation:
        r = h.now
        req = r.get("req", False)
        support = [self.ev("extended_vs_mean", "momentum", req),
                   self.ev("deceleration", "momentum", r.get("strong", req))]
        return Observation({"extended_vs_mean": req}, support, [])

    def confirmed(self, h, ep, obs):
        return bool(h.now.get("conf"))

    def invalidated(self, h, ep, obs):
        return bool(h.now.get("inv"))


def run_script(overrides, n=20):
    rows = make_rows(n, overrides)
    return Scripted("bearish", "T", "1Day", MACHINE).run(rows, [bar_contexts(r, []) for r in rows])


def test_candidate_developing_confirmed_then_ttl_close_and_cooldown():
    o = span({}, 2, 20, req=True)
    o[6]["conf"] = True
    results = run_script(o)
    assert states(results)[:12] == ["inactive", "inactive", "candidate", "candidate", "developing",
                                    "developing", "confirmed", "confirmed", "confirmed", "confirmed",
                                    "inactive", "inactive"]
    closing = results[9]
    assert closing.expired_at == closing.timestamp and closing.confirmed_at == results[6].timestamp
    assert results[12].state == "candidate"  # cooldown of 2 bars passed, required still true


def test_single_bar_dropout_within_grace_does_not_flicker():
    o = span({}, 2, 20, req=True)
    o[5]["req"] = False
    o[8]["conf"] = True
    results = run_script(o)
    assert results[5].state in ("candidate", "developing")
    assert results[8].state == "confirmed"
    assert len({r.first_detected_at for r in results[2:9]}) == 1


def test_expires_after_grace_lapses():
    o = span({}, 2, 4, req=True)
    results = run_script(o)
    # lapses at bars 4, 5, 6: the third exceeds grace_bars=2
    assert results[6].state == "expired" and results[6].expired_at == results[6].timestamp
    assert results[7].state == "inactive"


def test_expires_after_max_pending():
    results = run_script(span({}, 2, 20, req=True))
    assert results[8].state == "expired"  # 7 active bars > max_pending_bars=6
    assert results[8].bars_active == 7


def test_invalidation_from_pending_and_from_confirmed():
    o = span({}, 2, 20, req=True)
    o[4]["inv"] = True
    assert run_script(o)[4].state == "invalidated"
    o = span({}, 2, 20, req=True)
    o[4]["conf"] = True
    o[6]["inv"] = True
    results = run_script(o)
    assert results[6].state == "invalidated" and results[6].confirmed_at == results[4].timestamp


def test_confirmation_needs_minimum_score():
    o = span({}, 2, 20, req=True)
    o[4].update(conf=True, req=False, strong=False)  # score 0 on the confirming bar
    assert run_script(o)[4].state != "confirmed"


def test_deterministic_and_only_allowed_transitions():
    o = span({}, 2, 20, req=True)
    o[7]["conf"] = True
    o[15]["inv"] = True
    first, second = run_script(o), run_script(o)
    assert [r.as_dict() for r in first] == [r.as_dict() for r in second]
    for a, b in zip(first, first[1:]):
        allowed = {"inactive"} if a.state == "confirmed" and a.expired_at else ALLOWED[a.state]
        assert b.state in allowed and b.state in STATES


def test_history_never_exposes_future_rows():
    rows = make_rows(5)
    h = History(rows, 2)
    assert h.at(3) is None and h.ago(-1) is None and len(h.window(10)) == 3
    assert h.now is rows[2]


def test_unknown_evidence_name_is_rejected():
    with pytest.raises(KeyError):
        Scripted("bearish", "T", "1Day").ev("rsi_overbought", "momentum", True)
