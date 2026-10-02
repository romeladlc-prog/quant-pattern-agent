"""breakout_state is current only within breakout_max_age_bars."""
import numpy as np
import pytest

from src.structure.market_structure import make_structure_snapshot
from synthetic import anchored, bars_from_close


def breakout_then_flat(extra):
    base = anchored({0: 100, 5: 110, 10: 100, 15: 110, 20: 100, 24: 113})
    tail = 113. + .01*np.arange(1, extra+1)
    return bars_from_close(np.r_[base.close.to_numpy(), tail])


def test_recent_breakout_is_active():
    snap = make_structure_snapshot("TEST", breakout_then_flat(1), breakout_max_age_bars=5)
    assert snap.breakout_state in {"breakout_candidate", "breakout_confirmed"}
    assert snap.latest_breakout_event["age_bars"] <= 5


def test_old_breakout_becomes_inactive_but_is_kept():
    snap = make_structure_snapshot("TEST", breakout_then_flat(30), breakout_max_age_bars=5)
    assert snap.breakout_state == "inactive"
    assert snap.latest_breakout_event is not None
    assert snap.latest_breakout_event["age_bars"] > 5


def test_age_threshold_is_inclusive_and_configurable():
    bars = breakout_then_flat(30)
    age = make_structure_snapshot("TEST", bars).latest_breakout_event["age_bars"]
    assert make_structure_snapshot("TEST", bars, breakout_max_age_bars=age).breakout_state != "inactive"
    assert make_structure_snapshot("TEST", bars, breakout_max_age_bars=age-1).breakout_state == "inactive"
    with pytest.raises(ValueError):
        make_structure_snapshot("TEST", bars, breakout_max_age_bars=-1)


def test_no_events_is_inactive():
    snap = make_structure_snapshot("TEST", bars_from_close(np.linspace(100, 101, 40)))
    assert snap.breakout_state == "inactive" and snap.latest_breakout_event is None
    assert snap.fake_breakout_state == "inactive"
