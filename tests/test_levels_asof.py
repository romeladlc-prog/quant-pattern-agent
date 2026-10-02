"""Phase 6.1: levels and breakouts are rebuilt as of each bar."""
import pandas as pd
import pytest

from src.structure.breakouts import detect_breakouts
from src.structure.levels import build_levels_asof, cluster_levels, levels_history
from src.structure.swings import confirmed_swings
from synthetic import anchored, random_walk

CUTS = (40, 75, 110, 159)


def test_levels_history_equals_per_bar_cluster():
    bars = random_walk()
    swings = confirmed_swings(bars)
    history = levels_history(bars, swings)
    for end in range(0, len(bars), 7):
        expected = cluster_levels(bars, swings, end)
        got = history.loc[history.asof_index.eq(end)].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, expected, check_dtype=False)


@pytest.mark.parametrize("cut", CUTS)
def test_future_bars_do_not_change_past_levels(cut):
    bars = random_walk()
    prefix = bars.iloc[:cut].reset_index(drop=True)
    stamp = prefix.timestamp.iloc[-1]
    from_prefix = build_levels_asof(prefix, stamp)
    from_full = build_levels_asof(bars, stamp)
    pd.testing.assert_frame_equal(from_prefix, from_full, check_dtype=False)
    history = levels_history(bars, confirmed_swings(bars))
    pd.testing.assert_frame_equal(
        history.loc[history.asof_index.eq(cut-1)].reset_index(drop=True), from_prefix,
        check_dtype=False)


def test_supplied_future_swings_are_ignored():
    bars = random_walk()
    full_swings = confirmed_swings(bars)
    stamp = bars.timestamp.iloc[80]
    levels = build_levels_asof(bars, stamp, swings=full_swings)
    assert levels.available_at.le(stamp).all()
    pd.testing.assert_frame_equal(levels, build_levels_asof(bars.iloc[:81], stamp),
                                  check_dtype=False)


def test_asof_between_labels_uses_last_known_bar():
    bars = random_walk()
    stamp = bars.timestamp.iloc[60] + pd.Timedelta(hours=12)
    levels = build_levels_asof(bars, stamp)
    assert levels.asof_index.eq(60).all()
    assert build_levels_asof(bars, bars.timestamp.iloc[0] - pd.Timedelta(days=1)).empty


def test_levels_respect_confirmed_at():
    # Highs at bars 5 and 15; the second is confirmed only at bar 17.
    bars = anchored({0: 100, 5: 110, 10: 100, 15: 110, 20: 100, 30: 104})
    swings = confirmed_swings(bars)
    before = build_levels_asof(bars, bars.timestamp.iloc[16], swings=swings)
    after = build_levels_asof(bars, bars.timestamp.iloc[17], swings=swings)
    assert not ((before.type == "resistance") & before.price_level.between(110, 110.5)).any()
    assert ((after.type == "resistance") & after.price_level.between(110, 110.5)).any()


@pytest.mark.parametrize("cut", CUTS)
def test_breakouts_are_prefix_invariant(cut):
    bars = random_walk()
    prefix = bars.iloc[:cut].reset_index(drop=True)
    full = detect_breakouts(bars)
    early = full.loc[full.timestamp.le(prefix.timestamp.iloc[-1])].reset_index(drop=True)
    pd.testing.assert_frame_equal(detect_breakouts(prefix), early, check_dtype=False)
    assert full.level_asof_timestamp.lt(full.timestamp).all()


def test_breakout_uses_level_known_before_crossing():
    # Resistance 110.2 exists after bar 17; a later high (bar 40) moves the
    # full-sample cluster. The crossing near bar 24 must use the as-of 110.2.
    bars = anchored({0: 100, 5: 110, 10: 100, 15: 110, 20: 100, 25: 116,
                     30: 105, 40: 110.6, 45: 100})
    final = cluster_levels(bars, confirmed_swings(bars))
    final_resistance = final.loc[final.price_level.between(109, 112), "price_level"]
    assert not final_resistance.empty and (final_resistance - 110.2).abs().min() > .1
    events = detect_breakouts(bars)
    up = events.loc[events.state.eq("breakout_candidate") & events.direction.eq("up")
                    & events.timestamp.lt(bars.timestamp.iloc[30])]
    assert not up.empty
    assert up.level.iloc[0] == pytest.approx(110.2)
    assert (up.level_asof_timestamp < up.timestamp).all()
