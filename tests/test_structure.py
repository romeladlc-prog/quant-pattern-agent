"""Deterministic phase 6 behavior and prefix-causality tests."""
import unittest

import numpy as np
import pandas as pd

from src.structure.acceleration import acceleration_features
from src.structure.breakouts import detect_breakouts_fixed_levels
from src.structure.compression import compression_features
from src.structure.gaps import detect_gaps
from src.structure.relative_strength import relative_strength
from src.structure.swings import confirmed_swings
from src.structure.trend_structure import classify_swings, structural_state


def bars_from_close(values, volumes=None, opens=None):
    close=np.asarray(values,dtype=float)
    op=np.asarray(opens if opens is not None else close,dtype=float)
    volume=np.asarray(volumes if volumes is not None else np.full(len(close),1000.),dtype=float)
    return pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=len(close),freq="D",tz="UTC"),
        "open":op,"high":np.maximum(op,close)+.2,"low":np.minimum(op,close)-.2,
        "close":close,"volume":volume})


def fake_levels(price, timestamp):
    return pd.DataFrame([{"price_level":price,"available_at":timestamp}])


class StructureTests(unittest.TestCase):
    def test_uptrend_downtrend_and_range(self):
        def state(highs,lows):
            rows=[]
            for i,(h,l) in enumerate(zip(highs,lows)):
                for kind,price,offset in (("swing_high",h,0),("swing_low",l,1)):
                    rows.append({"confirmation_index":i*2+offset,"bar_index":i*2+offset,
                                 "type":kind,"price_level":price})
            return structural_state(classify_swings(pd.DataFrame(rows)))
        self.assertEqual(state([11,12,13],[8,9,10]),"uptrend")
        self.assertEqual(state([13,12,11],[10,9,8]),"downtrend")
        self.assertEqual(state([12,12,12],[8,8,8]),"range")
        self.assertEqual(state([11],[8]),"insufficient_swings")

    def test_swing_confirmation_latency(self):
        frame=bars_from_close([10,11,14,12,10,9,12])
        pivots=confirmed_swings(frame,2,2)
        self.assertTrue((pivots.confirmation_index-pivots.bar_index==2).all())
        self.assertTrue((pivots.confirmed_at>pivots.timestamp).all())
        self.assertTrue(pivots.loc[pivots.type.eq("swing_high"),"bar_index"].eq(2).any())
        self.assertTrue(confirmed_swings(frame.iloc[:4],2,2).empty)

    def test_real_and_fake_breakout(self):
        prefix=[100.]*25
        confirmed=bars_from_close(prefix+[102,103,104],volumes=[1000.]*25+[3000.]*3)
        levels=fake_levels(101.,confirmed.timestamp.iloc[20])
        events=detect_breakouts_fixed_levels(confirmed,levels,confirm_bars=2)
        self.assertIn("breakout_confirmed",set(events.state))
        failed=bars_from_close(prefix+[102,100,99],volumes=[1000.]*25+[3000.]*3)
        events=detect_breakouts_fixed_levels(failed,levels)
        fake=events.loc[events.state.eq("fake_breakout_up")]
        self.assertEqual(int(fake.iloc[0].failure_bars),1)
        self.assertEqual(fake.iloc[0].timestamp,failed.timestamp.iloc[26])

    def test_compression_then_expansion(self):
        rng=np.random.default_rng(7)
        stable=100+np.cumsum(rng.normal(0,.2,90))
        quiet=stable[-1]+np.cumsum(rng.normal(0,.002,40))
        volatile=quiet[-1]+np.cumsum(rng.normal(0,2,40))
        frame=bars_from_close(np.r_[stable,quiet,volatile])
        features=compression_features(frame,window=10,history=30)
        self.assertTrue(features.volatility_compression.iloc[90:130].any())
        self.assertTrue(features.volatility_expansion.iloc[130:].any())
        self.assertGreater(features.compression_duration.max(),0)

    def test_gap_and_fill(self):
        frame=bars_from_close([100,104,103,100],opens=[100,105,103,100])
        events=detect_gaps(frame,min_gap_pct=.02)
        gap=events.loc[events.type.eq("gap_up") & events.state.eq("unfilled_gap")].iloc[0]
        fill=events.loc[events.type.eq("gap_up") & events.state.eq("gap_fill")].iloc[0]
        self.assertEqual(gap.timestamp,frame.timestamp.iloc[1])
        self.assertGreater(fill.timestamp,gap.timestamp)
        self.assertEqual(fill.time_to_fill_bars,2)

    def test_prefix_causality(self):
        frame=bars_from_close(100+np.cumsum(np.random.default_rng(3).normal(0,1,100)))
        first=frame.iloc[:80].copy()
        pivots=confirmed_swings(first,3,2)
        later=confirmed_swings(frame,3,2)
        pd.testing.assert_frame_equal(pivots,later.loc[later.confirmation_index.lt(80)].reset_index(drop=True))
        for fn in (compression_features,acceleration_features):
            a=fn(first); b=fn(frame).iloc[:80].reset_index(drop=True)
            pd.testing.assert_frame_equal(a.reset_index(drop=True),b,check_dtype=False)
        rs_a=relative_strength(first,first)
        rs_b=relative_strength(frame,frame).iloc[:80].reset_index(drop=True)
        pd.testing.assert_frame_equal(rs_a.reset_index(drop=True),rs_b,check_dtype=False)


if __name__=="__main__": unittest.main()
