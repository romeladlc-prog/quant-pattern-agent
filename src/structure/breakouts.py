"""Causal close-based breakout events, including observed failures."""
from __future__ import annotations

import pandas as pd
from .swings import atr


def detect_breakouts(bars: pd.DataFrame, levels: pd.DataFrame, min_atr_distance: float = .1,
                     min_relative_volume: float = 1., confirm_bars: int = 2,
                     fail_within: int = 5) -> pd.DataFrame:
    columns = ["timestamp", "level", "direction", "state", "distance_atr",
               "relative_volume", "outside_bars", "retest", "failure_bars",
               "breakout_amplitude", "confidence"]
    if bars.empty or levels.empty: return pd.DataFrame(columns=columns)
    av = atr(bars)
    rv = bars.volume/bars.volume.shift().rolling(20, min_periods=5).mean()
    events = []
    for level in levels.itertuples():
        price = float(level.price_level)
        for i in range(1, len(bars)):
            if bars.timestamp.iloc[i] < level.available_at or pd.isna(av.iloc[i]) or av.iloc[i]<=0:
                continue
            prev, close = float(bars.close.iloc[i-1]), float(bars.close.iloc[i])
            direction = 1 if prev <= price < close else -1 if prev >= price > close else 0
            if not direction: continue
            distance = abs(close-price)/float(av.iloc[i])
            if distance < min_atr_distance: continue
            relative = float(rv.iloc[i]) if pd.notna(rv.iloc[i]) else None
            volume_ok = relative is not None and relative >= min_relative_volume
            base = {"level":price, "direction":"up" if direction==1 else "down",
                    "distance_atr":distance, "relative_volume":relative,
                    "breakout_amplitude":abs(close-price)}
            events.append({**base, "timestamp":bars.timestamp.iloc[i], "state":"breakout_candidate",
                "outside_bars":1, "retest":False, "failure_bars":None,
                "confidence":round(min(1., .25+.25*min(distance,1)+.25*volume_ok),3)})
            retest = False
            for j in range(i+1, min(len(bars),i+max(confirm_bars,fail_within)+1)):
                c = float(bars.close.iloc[j])
                if direction*(c-price) <= 0:
                    if j-i <= fail_within:
                        events.append({**base, "timestamp":bars.timestamp.iloc[j],
                            "state":"fake_breakout_up" if direction==1 else "fake_breakout_down",
                            "outside_bars":j-i, "retest":retest, "failure_bars":j-i,
                            "confidence":round(min(1.,.5+.25*min(distance,1)+.25*(j-i<=2)),3)})
                    break
                if (direction==1 and bars.low.iloc[j]<=price or
                    direction==-1 and bars.high.iloc[j]>=price): retest=True
                if j-i+1 == confirm_bars and volume_ok:
                    events.append({**base, "timestamp":bars.timestamp.iloc[j],
                        "state":"breakout_confirmed", "outside_bars":confirm_bars,
                        "retest":retest, "failure_bars":None,
                        "confidence":round(min(1.,.5+.25*min(distance,1)+.25*retest),3)})
    return pd.DataFrame(events, columns=columns).sort_values("timestamp").reset_index(drop=True)
