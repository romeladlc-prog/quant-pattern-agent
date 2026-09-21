"""Regular-session daily gaps and fill events, dated when observed."""
from __future__ import annotations

import pandas as pd
from .swings import atr


def detect_gaps(bars: pd.DataFrame, min_gap_pct: float = .002,
                atr_window: int = 14) -> pd.DataFrame:
    columns = ["timestamp", "gap_timestamp", "type", "size_abs", "size_pct",
               "size_atr", "filled", "time_to_fill_bars", "state"]
    if len(bars)<2: return pd.DataFrame(columns=columns)
    av = atr(bars, atr_window)
    records=[]; active=[]
    for i in range(1,len(bars)):
        still=[]
        for gap in active:
            if (gap["direction"]==1 and bars.low.iloc[i]<=gap["previous"] or
                gap["direction"]==-1 and bars.high.iloc[i]>=gap["previous"]):
                records.append({**gap["fields"], "timestamp":bars.timestamp.iloc[i],
                    "filled":True, "time_to_fill_bars":i-gap["index"], "state":"gap_fill"})
            else: still.append(gap)
        active=still
        previous=float(bars.close.iloc[i-1]); opening=float(bars.open.iloc[i]); size=opening-previous
        if abs(size)/previous<min_gap_pct: continue
        fields={"gap_timestamp":bars.timestamp.iloc[i],
            "type":"gap_up" if size>0 else "gap_down", "size_abs":abs(size),
            "size_pct":abs(size)/previous,
            "size_atr":abs(size)/av.iloc[i-1] if pd.notna(av.iloc[i-1]) and av.iloc[i-1]>0 else None}
        records.append({**fields,"timestamp":bars.timestamp.iloc[i],"filled":False,
                        "time_to_fill_bars":None,"state":"unfilled_gap"})
        if (size>0 and bars.low.iloc[i]<=previous or size<0 and bars.high.iloc[i]>=previous):
            records.append({**fields,"timestamp":bars.timestamp.iloc[i],"filled":True,
                            "time_to_fill_bars":0,"state":"gap_fill"})
        else: active.append({"fields":fields,"direction":1 if size>0 else -1,
                             "previous":previous,"index":i})
    return pd.DataFrame(records,columns=columns).sort_values("timestamp",kind="stable").reset_index(drop=True)
