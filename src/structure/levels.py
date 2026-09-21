"""ATR/percent clustered swing levels; descriptive strength only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .swings import atr


def cluster_levels(bars: pd.DataFrame, swings: pd.DataFrame, asof_index: int | None = None,
                   atr_tolerance: float = 0.5, pct_tolerance: float = 0.005,
                   min_touches: int = 2) -> pd.DataFrame:
    columns = ["price_level", "type", "touch_count", "first_touch", "last_touch",
               "distance_from_current_price", "strength_score", "available_at"]
    if bars.empty or swings.empty: return pd.DataFrame(columns=columns)
    end = len(bars)-1 if asof_index is None else asof_index
    known = swings.loc[swings.confirmation_index.le(end)].sort_values("price_level")
    if known.empty: return pd.DataFrame(columns=columns)
    current = float(bars.close.iloc[end]); av = atr(bars.iloc[:end+1]).iloc[-1]
    tolerance = max(float(av)*atr_tolerance if pd.notna(av) else 0, current*pct_tolerance)
    groups = []
    for row in known.itertuples():
        eligible = [g for g in groups if abs(row.price_level-g["center"]) <= tolerance]
        if eligible:
            g = min(eligible, key=lambda item: abs(row.price_level-item["center"]))
            g["rows"].append(row)
            g["center"] = np.mean([r.price_level for r in g["rows"]])
        else: groups.append({"center": row.price_level, "rows": [row]})
    result = []
    for group in groups:
        rows = group["rows"]
        if len(rows) < min_touches: continue
        kinds = {r.type for r in rows}
        kind = "mixed" if len(kinds)>1 else ("resistance" if "swing_high" in kinds else "support")
        last_index = max(r.confirmation_index for r in rows)
        recency = 1/(1+(end-last_index)/20)
        volume = bars.volume.iloc[[r.bar_index for r in rows]].mean()
        base = bars.volume.iloc[:end+1].median()
        vol_factor = min(float(volume/base), 2) if base > 0 else 1
        price = float(group["center"])
        result.append({"price_level": price, "type": kind, "touch_count": len(rows),
          "first_touch": min(r.timestamp for r in rows), "last_touch": max(r.timestamp for r in rows),
          "distance_from_current_price": (price/current)-1, "strength_score":
          round(np.log1p(len(rows))*recency*vol_factor, 4),
          "available_at": max(r.confirmed_at for r in rows)})
    return pd.DataFrame(result, columns=columns).sort_values("price_level").reset_index(drop=True)
