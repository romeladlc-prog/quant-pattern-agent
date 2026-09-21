"""Transparent HH/HL/LH/LL classification from confirmed pivots."""
from __future__ import annotations

import pandas as pd


def classify_swings(swings: pd.DataFrame, equality_pct: float = .005) -> pd.DataFrame:
    if equality_pct < 0:
        raise ValueError("equality_pct must be nonnegative")
    out = swings.sort_values(["confirmation_index", "bar_index"]).copy()
    out["structure_label"] = None
    for kind, high_label, low_label in (("swing_high", "HH", "LH"),
                                        ("swing_low", "HL", "LL")):
        subset = out.loc[out.type.eq(kind)]
        previous = subset.price_level.shift()
        labels = ["EQ" if abs(current-prior)/prior <= equality_pct else
                  high_label if current > prior else low_label
                  for current, prior in zip(subset.price_level, previous) if pd.notna(prior)]
        labels = [None] + labels if len(subset) else []
        out.loc[subset.index, "structure_label"] = labels
    return out


def structural_state(classified: pd.DataFrame, asof_index: int | None = None) -> str:
    known = classified if asof_index is None else classified.loc[
        classified.confirmation_index.le(asof_index)]
    highs = known.loc[known.type.eq("swing_high"), "structure_label"].dropna().tail(2).tolist()
    lows = known.loc[known.type.eq("swing_low"), "structure_label"].dropna().tail(2).tolist()
    if not highs or not lows:
        return "insufficient_swings"
    h, l = highs[-1], lows[-1]
    if h == "HH" and l == "HL": return "uptrend"
    if h == "LH" and l == "LL": return "downtrend"
    if h == "EQ" or l == "EQ" or (h == "LH" and l == "HL"): return "range"
    return "transition"
