"""Prefix-divergence diagnosis for ``validate_patterns.py --debug-prefix``.

When ``prefix_check`` fails, this finds, for the earliest bar whose stored
result differs between run A (data up to T) and run B (full sample):

* which patterns and which ``PatternResult`` fields differ (nested keys too);
* the first *input* that differs: the earliest (bar, evidence column) in
  pipeline order, including columns present in one run only (schema drift);
* the module that produces that input.

It only reads two finished runs; it never changes a check or a result.
"""
from __future__ import annotations

import math

import pandas as pd

from .base import clean
from .checks import normalized
from .engine import PatternRun
from .external_context import EXTERNAL_COLUMNS
from .quant_context import QUANT_COLUMNS

# Evidence columns -> producing module, in pipeline order (first match wins).
SOURCES: list[tuple[str, tuple[str, ...]]] = [
    ("bars (complete_bars / OHLCV)", ("timestamp", "open", "high", "low", "close", "volume",
                                      "close_time")),
    ("features (statistical_features, ATR, rv20_percentile)",
     ("atr", "log_return", "zscore_20", "momentum_5", "momentum_20", "realized_volatility_20",
      "rolling_slope_20", "rolling_mean_20", "slope_20_pct", "rv20_percentile")),
    ("structure.acceleration", ("slope", "slope_change", "price_zscore", "distance_to_mean",
                                "cumulative_log_return", "relative_volume", "atr_expansion",
                                "acceleration_state", "exhaustion_candidate", "accel_up_recent",
                                "accel_down_recent")),
    ("structure.compression", ("volatility_compression", "volatility_expansion",
                               "compression_duration", "compression_recent")),
    ("market structure snapshot (swings, structure_asof)",
     ("structural_state", "last_high_label", "last_low_label", "last_high", "last_high_index",
      "prev_high", "prev_high_index", "last_low", "last_low_index", "prev_low", "prev_low_index",
      "new_swing_high", "new_swing_low", "structure_changed_recent")),
    ("levels as-of (levels_history)", ("nearest_support", "nearest_resistance", "support_touches",
                                       "resistance_touches", "support_dist_atr",
                                       "resistance_dist_atr", "levels_asof_timestamp")),
    ("breakout state (detect_breakouts)", ("breakout_state", "breakout_direction", "breakout_age",
                                           "fake_breakout_state", "fake_breakout_age")),
    ("gaps", ("gap_up_recent", "gap_down_recent")),
    ("relative strength", ("relative_return_5", "relative_return_20", "price_benchmark_ratio",
                           "ratio_slope", "ratio_zscore", "benchmark_available")),
    ("HMM2 (quant_context)", ("hmm_p_high", "hmm_p_high_lag", "hmm_fit_converged")),
    ("Markov2 (quant_context)", ("markov_p_high", "markov_p_high_lag", "markov_fit_converged")),
    ("Kalman local level (quant_context)", ("kalman_level", "kalman_level_change",
                                            "kalman_level_change_lag", "kalman_fit_converged")),
    ("GARCH (quant_context)", ("garch_vol", "garch_vol_ratio", "garch_fit_converged")),
    ("change points PELT (quant_context)", ("cp_rv20_recent", "cp_slope_recent", "cp_rv20_age",
                                            "cp_slope_age")),
    ("complexity (quant_context)", ("hurst", "permutation_entropy", "quant_fit_index")),
    ("external context (available_at as-of)", tuple(EXTERNAL_COLUMNS)),
    ("weekly context (w1_*)", ("w1_",)),
    ("daily context (d1_*)", ("d1_",)),
    ("4Hour context (h4_*)", ("h4_",)),
    ("1Hour context (h1_*)", ("h1_",)),
    ("higher timeframe state", ("htf_structural_state",)),
]
_KNOWN = set(QUANT_COLUMNS)


def source_of(column: str) -> str:
    for name, keys in SOURCES:
        if column in keys or any(k.endswith("_") and column.startswith(k) for k in keys):
            return name
    return "quant_context" if column in _KNOWN else "unknown"


def _order(column: str) -> int:
    name = source_of(column)
    return next((k for k, (n, _) in enumerate(SOURCES) if n == name), len(SOURCES))


def same(a, b) -> bool:
    a, b = clean(a), clean(b)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, pd.Timestamp) or isinstance(b, pd.Timestamp):
        return (a is None and b is None) or (a is not None and b is not None
                                             and pd.Timestamp(a) == pd.Timestamp(b))
    return a == b


def field_differences(a, b, path: str = "") -> list[str]:
    """Dotted paths where two ``as_dict()`` values differ (missing key included)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            where = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out.append(f"{where} (only in {'prefix' if key in a else 'full'})")
            else:
                out += field_differences(a[key], b[key], where)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for k, (x, y) in enumerate(zip(a, b)) for p in field_differences(x, y, f"{path}[{k}]")]
    return [] if same(a, b) else [path or "<value>"]


def input_differences(full: pd.DataFrame, prefix: pd.DataFrame, upto: int | None = None) -> list[dict]:
    """Per evidence column, the first bar (<= upto) where prefix and full disagree."""
    n = len(prefix) if upto is None else min(upto + 1, len(prefix))
    found = []
    for column in sorted(set(full.columns) | set(prefix.columns), key=lambda c: (_order(c), c)):
        if column not in full or column not in prefix:
            found.append({"bar": 0, "column": column, "source": source_of(column),
                          "kind": f"schema: column only in {'prefix' if column in prefix else 'full'}",
                          "prefix": None, "full": None})
            continue
        for i in range(n):
            x, y = prefix[column].iloc[i], full[column].iloc[i]
            if not same(x, y):
                found.append({"bar": i, "column": column, "source": source_of(column),
                              "kind": "value", "prefix": clean(x), "full": clean(y)})
                break
    return sorted(found, key=lambda d: (d["bar"], _order(d["column"]), d["column"]))


def _attrs(run: PatternRun) -> dict:
    return run.meta.get("quant_attrs", {}) or {}


def diagnose_prefix(full: PatternRun, prefix: PatternRun) -> dict:
    """Granular A (prefix, data <= T) vs B (full) comparison at the first divergent bar."""
    a, b = normalized(prefix.results), normalized(full.results)
    keys = [k for k, v in a.items() if b.get(k) != v]
    report = {"ticker": full.ticker, "timeframe": full.timeframe,
              "T": prefix.evidence.timestamp.iloc[-1] if len(prefix.evidence) else None,
              "differing_results": len(keys), "compared": len(a)}
    if not keys:
        return report
    first_bar = min(bar for _, bar in keys)
    at_bar = sorted(name for name, bar in keys if bar == first_bar)
    fields: dict[str, list[str]] = {}
    for name in at_bar:
        fields[name] = field_differences(a[(name, first_bar)], b[(name, first_bar)])
    inputs = input_differences(full.evidence, prefix.evidence)
    full_attrs, prefix_attrs = _attrs(full), _attrs(prefix)
    t_bar = len(prefix.evidence) - 1
    refit_full = [r for r in full_attrs.get("refit_positions", []) if r <= t_bar]
    diag_full = [d for d in full_attrs.get("fit_diagnostics", []) if d["refit"] <= t_bar]
    report.update({
        "first_bar": first_bar,
        "first_timestamp": prefix.evidence.timestamp.iloc[first_bar],
        "patterns_at_first_bar": at_bar,
        "fields_at_first_bar": fields,
        "first_input": inputs[0] if inputs else None,
        "inputs": inputs,
        "context_timeframes": {"prefix": prefix.meta.get("context_timeframes"),
                               "full": full.meta.get("context_timeframes")},
        "quant_refits_equal": prefix_attrs.get("refit_positions", []) == refit_full,
        "quant_fit_diagnostics_equal": prefix_attrs.get("fit_diagnostics", []) == diag_full,
        "not_converged_fits_upto_T": [d for d in diag_full if not d["converged"]],
    })
    return report


def format_report(report: dict, limit: int = 12) -> str:
    lines = [f"[debug-prefix] {report['ticker']} {report['timeframe']} T={report['T']}: "
             f"{report['differing_results']}/{report['compared']} results differ"]
    if not report["differing_results"]:
        return lines[0]
    first = report["first_input"]
    lines.append(f"  first divergent bar: {report['first_bar']} ({report['first_timestamp']})")
    lines.append(f"  patterns at that bar: {', '.join(report['patterns_at_first_bar'])}")
    sample = report["patterns_at_first_bar"][0]
    fields = report["fields_at_first_bar"][sample]
    lines.append(f"  fields differing for {sample}: {', '.join(fields[:limit])}"
                 + (" ..." if len(fields) > limit else ""))
    if first is None:
        lines.append("  first divergent input: none in the evidence frame (state machine / detector)")
    else:
        lines.append(f"  first divergent input: {first['column']} at bar {first['bar']} "
                     f"[{first['kind']}] prefix={first['prefix']!r} full={first['full']!r}")
        lines.append(f"  origin: {first['source']}")
    by_source: dict[str, list[str]] = {}
    for item in report["inputs"]:
        by_source.setdefault(item["source"], []).append(f"{item['column']}@{item['bar']}")
    for source, columns in by_source.items():
        lines.append(f"  - {source}: {', '.join(columns[:limit])}" + (" ..." if len(columns) > limit else ""))
    lines.append(f"  context timeframes prefix/full: {report['context_timeframes']['prefix']} / "
                 f"{report['context_timeframes']['full']}")
    lines.append(f"  quant refits equal up to T: {report['quant_refits_equal']}; "
                 f"fit diagnostics equal: {report['quant_fit_diagnostics_equal']}; "
                 f"non-converged fits up to T: {len(report['not_converged_fits_upto_T'])}")
    return "\n".join(lines)


def external_rows_report(full_inputs: dict | None, prefix_inputs: dict | None, limit: int = 1) -> list[str]:
    """Per external family: rows given to each run and the first row only the full run saw."""
    lines = []
    for family in sorted(set(full_inputs or {}) | set(prefix_inputs or {})):
        full, prefix = (full_inputs or {}).get(family), (prefix_inputs or {}).get(family)
        if full is None or prefix is None:
            lines.append(f"  external {family}: supplied to {'full' if prefix is None else 'prefix'} only")
            continue
        lines.append(f"  external {family}: rows prefix/full = {len(prefix)}/{len(full)}")
        extra = full.loc[~full.index.isin(prefix.index)]
        if extra.empty:
            continue
        columns = [c for c in ("series", "observation_date", "release_date", "available_at", "asof_safe")
                   if c in extra]
        first = extra.sort_values("available_at").head(limit) if "available_at" in extra else extra.head(limit)
        for row in first[columns].to_dict("records"):
            lines.append(f"    first row only in full: {row}")
    return lines
