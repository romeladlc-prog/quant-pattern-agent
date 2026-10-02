"""Validation checks shared by validate_patterns.py and the tests.

Each check raises ``AssertionError`` with a precise message. ``prefix_check``
is the leakage test: results stored for dates <= T must be identical whether
the engine saw data up to T only, or T plus the future.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .config import BREAKOUT_MAX_AGE_BARS, COMPONENTS, STATES
from .engine import PatternRun

ALLOWED = {
    "inactive": {"inactive", "candidate"},
    "candidate": {"candidate", "developing", "confirmed", "invalidated", "expired"},
    "developing": {"developing", "confirmed", "invalidated", "expired"},
    "confirmed": {"confirmed", "invalidated"},
    "invalidated": {"inactive"},
    "expired": {"inactive"},
}
FORBIDDEN_NAMES = ("probability", "expected_return", "profit", "signal", "buy", "sell")


def check_transitions(run: PatternRun) -> int:
    by_pattern: dict[str, list] = {}
    for result in run.results:
        by_pattern.setdefault(result.pattern_name, []).append(result)
    checked = 0
    for name, items in by_pattern.items():
        for previous, current in zip(items, items[1:]):
            assert current.bar_index == previous.bar_index + 1, f"{name}: bars out of order"
            allowed = ALLOWED[previous.state]
            if previous.state == "confirmed" and previous.expired_at is not None:
                allowed = {"inactive"}  # confirmed episode aged out on the previous bar
            assert current.state in allowed, \
                f"{name} @ {current.timestamp}: {previous.state} -> {current.state}"
            checked += 1
    return checked


def check_results(run: PatternRun, *, context_close: pd.Series | None = None) -> dict:
    closes = run.evidence.close_time.reset_index(drop=True)
    counts = {state: 0 for state in STATES}
    for r in run.results:
        counts[r.state] += 1
        assert r.state in STATES, r.state
        assert 0.0 <= r.score <= 1.0, f"{r.pattern_name}: score {r.score}"
        for component in COMPONENTS:
            value = getattr(r, f"score_{component}")
            assert value is None or 0.0 <= value <= 1.0, f"{r.pattern_name}: {component} {value}"
        assert isinstance(r.evidence_for, list) and isinstance(r.evidence_against, list)
        assert not set(r.evidence_for) & set(r.evidence_against), r.pattern_name
        if r.state != "inactive":
            assert r.first_detected_at is not None and r.first_detected_at <= r.timestamp
            assert r.bars_active >= 1
        for stamp in (r.confirmed_at, r.invalidated_at, r.expired_at):
            assert stamp is None or stamp <= r.timestamp, f"{r.pattern_name}: future timestamp"
        if r.state == "confirmed":
            assert r.confirmed_at is not None
        if r.state == "invalidated":
            assert r.invalidated_at == r.timestamp
        asof = r.details.get("level_asof_timestamp")
        start = r.details.get("breakout_timestamp")
        if asof is not None and start is not None:
            assert asof < start, f"{r.pattern_name}: level not known before breakout"
        external = r.external_context
        close = closes[r.bar_index]
        for key in ("vix_available_at", "breadth_available_at"):
            if external.get(key) is not None:
                assert pd.Timestamp(external[key]) <= close, f"{key} after bar close"
        age = r.structure_context.get("breakout_age")
        assert age is None or age <= BREAKOUT_MAX_AGE_BARS, "stale breakout used"
        for name in r.as_dict():
            assert not any(word in name for word in FORBIDDEN_NAMES), name
    for prefix in ("w1", "d1", "h4", "h1"):
        column = f"{prefix}_close_time"
        if column in run.evidence:
            other = pd.to_datetime(run.evidence[column], utc=True)
            assert (other.isna() | (other <= run.evidence.close_time)).all(), f"{column} after close"
    transitions = check_transitions(run)
    return {"results": len(run.results), "transitions_checked": transitions, **counts}


def normalized(results) -> dict:
    return {(r.pattern_name, r.bar_index): r.as_dict() for r in results}


def prefix_check(run_full: PatternRun, run_prefix: Callable[[pd.Timestamp], PatternRun],
                 cut_positions: list[int]) -> list[dict]:
    """Compare stored results for every bar <= T between a run that saw only
    data up to T and the full run. Any difference means look-ahead."""
    reports = []
    full = normalized(run_full.results)
    for position in cut_positions:
        stamp = run_full.evidence.timestamp.iloc[position]
        prefix = run_prefix(stamp)
        mine = normalized(prefix.results)
        differences = [key for key, value in mine.items() if full.get(key) != value]
        assert not differences, \
            f"look-ahead: {len(differences)} results differ at T={stamp}, e.g. {differences[:3]}"
        reports.append({"T": stamp, "compared": len(mine)})
    return reports
