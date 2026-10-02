"""Small evidence helpers shared by detectors. Inputs are evidence rows."""
from __future__ import annotations

from .base import clean, num
from .config import THRESHOLDS

TREND = {1: "uptrend", -1: "downtrend"}


def sgn_value(value: float | None, sign: int) -> float | None:
    return None if value is None else value*sign


def gt(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value > threshold


def ge(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value >= threshold


def lt(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value < threshold


def le(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value <= threshold


def any_true(*values) -> bool | None:
    """True if any is True; None only if every input is missing."""
    if any(v is True for v in values):
        return True
    return None if all(v is None for v in values) else False


def higher_tf_state(row: dict) -> str | None:
    state = clean(row.get("htf_structural_state"))
    return state if isinstance(state, str) else None


def higher_tf_is(row: dict, trend: str) -> bool | None:
    state = higher_tf_state(row)
    return None if state is None else state == trend


def near_opposing_level(row: dict, move: int) -> bool | None:
    """Resistance near an up move, support near a down move (as-of levels)."""
    distance = num(row, "resistance_dist_atr" if move > 0 else "support_dist_atr")
    return le(distance, THRESHOLDS["near_level_atr"])


def volatility_expanding(row: dict) -> bool | None:
    return any_true(clean(row.get("volatility_expansion")),
                    ge(num(row, "atr_expansion"), THRESHOLDS["atr_expansion"]))


def regime_values(row: dict) -> tuple[float | None, float | None, float | None, float | None]:
    return (num(row, "hmm_p_high"), num(row, "hmm_p_high_lag"),
            num(row, "markov_p_high"), num(row, "markov_p_high_lag"))


def regime_shift(row: dict) -> float | None:
    """Largest |p_high - p_high_lag| across HMM2 and Markov2, if available."""
    hmm, hmm_lag, mk, mk_lag = regime_values(row)
    shifts = [abs(a-b) for a, b in ((hmm, hmm_lag), (mk, mk_lag)) if a is not None and b is not None]
    return max(shifts) if shifts else None


def regime_stable_low(row: dict) -> bool | None:
    hmm, _, mk, _ = regime_values(row)
    values = [v for v in (hmm, mk) if v is not None]
    if not values:
        return None
    shift = regime_shift(row)
    return all(v < THRESHOLDS["regime_probability"] for v in values) and \
        (shift is None or shift < THRESHOLDS["regime_shift"])


def high_vol_rising(row: dict) -> bool | None:
    hmm, hmm_lag, mk, mk_lag = regime_values(row)
    rises = [a-b for a, b in ((hmm, hmm_lag), (mk, mk_lag)) if a is not None and b is not None]
    return None if not rises else max(rises) >= THRESHOLDS["regime_shift"]


def change_point_recent(row: dict) -> bool | None:
    return any_true(clean(row.get("cp_rv20_recent")), clean(row.get("cp_slope_recent")))


def breadth_supportive(row: dict, sign: int) -> bool | None:
    value = num(row, "breadth_sma50")
    if value is None:
        return None
    return value >= THRESHOLDS["breadth_high"] if sign > 0 else value <= THRESHOLDS["breadth_low"]


def vix_supportive(row: dict, sign: int) -> bool | None:
    z = num(row, "vix_zscore")
    if z is None:
        return None
    return z < THRESHOLDS["vix_zscore_high"] if sign > 0 else z > THRESHOLDS["vix_zscore_high"]
