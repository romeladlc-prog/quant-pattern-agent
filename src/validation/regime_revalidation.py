"""Phase 4.1 regime revalidation: convergence, reproducibility and fold stability.

Re-runs HMM and Markov regime fits after three changes that make the original
Phase 4.1 regime numbers ``legacy_pre_revalidation``:

1. ``fit_markov_regimes`` passes ``rng`` to statsmodels >= 0.15 (before, the
   search starts were not reproducible);
2. ``fit_hmm_regimes`` fits with one OpenMP thread (KMeans init inside
   hmmlearn was not bit-reproducible with several threads);
3. HMM convergence uses ``fit_quality["converged"]``: hmmlearn's
   ``monitor_.converged`` is also True when EM stops at ``n_iter`` or when the
   log-likelihood went down.

Nothing here changes a model, a parameter or a selection rule. Each fold
trains on rows <= ``train_end`` and filters only up to its test row, so no
fold sees data after its test date.

Convergence (did the optimiser finish cleanly), reproducibility (same input
and seed give the same output) and stability (similar regimes across folds,
seeds and assets) are different properties. None of them means the regimes
are predictively useful.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import warnings

import numpy as np
import pandas as pd

from src.models.regime_models import REGIME_INPUTS, fit_hmm_regimes, fit_markov_regimes
from src.security import redact_secrets
from src.validation.walk_forward import WalkForwardConfig, walk_forward_splits

# Same input + same configuration (+ same seed) must agree within these limits.
# The fits are deterministic on one machine (exact equality is expected); the
# tolerances absorb BLAS/CPU differences between machines and library builds,
# far below anything that could change a state.
TOLERANCES = {
    "probability_atol": 1e-9,   # filtered state probabilities, absolute
    "parameter_atol": 1e-7,     # fitted parameters (standardised / percent units), absolute
    "loglik_rtol": 1e-9,        # train log-likelihood, relative
}
STATUSES = ("converged", "not_converged", "fit_failed")
MODELS = {"HMM": fit_hmm_regimes, "Markov": fit_markov_regimes}
FIT_COLUMNS = [
    "asset", "model", "n_states", "cut", "fold", "train_start", "train_end", "test_date",
    "n_train", "n_scope", "run", "seed", "status", "monitor_converged", "iterations",
    "hit_max_iter", "loglik_decreases", "seeds_not_converged", "train_loglik", "aic", "bic",
    "high_state", "final_regime", "p_high_final", "p_state_final", "obs_per_regime",
    "init_loglik_spread", "init_min_agreement", "error",
]
LEGACY_LABEL = "legacy_pre_revalidation"


@dataclass(frozen=True)
class RevalidationConfig:
    min_train: int = 252          # as Phase 4.1
    step: int = 63                # as Phase 4.1
    repeats: int = 3              # identical fits per fold (same seed)
    random_state: int = 42        # default seed of both fitters
    alt_seed_offset: int = 1      # one extra fit with seed+offset: initialisation sensitivity
    states: tuple[int, ...] = (2,)  # 3-state models stay experimental; opt-in only
    legacy_cut: bool = True       # also refit the single last cut Phase 4.1 used


@dataclass
class Fit:
    row: dict
    p_high: pd.Series | None = None
    params: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def cut_plan(index: pd.DatetimeIndex, cfg: RevalidationConfig) -> list[dict]:
    """Walk-forward cuts (expanding, as Phase 4.1) plus the legacy last cut.

    A walk-forward cut trains on rows [0, train_end] and filters rows up to the
    test row only. The legacy cut reproduces Phase 4.1: train on the first
    ``max(min_train, n - step)`` rows, filter the whole sample.
    """
    plan = []
    splits = walk_forward_splits(index, WalkForwardConfig(cfg.min_train, cfg.step, 1))
    for fold, (train, position) in enumerate(splits):
        plan.append({"cut": "walk_forward", "fold": fold, "train_stop": train.stop,
                     "scope_stop": position + 1})
    if cfg.legacy_cut:
        stop = max(cfg.min_train, len(index) - cfg.step)
        plan.append({"cut": "legacy_last_cut", "fold": -1, "train_stop": stop,
                     "scope_stop": len(index)})
    return plan


def _status(model: str, fit: dict) -> str:
    converged = fit["fit_quality"]["converged"] if model == "HMM" else fit["converged"]
    return "converged" if converged else "not_converged"


def _params(model: str, fit: dict, high: int) -> np.ndarray:
    if model == "HMM":
        m = fit["model"]
        return np.concatenate([m.means_.ravel(), m.covars_.ravel(), m.transmat_.ravel(),
                               m.startprob_.ravel()])
    return np.asarray(fit["result"].params, dtype=float)


def fit_once(asset: str, model: str, n_states: int, features: pd.DataFrame, cut: dict,
             seed: int, run: str) -> Fit:
    """One fit on ``features[:train_stop]``, filtered on ``features[:scope_stop]``."""
    train = features.iloc[:cut["train_stop"]]
    scope = features.iloc[:cut["scope_stop"]]
    row = {c: None for c in FIT_COLUMNS}
    row.update({"asset": asset, "model": f"{model}{n_states}", "n_states": n_states,
                "cut": cut["cut"], "fold": cut["fold"], "train_start": train.index[0],
                "train_end": train.index[-1], "test_date": scope.index[-1],
                "n_train": int(train[REGIME_INPUTS].dropna().shape[0]),
                "n_scope": int(scope[REGIME_INPUTS].dropna().shape[0]), "run": run, "seed": seed})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # quality is recorded explicitly below
            fit = MODELS[model](train, scope, n_states=n_states, random_state=seed)
    except Exception as error:  # recorded, never replaced by a fake fit
        return _failed(row, error)
    try:
        return _describe(model, fit, row)
    except Exception as error:  # a fit whose states cannot be described is not usable
        return _failed(row, error)


def _failed(row: dict, error: Exception) -> Fit:
    row.update({"status": "fit_failed",
                "error": redact_secrets(f"{type(error).__name__}: {error}")[:300]})
    return Fit(row)


def _describe(model: str, fit: dict, row: dict) -> Fit:
    stats = fit["stats"]
    # Volatility rank, 1 = lowest. A state with no training rows has NaN
    # volatility: ranked lowest and visible as 0 in obs_per_regime.
    order = stats["volatility"].rank(method="first", na_option="top").astype(int)
    high = int(stats["volatility"].idxmax())
    probabilities = fit["probabilities"]
    final = probabilities.iloc[-1].to_numpy()
    row.update({
        "status": _status(model, fit), "high_state": high,
        "final_regime": int(order[int(np.argmax(final))]),  # volatility rank of the argmax state
        "p_high_final": float(final[high]),
        "p_state_final": ";".join(f"{x:.6f}" for x in final[order.sort_values().index.to_numpy()]),
        "obs_per_regime": ";".join(str(int(stats.loc[s, "observations"]))
                                   for s in order.sort_values().index),
        "aic": float(fit["aic"]) if "aic" in fit else None,
        "bic": float(fit["bic"]) if "bic" in fit else None,
    })
    if model == "HMM":
        quality = fit["fit_quality"]
        scores = fit["initialization_scores"]
        row.update({"monitor_converged": quality["monitor_converged"],
                    "iterations": quality["iterations"], "hit_max_iter": quality["hit_max_iter"],
                    "loglik_decreases": quality["loglik_decreases"],
                    "seeds_not_converged": sum(not q["converged"] for q in fit["seed_quality"]),
                    "train_loglik": float(fit["train_loglik"]),
                    "init_loglik_spread": float(max(scores) - min(scores)),
                    "init_min_agreement": float(min(fit["assignment_agreement"]))})
    else:
        row.update({"monitor_converged": fit["converged"],
                    "train_loglik": float(fit["result"].llf)})
    p_high = probabilities[f"state_{high}"].rename(None)
    return Fit(row, p_high, _params(model, fit, high))


def _max_abs(a, b) -> float:
    if a is None or b is None or len(a) != len(b):
        return float("nan")
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))) if len(a) else 0.0


def reproducibility_row(fits: list[Fit]) -> dict:
    """Identical repeats (same input, configuration and seed) against the first."""
    base = fits[0]
    statuses = {f.row["status"] for f in fits}
    out = {k: base.row[k] for k in ("asset", "model", "cut", "fold", "test_date", "seed")}
    out.update({"repeats": len(fits), "same_status": len(statuses) == 1,
                "same_final_regime": len({f.row["final_regime"] for f in fits}) == 1})
    if base.row["status"] == "fit_failed":
        out.update({"max_abs_probability_diff": np.nan, "max_abs_parameter_diff": np.nan,
                    "max_rel_loglik_diff": np.nan,
                    "reproducible": statuses == {"fit_failed"}})
        return out
    prob = max(_max_abs(base.p_high.to_numpy(), f.p_high.to_numpy()) if f.p_high is not None
               else np.inf for f in fits[1:]) if len(fits) > 1 else 0.0
    par = max(_max_abs(base.params, f.params) for f in fits[1:]) if len(fits) > 1 else 0.0
    ll = max(abs((f.row["train_loglik"] or np.nan) - base.row["train_loglik"]) /
             max(abs(base.row["train_loglik"]), 1.0) for f in fits[1:]) if len(fits) > 1 else 0.0
    out.update({"max_abs_probability_diff": prob, "max_abs_parameter_diff": par,
                "max_rel_loglik_diff": ll,
                "reproducible": bool(out["same_status"] and out["same_final_regime"]
                                     and prob <= TOLERANCES["probability_atol"]
                                     and (np.isnan(par) or par <= TOLERANCES["parameter_atol"])
                                     and ll <= TOLERANCES["loglik_rtol"])})
    return out


def agreement(a: pd.Series | None, b: pd.Series | None) -> tuple[float, float, int]:
    """High-volatility regime agreement (p_high >= 0.5) and mean |dp| on common dates."""
    if a is None or b is None:
        return np.nan, np.nan, 0
    common = a.index.intersection(b.index)
    if common.empty:
        return np.nan, np.nan, 0
    x, y = a.loc[common], b.loc[common]
    return float(((x >= 0.5) == (y >= 0.5)).mean()), float((x - y).abs().mean()), len(common)


def seed_sensitivity_row(base: Fit, alternative: Fit) -> dict:
    match, diff, n = agreement(base.p_high, alternative.p_high)
    return {**{k: base.row[k] for k in ("asset", "model", "cut", "fold", "test_date")},
            "seed": base.row["seed"], "alt_seed": alternative.row["seed"],
            "alt_status": alternative.row["status"], "regime_agreement": match,
            "mean_abs_p_high_diff": diff, "dates": n}


def fold_stability_rows(fits: list[Fit]) -> list[dict]:
    """Consecutive walk-forward folds: do they describe the same past dates alike?

    Compared on dates both folds filtered, i.e. up to the earlier fold's test row.
    """
    ordered = sorted((f for f in fits if f.row["cut"] == "walk_forward"), key=lambda f: f.row["fold"])
    rows = []
    for earlier, later in zip(ordered, ordered[1:]):
        match, diff, n = agreement(earlier.p_high, later.p_high)
        rows.append({"asset": earlier.row["asset"], "model": earlier.row["model"],
                     "fold": earlier.row["fold"], "next_fold": later.row["fold"],
                     "statuses": f"{earlier.row['status']}/{later.row['status']}",
                     "regime_agreement": match, "mean_abs_p_high_diff": diff, "dates": n})
    return rows


def revalidate_asset(asset: str, bars: pd.DataFrame, cfg: RevalidationConfig) -> dict[str, list[dict]]:
    """All fits for one asset. ``bars`` are complete 1Day OHLCV bars (timestamp column)."""
    from src.features.statistical_features import calculate_statistical_features
    features = calculate_statistical_features(bars.reset_index(drop=True))
    out = {"fits": [], "reproducibility": [], "seed_sensitivity": [], "fold_stability": []}
    for n_states in cfg.states:
        for model in MODELS:
            bases = []
            for cut in cut_plan(features.index, cfg):
                repeats = [fit_once(asset, model, n_states, features, cut, cfg.random_state,
                                    f"repeat_{r}") for r in range(cfg.repeats)]
                alternative = fit_once(asset, model, n_states, features, cut,
                                       cfg.random_state + cfg.alt_seed_offset, "alt_seed")
                out["fits"] += [f.row for f in repeats + [alternative]]
                out["reproducibility"].append(reproducibility_row(repeats))
                if repeats[0].row["status"] != "fit_failed":
                    out["seed_sensitivity"].append(seed_sensitivity_row(repeats[0], alternative))
                bases.append(repeats[0])
            out["fold_stability"] += fold_stability_rows(bases)
    return out


def convergence_summary(fits: pd.DataFrame) -> pd.DataFrame:
    """Per model: base-seed windows by status, with the assets and folds affected."""
    base = fits.loc[fits.run.eq("repeat_0")]
    rows = []
    for (model, cut), group in base.groupby(["model", "cut"]):
        counts = group.status.value_counts()
        bad = group.loc[group.status.ne("converged")]
        rows.append({"model": model, "cut": cut, "windows": len(group),
                     **{s: int(counts.get(s, 0)) for s in STATUSES},
                     "not_converged_pct": round(100*counts.get("not_converged", 0)/len(group), 1),
                     "fit_failed_pct": round(100*counts.get("fit_failed", 0)/len(group), 1),
                     "assets_affected": ";".join(sorted(bad.asset.unique())),
                     "folds_affected": ";".join(f"{a}:{f}" for a, f in zip(bad.asset, bad.fold))})
    return pd.DataFrame(rows)


def compare_legacy(legacy: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Phase 4.1 ``regimes.csv`` (one last cut per asset) next to the strict refit."""
    legacy = legacy.assign(label=LEGACY_LABEL).rename(
        columns={"converged": "legacy_converged", "stable": "legacy_stable"})
    new = fits.loc[fits.cut.eq("legacy_last_cut") & fits.run.eq("repeat_0"),
                   ["asset", "model", "status", "monitor_converged", "init_min_agreement"]]
    return legacy.merge(new, on=["asset", "model"], how="left")


def config_dict(cfg: RevalidationConfig) -> dict:
    return {**asdict(cfg), "states": list(cfg.states), "tolerances": TOLERANCES}
