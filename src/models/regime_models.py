"""Retrospective change points and causally filtered, unlabeled regimes."""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import logging
import warnings

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from .model_status import model_status


REGIME_INPUTS = ["log_return", "realized_volatility_20", "rolling_slope_20"]


def _regime_status(family: str, n_states: int) -> dict:
    """Three-state fits still run, but are flagged experimental (Phase 4.1)."""
    status = model_status(f"{family}{n_states}")
    if status.experimental:
        warnings.warn(f"{family.upper()}{n_states}: experimental, {status.warning}.", stacklevel=3)
    return status.as_dict()


def detect_change_points(series: pd.Series, penalty: float = 3.0,
                         min_size: int = 20) -> list[pd.Timestamp]:
    """PELT mean-shift dates, retrospective over the supplied series only."""
    clean = series.dropna()
    if len(clean) < 2 * min_size:
        return []
    values = clean.to_numpy(dtype=float)
    scale = np.std(values)
    if scale <= 0:
        return []
    standardized = ((values - values.mean()) / scale).reshape(-1, 1)
    endpoints = rpt.Pelt(model="l2", min_size=min_size).fit(standardized).predict(pen=penalty)
    return [clean.index[position - 1] for position in endpoints if position < len(clean)]


def _descriptions(stats: pd.DataFrame) -> pd.DataFrame:
    """Name states from fitted statistics; do not attach market labels."""
    stats = stats.copy()
    tolerance = max(stats["mean_return"].std(ddof=0) * 0.25, 1e-5)
    stats["trend_description"] = np.select(
        [stats["mean_return"] > tolerance, stats["mean_return"] < -tolerance],
        ["positiva", "negativa"], default="neutral")
    order = stats["volatility"].rank(method="first")
    if len(stats) == 2:
        labels = {1: "baja", 2: "alta"}
    else:
        labels = {1: "baja", 2: "media", 3: "alta"}
    stats["volatility_description"] = order.map(labels)
    return stats


def _state_stats(states: pd.Series, returns: pd.Series, volatility: pd.Series,
                 transition: np.ndarray, n_states: int,
                 slope: pd.Series | None = None) -> pd.DataFrame:
    records = []
    for state in range(n_states):
        mask = states.eq(state)
        persistence = float(transition[state, state])
        records.append({"state": state, "mean_return": returns.reindex(states.index)[mask].mean(),
                        "volatility": volatility.reindex(states.index)[mask].mean(),
                        "mean_slope": slope.reindex(states.index)[mask].mean() if slope is not None else np.nan,
                        "avg_duration": 1 / max(1 - persistence, 1e-8),
                        "observations": int(mask.sum())})
    return _descriptions(pd.DataFrame(records).set_index("state"))


def fit_markov_regimes(train_features: pd.DataFrame, all_features: pd.DataFrame,
                       n_states: int = 2, search_reps: int = 5,
                       random_state: int = 42) -> dict:
    """Markov switching on returns; frozen train parameters and filtered posteriors."""
    if n_states not in (2, 3):
        raise ValueError("Solo 2 o 3 regímenes.")
    status = _regime_status("markov", n_states)
    train = (train_features["log_return"].dropna() * 100)
    full = (all_features["log_return"].dropna() * 100)
    model = MarkovRegression(train, k_regimes=n_states, trend="c", switching_variance=True)
    previous_rng = np.random.get_state()
    try:
        np.random.seed(random_state)
        # statsmodels >= 0.15 draws search starts from its own ``rng``; without it
        # the global seed is ignored and fits differ run to run.
        rng = {"rng": random_state} if "rng" in inspect.signature(model.fit).parameters else {}
        result = model.fit(em_iter=20, search_reps=search_reps, search_iter=20, disp=False, **rng)
    finally:
        np.random.set_state(previous_rng)
    if not result.mle_retvals.get("converged", False):
        warnings.warn(f"Markov {n_states}: no convergió.", stacklevel=2)
    filtered = MarkovRegression(full, k_regimes=n_states, trend="c",
                                switching_variance=True).filter(result.params)
    probabilities = filtered.filtered_marginal_probabilities
    probabilities.columns = [f"state_{i}" for i in range(n_states)]
    states = probabilities.to_numpy().argmax(axis=1)
    train_states = pd.Series(states[:len(train)], index=train.index)
    transition = np.asarray(result.regime_transition[:, :, 0]).T
    stats = _state_stats(train_states, all_features["log_return"],
                         all_features["realized_volatility_20"], transition, n_states,
                         all_features["rolling_slope_20"])
    return {"result": result, "probabilities": probabilities,
            "states": pd.Series(states, index=full.index), "stats": stats,
            "aic": result.aic, "bic": result.bic,
            "converged": bool(result.mle_retvals.get("converged", False)),
            "model_status": status}


def _hmm_forward(model, x: np.ndarray) -> np.ndarray:
    """Forward-only Gaussian HMM filter, avoiding predict_proba smoothing."""
    emissions = np.column_stack([
        multivariate_normal.logpdf(x, mean=model.means_[state],
                                   cov=model.covars_[state], allow_singular=True)
        for state in range(model.n_components)])
    probabilities = np.empty_like(emissions)
    log_transition = np.log(np.clip(model.transmat_, 1e-300, None))
    previous = np.log(np.clip(model.startprob_, 1e-300, None))
    for t in range(len(x)):
        prior = previous if t == 0 else logsumexp(previous[:, None] + log_transition, axis=0)
        current = emissions[t] + prior
        current -= logsumexp(current)
        probabilities[t] = np.exp(current)
        previous = current
    return probabilities


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def _hmmlearn_messages():
    """Collect hmmlearn's ``Model is not converging`` log lines (EM log-likelihood
    decreased) instead of letting them reach stderr unattributed."""
    logger = logging.getLogger("hmmlearn")
    handler, propagate = _Collect(), logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate


def _em_quality(model, decreases: int) -> dict:
    """hmmlearn's ``monitor_.converged`` is also True when EM stops at n_iter or
    when the log-likelihood went *down*; report those cases explicitly."""
    monitor = model.monitor_
    hit_max_iter = monitor.iter >= monitor.n_iter
    return {"monitor_converged": bool(monitor.converged), "iterations": int(monitor.iter),
            "hit_max_iter": bool(hit_max_iter), "loglik_decreases": int(decreases),
            "converged": bool(monitor.converged) and not hit_max_iter and decreases == 0}


def fit_hmm_regimes(train_features: pd.DataFrame, all_features: pd.DataFrame,
                    n_states: int = 2, random_state: int = 42) -> dict:
    """Gaussian HMM trained on standardized train features; filtered posteriors."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as error:
        raise RuntimeError("hmmlearn no está instalado para este intérprete") from error
    if n_states not in (2, 3):
        raise ValueError("Solo 2 o 3 regímenes.")
    status = _regime_status("hmm", n_states)
    train = train_features[REGIME_INPUTS].dropna()
    full = all_features[REGIME_INPUTS].dropna()
    scaler = StandardScaler().fit(train)
    x_train, x_full = scaler.transform(train), scaler.transform(full)
    candidates = []
    for seed in (random_state, random_state + 1, random_state + 2):
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200,
                            tol=1e-4, random_state=seed, min_covar=1e-4)
        # hmmlearn initialises means with sklearn KMeans, whose OpenMP reductions
        # are not bit-reproducible across runs with several threads: identical
        # training data could give parameters differing in the last bits. One
        # OpenMP thread makes the fit deterministic (same algorithm and seed).
        with _hmmlearn_messages() as messages, threadpool_limits(limits=1, user_api="openmp"):
            model.fit(x_train)
        quality = _em_quality(model, sum("not converging" in m for m in messages))
        if not quality["monitor_converged"]:
            warnings.warn(f"HMM {n_states} seed {seed}: no convergió.", stacklevel=2)
        elif not quality["converged"]:
            warnings.warn(f"HMM {n_states} seed {seed}: EM sin convergencia limpia "
                          f"(log-likelihood decreció {quality['loglik_decreases']} veces, "
                          f"iteraciones {quality['iterations']}/{model.n_iter}).", stacklevel=2)
        candidates.append((model.score(x_train), model,
                           _hmm_forward(model, x_train).argmax(axis=1), quality))
    scores = [score for score, *_ in candidates]
    if max(scores) - min(scores) > max(10.0, abs(max(scores)) * 0.05):
        warnings.warn(f"HMM {n_states}: log-likelihood varía entre inicializaciones: {scores}",
                      stacklevel=2)
    _, model, reference, quality = max(candidates, key=lambda pair: pair[0])
    agreements = []
    label_changes = []
    for _, _, labels, _ in candidates:
        confusion = np.array([[(reference == i).__and__(labels == j).sum()
                               for j in range(n_states)] for i in range(n_states)])
        row, col = linear_sum_assignment(-confusion)
        mapping = {j: i for i, j in zip(row, col)}
        agreements.append(float(np.mean([mapping[label] == ref for label, ref in zip(labels, reference)])))
        label_changes.append(any(mapping[i] != i for i in range(n_states)))
    probabilities = pd.DataFrame(_hmm_forward(model, x_full), index=full.index,
                                 columns=[f"state_{i}" for i in range(n_states)])
    states = pd.Series(probabilities.to_numpy().argmax(axis=1), index=full.index)
    stats = _state_stats(states.reindex(train.index), all_features["log_return"],
                         all_features["realized_volatility_20"], model.transmat_, n_states,
                         all_features["rolling_slope_20"])
    return {"model": model, "scaler": scaler, "probabilities": probabilities,
            "states": states, "stats": stats, "train_loglik": model.score(x_train),
            "initialization_scores": scores, "assignment_agreement": agreements,
            "label_permutations": label_changes,
            "converged": bool(model.monitor_.converged), "fit_quality": quality,
            "seed_quality": [q for *_, q in candidates], "model_status": status}


def regime_changes(states: pd.Series) -> pd.DatetimeIndex:
    """Dates when the maximum filtered posterior changes state."""
    return pd.DatetimeIndex(states.index[states.ne(states.shift(1))][1:])
