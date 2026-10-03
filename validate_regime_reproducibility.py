"""Phase 4.1 regime revalidation (HMM2 / Markov2) after the Phase 7 fixes.

Live mode (default) downloads ARM, NVDA, AMD, AVGO and QQQ 1Day bars with the
Alpaca credentials in ``.env`` (never printed) on the same common dates as
``validate_walk_forward.py``, ending before the current session. ``--synthetic``
runs the same pipeline offline on deterministic synthetic series; its numbers
describe the synthetic data only. ``--csv-dir`` reads ``<ASSET>.csv`` files
(timestamp, open, high, low, close, volume) instead of downloading.

For every asset, model and cut (8 expanding walk-forward folds plus the single
last cut Phase 4.1 used) it records convergence (strict for HMM), repeat
reproducibility, sensitivity to the seed and agreement between folds. Original
Phase 4.1 results in ``phase4_1_results/`` are read, never overwritten, and
reported as ``legacy_pre_revalidation``.

No model, parameter or selection rule is changed or tuned. No returns, PnL or
signals are computed.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from src.security import redact_secrets
from src.validation.regime_revalidation import (LEGACY_LABEL, TOLERANCES, RevalidationConfig,
                                                compare_legacy, config_dict, convergence_summary,
                                                revalidate_asset)

ASSETS = ("ARM", "NVDA", "AMD", "AVGO", "QQQ")
START = "2023-09-14"  # as validate_walk_forward.py
OUT = Path("phase4_1_revalidation")
LEGACY = Path("phase4_1_results") / "regimes.csv"


def synthetic_bars(n: int = 756, assets=ASSETS) -> dict[str, pd.DataFrame]:
    """Deterministic two-volatility-regime walks on business days."""
    stamps = pd.bdate_range("2023-09-14", periods=n).tz_localize("America/New_York").tz_convert("UTC")
    data = {}
    for k, name in enumerate(assets):
        rng = np.random.default_rng(500 + k)
        vol = np.where((np.arange(n)//90) % 2 == 0, 0.012, 0.03)
        close = 100*np.exp(np.cumsum(vol*rng.standard_normal(n)))
        opens = close*np.exp(0.2*vol*rng.standard_normal(n))
        spread = np.abs(vol*rng.standard_normal(n))*close
        data[f"SYN_{name}"] = pd.DataFrame({"timestamp": stamps, "open": opens,
            "high": np.maximum(opens, close)+spread, "low": np.minimum(opens, close)-spread,
            "close": close, "volume": rng.uniform(5e5, 2e6, n)})
    return data


def load_live(assets) -> tuple[dict[str, pd.DataFrame], dict]:
    from src.data.alpaca_client import data_feed, download_historical_bars
    end = date.today().isoformat()  # current session excluded
    data = {a: download_historical_bars(a, START, end, "1Day", "regular") for a in assets}
    return data, {"source": "alpaca", "feed": data_feed(), "start": START, "end_exclusive": end}


def load_csv(folder: Path, assets) -> tuple[dict[str, pd.DataFrame], dict]:
    data = {}
    for asset in assets:
        frame = pd.read_csv(folder / f"{asset}.csv")
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        data[asset] = frame
    return data, {"source": f"csv:{folder}"}


def common_dates(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Same common-date universe as validate_walk_forward.py."""
    common = sorted(set.intersection(*(set(f.timestamp) for f in data.values())))
    return {a: f.set_index("timestamp").loc[common].reset_index() for a, f in data.items()}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "hmmlearn", "threadpoolctl"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def table(frame: pd.DataFrame) -> list[str]:
    return ["```text", frame.to_string(index=False) if not frame.empty else "(vacío)", "```", ""]


def write_summary(out: Path, label: str, tables: dict[str, pd.DataFrame], meta: dict) -> None:
    fits, repro = tables["fits"], tables["reproducibility"]
    seeds, folds = tables["seed_sensitivity"], tables["fold_stability"]
    conv = convergence_summary(fits)
    conv.to_csv(out/"convergence_summary.csv", index=False)
    by_model = []
    for model, group in repro.groupby("model"):
        s = seeds.loc[seeds.model.eq(model)]
        f = folds.loc[folds.model.eq(model)]
        by_model.append({"model": model, "windows": len(group),
                         "reproducible": f"{int(group.reproducible.sum())}/{len(group)}",
                         "max_abs_p_diff": group.max_abs_probability_diff.max(),
                         "seed_agreement_median": s.regime_agreement.median(),
                         "seed_agreement_min": s.regime_agreement.min(),
                         "fold_agreement_median": f.regime_agreement.median(),
                         "fold_agreement_min": f.regime_agreement.min()})
    text = [f"# Fase 4.1: revalidación de regímenes — {label}", "",
            f"Generado {meta['generated_at']} (commit {meta['git_commit']}). Fuente: {meta['data']['source']}.",
            "", "Convergencia, reproducibilidad y estabilidad son propiedades distintas. "
            "`converged` no significa que el régimen sea útil para predecir. "
            "Los modelos de 3 estados siguen siendo experimentales.", "",
            "## Convergencia (semilla base)", "",
            "HMM: `converged` estricto (`fit_quality`: sin bajadas de log-verosimilitud del EM y sin "
            "agotar `n_iter`). Markov: `mle_retvals['converged']` de statsmodels. `fit_failed`: excepción.",
            "", *table(conv),
            "## Reproducibilidad, sensibilidad a semilla y acuerdo entre folds", "",
            f"Tolerancias: {json.dumps(TOLERANCES)}.", "", *table(pd.DataFrame(by_model)),
            "## Ventanas no reproducibles", "",
            *table(repro.loc[~repro.reproducible]),
            "## Último corte de Fase 4.1 (refit estricto)", "",
            *table(fits.loc[fits.cut.eq("legacy_last_cut") & fits.run.eq("repeat_0"),
                            ["asset", "model", "status", "monitor_converged", "loglik_decreases",
                             "iterations", "init_min_agreement", "final_regime", "p_high_final",
                             "obs_per_regime"]])]
    if "legacy" in tables:
        text += [f"## Resultados originales ({LEGACY_LABEL})", "",
                 "`phase4_1_results/regimes.csv` tal cual (no se modifica), junto al refit estricto "
                 "del mismo corte.", "", *table(tables["legacy"])]
    else:
        text += [f"## Resultados originales ({LEGACY_LABEL})", "",
                 f"No se encontró `{LEGACY}`; no hay comparación con la Fase 4.1 original.", ""]
    text += ["## Conclusión", "",
             "Este informe no cambia por sí mismo la decisión sobre HMM2/Markov2 en "
             "`docs/MODEL_DECISIONS.md`; se revisa con estos números."]
    (out/"summary.md").write_text("\n".join(text)+"\n", encoding="utf-8")


def run(data: dict[str, pd.DataFrame], cfg: RevalidationConfig, out: Path, label: str,
        data_meta: dict, legacy: Path | None = LEGACY) -> dict[str, pd.DataFrame]:
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    collected = {"fits": [], "reproducibility": [], "seed_sensitivity": [], "fold_stability": []}
    data = common_dates(data)
    for asset, bars in data.items():
        print(f"{asset}: {len(bars)} bars {bars.timestamp.iloc[0]} .. {bars.timestamp.iloc[-1]}", flush=True)
        result = revalidate_asset(asset, bars, cfg)
        for key in collected:
            collected[key] += result[key]
    tables = {k: pd.DataFrame(v) for k, v in collected.items()}
    tables["fits"].to_csv(out/"regime_revalidation.csv", index=False)
    tables["reproducibility"].to_csv(out/"regime_reproducibility.csv", index=False)
    tables["seed_sensitivity"].to_csv(out/"regime_seed_sensitivity.csv", index=False)
    tables["fold_stability"].to_csv(out/"regime_fold_stability.csv", index=False)
    if legacy is not None and legacy.exists():
        tables["legacy"] = compare_legacy(pd.read_csv(legacy), tables["fits"])
        tables["legacy"].to_csv(out/f"{LEGACY_LABEL}_regimes.csv", index=False)
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": label, "git_commit": git_commit(), "data": data_meta,
            "assets": {a: {"bars": len(b), "first": str(b.timestamp.iloc[0]),
                           "last": str(b.timestamp.iloc[-1])} for a, b in data.items()},
            "config": config_dict(cfg), "versions": versions(),
            "hmm_openmp_threads": 1, "seconds": round(time.time()-started, 1)}
    (out/"run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    write_summary(out, label, tables, meta)
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="offline synthetic run")
    parser.add_argument("--csv-dir", type=Path, help="read <ASSET>.csv instead of downloading")
    parser.add_argument("--assets", nargs="+", default=list(ASSETS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--states", nargs="+", type=int, default=[2],
                        help="2 (default); add 3 only to inspect experimental models")
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--synthetic-bars", type=int, default=756)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    cfg = RevalidationConfig(min_train=args.min_train, step=args.step, repeats=args.repeats,
                             states=tuple(args.states))
    try:
        if args.synthetic:
            data = synthetic_bars(args.synthetic_bars, args.assets)
            out, label, meta, legacy = args.out or OUT/"synthetic", "SINTÉTICO (no describe mercados)", \
                {"source": "synthetic"}, None
        elif args.csv_dir:
            (data, meta), out, label, legacy = load_csv(args.csv_dir, args.assets), args.out or OUT, \
                "datos CSV locales", LEGACY
        else:
            (data, meta), out, label, legacy = load_live(args.assets), args.out or OUT, \
                "datos Alpaca", LEGACY
        tables = run(data, cfg, out, label, meta, legacy)
    except Exception:
        print("FAIL: " + redact_secrets(traceback.format_exc()))
        raise
    conv = convergence_summary(tables["fits"])
    print(conv.to_string(index=False))
    print(f"reproducible windows: {int(tables['reproducibility'].reproducible.sum())}/"
          f"{len(tables['reproducibility'])}")
    print(f"DONE: outputs in {out}")


if __name__ == "__main__":
    main()
