"""Validate causal statistical features on over one year of ARM daily bars."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.alpaca_client import download_historical_bars
from src.features import calculate_statistical_features, run_statistical_diagnostics


def main() -> None:
    bars = download_historical_bars(
        "ARM", "2024-01-01", "2025-04-01", timeframe="1Day", session="regular")
    if bars.empty or len(bars) < 250:
        raise AssertionError(f"Se esperaba al menos un año de barras; recibidas: {len(bars)}")
    features = calculate_statistical_features(bars)
    assert len(features) == len(bars)
    assert features.index.equals(pd.DatetimeIndex(bars["timestamp"], name="timestamp"))
    assert len(features.columns) == 15

    close = bars["close"].to_numpy(dtype=float)
    expected_log = np.log(close[1:] / close[:-1])
    np.testing.assert_allclose(features["log_return"].iloc[1:], expected_log,
                               rtol=1e-12, atol=1e-12)
    assert pd.isna(features["log_return"].iloc[0])

    sample = 30
    window = close[sample - 19:sample + 1]
    manual_z = (close[sample] - window.mean()) / window.std(ddof=1)
    np.testing.assert_allclose(features["zscore_20"].iloc[sample], manual_z,
                               rtol=1e-11, atol=1e-11)
    assert features["zscore_20"].iloc[:19].isna().all()
    assert features["rolling_max_drawdown_60"].iloc[:59].isna().all()
    assert features["drawdown"].le(1e-12).all()
    assert features["rolling_max_drawdown_60"].dropna().le(1e-12).all()

    # Recalculate on prefixes. Every value already published at the cutoff
    # must remain identical when later observations are removed.
    for length in (20, 21, 60, 100, len(bars) - 1):
        prefix = calculate_statistical_features(bars.iloc[:length])
        pd.testing.assert_frame_equal(prefix, features.iloc[:length],
                                      check_exact=False, rtol=1e-10, atol=1e-10)

    print(f"ARM NORMALIZED 1Day regular: {len(bars)} filas, "
          f"{bars['timestamp'].iloc[0]} a {bars['timestamp'].iloc[-1]}")
    print("VALIDACIÓN OK: 15 features, índice, NaN iniciales, fórmulas y ausencia de look-ahead")
    print("\nÚltimas 10 filas:")
    print(features.tail(10).to_string())
    print("\nDiagnósticos estadísticos (muestra completa):")
    print(run_statistical_diagnostics(bars, features).to_string())
    print("\nAdvertencia: p-values descriptivos; dependen de los supuestos de cada prueba "
          "y no implican pronóstico ni causalidad.")


if __name__ == "__main__":
    main()
