"""Relative strength aligns exact timestamps and never fills benchmark gaps."""
import numpy as np
import pandas as pd

from src.structure.relative_strength import relative_strength
from synthetic import bars_from_close


def test_missing_benchmark_dates_stay_missing():
    asset = bars_from_close(100+np.arange(80.))
    benchmark = bars_from_close(200+2*np.arange(80.))
    missing = [10, 11, 40]
    rs = relative_strength(asset, benchmark.drop(index=missing).reset_index(drop=True))
    assert rs.timestamp.equals(asset.timestamp)
    assert not rs.benchmark_available.iloc[missing].any()
    assert rs.benchmark_close.iloc[missing].isna().all()
    # Each period uses bars t and t-period; either side missing gives NaN.
    for period in (5, 20):
        column = f"relative_return_{period}"
        for m in missing:
            assert np.isnan(rs[column].iloc[m])
            if m+period < len(rs):
                assert np.isnan(rs[column].iloc[m+period])
    # Ratio z-score windows containing a missing bar are NaN, not filled.
    assert rs.ratio_zscore.iloc[40:60].isna().all()
    # Unaffected rows equal the complete-data computation.
    complete = relative_strength(asset, benchmark)
    unaffected = ~rs.relative_return_5.isna() & complete.relative_return_5.notna()
    np.testing.assert_allclose(rs.relative_return_5[unaffected], complete.relative_return_5[unaffected])


def test_extra_and_shifted_benchmark_rows_do_not_join():
    asset = bars_from_close(100+np.arange(30.))
    shifted = bars_from_close(200+np.arange(30.))
    shifted["timestamp"] = shifted.timestamp + pd.Timedelta(hours=1)
    rs = relative_strength(asset, shifted)
    assert len(rs) == len(asset) and rs.benchmark_close.isna().all()
