from __future__ import annotations

import numpy as np
import pandas as pd

extra_data_dict = {
    "big_ultra_harmonic_rm240_mean_1d": ["big_ultra_harmonic_rm240_mean_1d"]
}


def _linear_decay(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if not mask.any():
        return np.nan

    weights = np.arange(1, len(values) + 1, dtype=float)
    values = values[mask]
    weights = weights[mask]
    return float(np.dot(values, weights) / weights.sum())


def _calc(df: pd.DataFrame, n: int) -> pd.Series:
    n = max(int(n), 1)
    raw = pd.to_numeric(df["big_ultra_harmonic_rm240_mean_1d"], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)

    out = raw.copy()
    if n > 1:
        daily_value = raw.dropna()
        out.loc[daily_value.index] = daily_value.rolling(n, min_periods=1).apply(
            _linear_decay,
            raw=True,
        )

    return out.ffill()


def signal(df, n, factor_name):
    df[factor_name] = _calc(df, n)
    return df


def signal_multi_params(df, param_list) -> dict:
    return {str(int(param)): _calc(df, int(param)) for param in param_list}
