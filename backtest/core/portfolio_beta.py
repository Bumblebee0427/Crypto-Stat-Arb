from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    window: int = 60,
    min_periods: int | None = None,
    name: str = 'portfolio_beta',
) -> pd.Series:
    if window < 2:
        raise ValueError('window must be at least 2')
    required = window if min_periods is None else int(min_periods)
    if required < 2 or required > window:
        raise ValueError('min_periods must be between 2 and window')

    aligned = pd.concat(
        [
            pd.to_numeric(asset_returns, errors='coerce').rename('asset'),
            pd.to_numeric(benchmark_returns, errors='coerce').rename('benchmark'),
        ],
        axis=1,
    )
    covariance = aligned['asset'].rolling(window, min_periods=required).cov(aligned['benchmark'])
    benchmark_variance = aligned['benchmark'].rolling(window, min_periods=required).var()
    beta = covariance / benchmark_variance.where(benchmark_variance.abs() > np.finfo(float).eps)
    return beta.rename(name)
