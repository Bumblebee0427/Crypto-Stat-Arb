from __future__ import annotations

import numpy as np
import pandas as pd

extra_data_dict = {
    "small_qty_misalignment_rankcorr_lag5_1d": ["small_qty_misalignment_rankcorr_lag5_1d"]
}


def _calc(df: pd.DataFrame, n: int) -> pd.Series:
    n = max(int(n), 1)
    raw = pd.to_numeric(df["small_qty_misalignment_rankcorr_lag5_1d"], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)

    out = raw.copy()
    if n > 1:
        daily_value = raw.dropna()
        out.loc[daily_value.index] = daily_value.rolling(n, min_periods=1).mean()

    return out.ffill()


def signal(df, n, factor_name):
    df[factor_name] = _calc(df, n)
    return df


def signal_multi_params(df, param_list) -> dict:
    return {str(int(param)): _calc(df, int(param)) for param in param_list}
