from __future__ import annotations

import pandas as pd


def _calc(df: pd.DataFrame, n: int) -> pd.Series:
    return df["quote_volume"].rolling(int(n), min_periods=1).mean()


def signal(df, n, factor_name):
    df[factor_name] = _calc(df, int(n))
    return df


def signal_multi_params(df, param_list) -> dict:
    return {str(int(param)): _calc(df, int(param)) for param in param_list}
