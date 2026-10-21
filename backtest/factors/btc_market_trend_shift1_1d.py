from __future__ import annotations

import numpy as np
import pandas as pd

from factors.btc_market_trend_1d import _load_btc_close, _periods_for_days


def _calc(df: pd.DataFrame, n: int) -> pd.Series:
    n = max(int(n), 1)
    times = pd.DatetimeIndex(df["candle_begin_time"])
    btc_close = _load_btc_close().reindex(times, method="ffill")
    periods = _periods_for_days(times, n)
    out = btc_close.shift(1).pct_change(periods)
    return pd.Series(out.to_numpy(), index=df.index).replace([np.inf, -np.inf], np.nan)


def signal(df, n, factor_name):
    df[factor_name] = _calc(df, n)
    return df


def signal_multi_params(df, param_list) -> dict:
    return {str(int(param)): _calc(df, int(param)) for param in param_list}
