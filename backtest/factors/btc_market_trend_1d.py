from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_btc_close() -> pd.Series:
    cache_path = Path(
        os.environ.get(
            'SELECT_COIN_BTC_MARKET_STATE_PATH',
            _project_root() / 'data' / 'cache' / 'btc_market_state.pkl',
        )
    )

    if cache_path.exists():
        obj = pd.read_pickle(cache_path)
        if isinstance(obj, pd.DataFrame):
            if 'close' in obj.columns:
                close = obj['close']
            elif 'BTC-USDT' in obj.columns:
                close = obj['BTC-USDT']
            else:
                close = obj.iloc[:, 0]
        else:
            close = obj
    else:
        import config

        pivot = pd.read_pickle(config.raw_data_path / 'market_pivot_swap.pkl')
        close = pivot['close']['BTC-USDT']

    close = pd.to_numeric(close, errors='coerce')
    close.index = pd.DatetimeIndex(close.index)
    return close.sort_index().replace([np.inf, -np.inf], np.nan).ffill()


def _periods_for_days(times: pd.Series | pd.Index, n: int) -> int:
    idx = pd.DatetimeIndex(times)
    if len(idx) < 3:
        return max(int(n), 1)
    diffs = idx.to_series().diff().dropna().dt.total_seconds() / 3600.0
    med_hours = float(diffs.median()) if not diffs.empty else 24.0
    if not np.isfinite(med_hours) or med_hours <= 0:
        med_hours = 24.0
    bars_per_day = max(1, int(round(24.0 / med_hours)))
    return max(int(n) * bars_per_day, 1)


def _calc(df: pd.DataFrame, n: int) -> pd.Series:
    n = max(int(n), 1)
    times = pd.DatetimeIndex(df['candle_begin_time'])
    btc_close = _load_btc_close().reindex(times, method='ffill')
    periods = _periods_for_days(times, n)
    out = btc_close.pct_change(periods)
    return pd.Series(out.to_numpy(), index=df.index).replace([np.inf, -np.inf], np.nan)


def signal(df, n, factor_name):
    df[factor_name] = _calc(df, n)
    return df


def signal_multi_params(df, param_list) -> dict:
    return {str(int(param)): _calc(df, int(param)) for param in param_list}
