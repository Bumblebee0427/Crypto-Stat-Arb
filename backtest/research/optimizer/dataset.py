"""Frozen source loading; prediction access is restricted to completed rows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from .contracts import file_hash, write_json


DEFAULT_MANIFEST = Path(os.environ.get('SELECT_COIN_BACKTEST_MANIFEST', Path(__file__).resolve().parents[2] / 'data/source_manifest.json'))


@dataclass
class Dataset:
    times: pd.DatetimeIndex
    symbols: list[str]
    weights: np.ndarray  # LEVERAGED target notional/equity, calculated at close
    prices: dict[str, np.ndarray]
    steps: np.ndarray
    reference: pd.DataFrame
    initial_equity: float
    fee_rate: float
    min_order: float
    margin_rate: float
    panel: pd.DataFrame
    daily_returns: pd.DataFrame
    daily_closes: pd.DataFrame
    manifest: dict
    config: object

    def daily_history(self, cutoff, days):
        cutoff = pd.Timestamp(cutoff)
        return self.daily_returns.loc[(self.daily_returns.index <= cutoff) &
                                      (self.daily_returns.index > cutoff-pd.Timedelta(days=days))].copy()


def prepare(manifest_path: Path, cache: Path) -> Dataset:
    from core.select_coin import agg_multi_strategy_ratio
    from core.utils.functions import load_min_qty

    manifest = json.loads(manifest_path.read_text())
    if 'same-hour' not in manifest['factor_formula'] or manifest['volume_filter_hours'] != 600:
        raise ValueError('this adapter requires frozen v2/600h data; v3 needs a separate adapter')
    result = Path(manifest['result_folder'])
    run = result.parents[2]
    conf = pd.read_pickle(result/'config.pkl')
    if conf.is_day_period or str(conf.rebalance_mode) != '每个周期rebalance':
        raise ValueError('adapter only implements the audited hourly RebAlways policy')
    ref = pd.read_csv(result/'资金曲线.csv', parse_dates=['candle_begin_time'])
    times = pd.DatetimeIndex(ref.candle_begin_time)
    if times.tz is not None or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('expected unique sorted naive UTC hours')
    selection = pd.read_pickle(result/'选币结果.pkl')
    spot, swap = agg_multi_strategy_ratio(conf, selection.copy())
    if spot.to_numpy().any():
        raise ValueError('unexpected spot exposure')
    swap = swap.reindex(times, fill_value=0).sort_index(axis=1)
    symbols = list(swap.columns)
    weights = swap.to_numpy(dtype=np.float64)*conf.leverage
    cache.mkdir(parents=True, exist_ok=True)
    source_paths = [manifest_path, result/'config.pkl', result/'选币结果.pkl',
                    result/'资金曲线.csv', run/'data/min_qty/最小下单量_swap.csv']
    source_paths += [run/'data/cache'/name for name in [
        'all_factors_kline.pkl', 'factor_big_ultra_harmonic_rm240_mean_1d_lindecay_12.pkl',
        'factor_small_qty_misalignment_rankcorr_lag5_1d_8.pkl',
        'factor_small_qty_misalignment_rankcorr_lag5_1d_4.pkl']]
    signatures = {str(p): file_hash(p) for p in source_paths}
    market_path = Path(manifest['market_source'])/'market_pivot_swap.pkl'
    signatures[str(market_path)] = file_hash(market_path)
    marker = cache/'prepared.json'
    expected = {'sources': signatures, 'adapter_version': 1}
    usable = marker.exists() and json.loads(marker.read_text()).get('identity') == expected
    if not usable:
        print('prepare market matrices and cached factor features', flush=True)
        market = pd.read_pickle(market_path)
        for key in ['open', 'close', 'vwap1m', 'funding_rate']:
            frame = market[key].reindex(index=times, columns=symbols)
            np.save(cache/(key+'.npy'), frame.to_numpy(dtype=np.float64))
        close = market['close']
        daily = close.loc[close.index.hour == 23].copy()
        daily.index = daily.index+pd.Timedelta(hours=1)
        daily.to_parquet(cache/'daily_closes.parquet')
        del market
        panel = feature_panel(run, times, symbols, weights)
        execution = np.load(cache/'vwap1m.npy', mmap_mode='r')
        entry = {t: i for i, t in enumerate(times) if t.hour == 0}
        y, maturity = [], []
        for row in panel.itertuples():
            t = row.asof; j = row.column
            first, last = entry.get(t), entry.get(t+pd.Timedelta(days=1))
            valid = first is not None and last is not None
            if valid:
                a, b = execution[first, j], execution[last, j]
                valid = np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0
            y.append(float(b/a-1) if valid else np.nan)
            # Conservative: entire next execution hour has closed before label is usable.
            maturity.append(t+pd.Timedelta(days=1, hours=1))
        panel['future_return'] = y
        panel['label_available_at'] = maturity
        panel.to_parquet(cache/'panel.parquet', index=False)
        write_json(marker, {'identity': expected, 'hours': len(times), 'symbols': len(symbols),
                            'factor_cache': 'aligned original per-hour cached Series',
                            'missing_labels': int(panel.future_return.isna().sum())})
    daily = pd.read_parquet(cache/'daily_closes.parquet')
    prices = {k: np.load(cache/(k+'.npy'), mmap_mode='r')
              for k in ['open', 'close', 'vwap1m', 'funding_rate']}
    default_step, step_map = load_min_qty(run/'data/min_qty/最小下单量_swap.csv')
    steps = (0.1**pd.Series(step_map)).reindex(symbols, fill_value=0.1**default_step).to_numpy()
    panel = pd.read_parquet(cache/'panel.parquet')
    return Dataset(times, symbols, weights, prices, steps, ref, conf.initial_usdt,
                   conf.swap_c_rate, conf.swap_min_order_limit, conf.margin_rate,
                   panel, daily.pct_change(fill_method=None), daily,
                   {'source_manifest': manifest, 'source_sha256': signatures,
                    'point_in_time_assumptions': ['frozen revised historical factors, not online vintages',
                        'factor close available at UTC midnight; settlement publication lag is assumed',
                        'legacy mark/open/NaN/delist handling preserved for both policies'],
                    'real_orders': False}, conf)


def feature_panel(run, times, symbols, weights):
    root = run/'data/cache'
    candle = pd.read_pickle(root/'all_factors_kline.pkl')
    mask = candle.candle_begin_time.dt.hour.eq(23) & candle.is_spot.eq(0)
    selected = candle.loc[mask, ['candle_begin_time', 'symbol']].copy()
    selected['asof'] = selected.candle_begin_time+pd.Timedelta(hours=1)
    for column, file in [('main', 'big_ultra_harmonic_rm240_mean_1d_lindecay_12'),
                         ('small_long', 'small_qty_misalignment_rankcorr_lag5_1d_8'),
                         ('small_short', 'small_qty_misalignment_rankcorr_lag5_1d_4')]:
        series = pd.read_pickle(root/('factor_'+file+'.pkl'))
        # Original kline is time-sorted but retains the concat row IDs;
        # factor Series is in concat order. Join by IDs, never by row position.
        if (len(series) != len(candle) or not candle.index.is_unique
                or not selected.index.isin(series.index).all()):
            raise ValueError('factor cache index drift')
        selected[column] = series.loc[selected.index].to_numpy()
    del candle
    if selected.duplicated(['asof', 'symbol']).any():
        raise ValueError('duplicate factor observation at decision')
    keys = selected.set_index(['asof', 'symbol'])
    records = []
    for i, t in enumerate(times):
        if t.hour != 23:
            continue
        asof = t+pd.Timedelta(hours=1)
        for j in np.flatnonzero(weights[i]):
            key = (asof, symbols[j])
            if key not in keys.index:
                raise ValueError('selected symbol missing frozen features: '+str(key))
            row = keys.loc[key]
            sign = 1 if weights[i, j] > 0 else -1
            records.append({'asof': asof, 'symbol': symbols[j], 'column': j, 'side': sign,
                            'w0': weights[i, j], 'main': row['main'],
                            'small': row['small_long' if sign > 0 else 'small_short']})
    panel = pd.DataFrame(records)
    for feature in ['main', 'small']:
        panel[feature+'_rank'] = panel.groupby(['asof', 'side'])[feature].rank(pct=True)*2-1
    return panel
