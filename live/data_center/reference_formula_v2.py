"""Corrected production daily formula; source certification is independent.

Evidence: saved Tardis events and pv/qty minute panels, including ETH 2026-03-01.
This is NOT source certification: complete public trades can differ from the
historical capture. Never pass aggTrades as individual trades, discard public
events to imitate capture gaps, or issue an actionable receipt from this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_center.base_strategy import MAIN, SMALL, FORMULA_VERSION
EPS = 1e-7


def aggregate_raw_trades(trades):
    """Individual trades ONLY: id, timestamp_ms, price, qty, side BUY/SELL.

    Consecutive observed IDs are necessary, not sufficient for day completeness;
    callers must independently establish the head/tail and exchange watermark.
    """
    d = trades[['id', 'timestamp_ms', 'price', 'qty', 'side']].copy()
    if d.empty or not d.side.isin(['BUY', 'SELL']).all():
        raise ValueError('empty input or invalid side')
    numeric = d[['id', 'timestamp_ms', 'price', 'qty']].to_numpy()
    if not np.isfinite(numeric).all() or (d[['price', 'qty']] <= 0).any().any():
        raise ValueError('invalid trade values')
    if ((d[['id', 'timestamp_ms']] % 1) != 0).any().any():
        raise ValueError('noninteger trade ID or timestamp')
    duplicate = d[d.duplicated('id', keep=False)]
    if not duplicate.empty and (duplicate.groupby('id').nunique() > 1).any().any():
        raise ValueError('conflicting duplicate trade ID')
    d = d.drop_duplicates('id').sort_values('id', kind='stable')
    if (np.diff(d.id.to_numpy(dtype=np.int64)) != 1).any():
        raise ValueError('individual trade ID gap')
    d['time'] = pd.to_datetime(d.timestamp_ms, unit='ms', utc=True)
    d['side'] = d.side.str.lower()
    d['notional'] = d.price * d.qty
    return d.groupby(['time', 'side'], sort=True).agg(qty=('qty', 'sum'), notional=('notional', 'sum')).reset_index()


def nearest_half_up(values, quantile):
    """Polars nearest semantics: choose floor((n-1)*q+0.5), NOT banker's rounding."""
    if not 0 <= quantile <= 1 or len(values) == 0:
        raise ValueError('invalid quantile input')
    return float(np.sort(values)[int(np.floor((len(values) - 1) * quantile + .5))])


def minute_features_from_events(events):
    """One UTC source partition; hourly quantiles, closed-both minute windows.

    Exact-minute events belong to both adjacent existing windows. Windows start
    at the first event's floored minute; do not invent a previous window before
    that. A late prior-day event can create a 23:59 row. Sparse no-event minutes
    stay absent, as in the saved engine output (not zero-filled).
    """
    e = events[['time', 'side', 'qty', 'notional']].copy()
    e['time'] = pd.to_datetime(e.time, utc=True)
    if e.empty or e.time.isna().any() or not e.side.isin(['buy', 'sell']).all():
        raise ValueError('invalid source events')
    # Tardis partitions by local receipt day; a delayed exchange event may be
    # just before midnight. Preserve that event, including its hour quantile.
    day = e.time.max().normalize()
    if e.time.min() < day - pd.Timedelta(minutes=1) or e.duplicated(['time', 'side']).any():
        raise ValueError('expected daily partition of unique timestamp/side events')
    if not np.isfinite(e[['qty', 'notional']].to_numpy()).all() or (e[['qty', 'notional']] <= 0).any().any():
        raise ValueError('invalid event values')
    hour = e.time.dt.floor('h')
    q50, q95, q995 = [e.notional.groupby(hour).transform(lambda x: nearest_half_up(x, q))
                      for q in (.5, .95, .995)]
    e['small'] = e.notional.le(q50)
    e['big'] = e.notional.gt(q95) & e.notional.le(q995)
    e['ultra'] = e.notional.gt(q995)
    e['minute'] = e.time.dt.floor('min')
    first_minute = e.minute.min()
    boundary = e.loc[e.time.eq(e.minute)].copy()
    boundary['minute'] -= pd.Timedelta(minutes=1)
    boundary = boundary.loc[boundary.minute >= first_minute]
    e = pd.concat([e, boundary], ignore_index=True).sort_values('time', kind='stable')
    index = pd.DatetimeIndex(sorted(e.minute.unique()))
    out = pd.DataFrame(index=index)
    for name in ('big', 'ultra'):
        bucket = e.loc[e[name]]
        buy = bucket.loc[bucket.side.eq('buy')].groupby('minute').notional.sum().reindex(index, fill_value=0.)
        sell = bucket.loc[bucket.side.eq('sell')].groupby('minute').notional.sum().reindex(index, fill_value=0.)
        total = bucket.groupby('minute').notional.sum().reindex(index, fill_value=0.)
        out[name] = (buy - sell) / (total + EPS)
    # Keep separate compensated buy/sell reductions, then add as the original
    # daily input did. Summing all sides together changes float ties and ranks.
    bucket = e.loc[e.small]
    buy = bucket.loc[bucket.side.eq('buy')].groupby('minute').qty.sum().reindex(index, fill_value=0.)
    sell = bucket.loc[bucket.side.eq('sell')].groupby('minute').qty.sum().reindex(index, fill_value=0.)
    out['small_qty'] = buy + sell
    return out


def daily_factors(minutes, day):
    """Compute rolling observations BEFORE selecting the output day's rows.

    Accept sparse saved minute partitions; no source-completeness assertion is
    made here. Only the one preceding boundary minute is allowed outside day.
    """
    m = minutes[['big', 'ultra', 'small_qty']].copy()
    index = pd.DatetimeIndex(pd.to_datetime(m.index, utc=True))
    start = pd.Timestamp(day, tz='UTC')
    end = start + pd.Timedelta(days=1)
    if (m.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or not index.equals(index.floor('min'))
            or index.min() < start - pd.Timedelta(minutes=1) or index.max() >= end):
        raise ValueError('invalid daily minute partition')
    if not np.isfinite(m.to_numpy()).all():
        raise ValueError('nonfinite minute inputs')
    if (m.small_qty < 0).any() or (m[['big', 'ultra']].abs() > 1 + 1e-12).any().any():
        raise ValueError('invalid minute values')
    m.index = index
    current = index >= start
    if not current.any():
        raise ValueError('no minute in requested day')
    harmonic = np.sign(m.big + m.ultra) * 2 * m.big.abs() * m.ultra.abs() / (m.big.abs() + m.ultra.abs() + EPS)
    rolling = harmonic.rolling(240, min_periods=240).mean()
    ranks = m.small_qty.rolling(240, min_periods=240).rank(method='average', pct=True)
    lag = m.small_qty.shift(5).rolling(240, min_periods=240).rank(method='average', pct=True)
    main, small = rolling.loc[current].mean(), -ranks.loc[current].corr(lag.loc[current])
    return {MAIN: float(main) if np.isfinite(main) else None,
            SMALL: float(small) if np.isfinite(small) else None,
            'formula_version': FORMULA_VERSION, 'minute_count': int(current.sum()),
            'boundary_minute_count': int((~current).sum()), 'source_certified': False}
