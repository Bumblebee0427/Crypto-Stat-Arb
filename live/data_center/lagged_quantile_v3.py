"""No-lookahead hourly trade-size thresholds for the minute factor candidate.

Each UTC hour uses quantiles of the immediately preceding *complete* UTC
hour for the same symbol. The first hour therefore needs the previous day's
23:00 event window. This is a new factor definition and must not be mixed
with the existing same-hour production history.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from data_center.reference_formula_v2 import EPS, nearest_half_up
from data_center.reference_formula_v2 import daily_factors


FORMULA_VERSION = 'ref_notional_previous_utc_hour_closed_both_v3'
FORMULA_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
COMPONENTS = ('big_buy', 'big_sell', 'big_total', 'ultra_buy',
              'ultra_sell', 'ultra_total', 'small_buy', 'small_sell')


def quantiles(events: pd.DataFrame) -> tuple[float, float, float]:
    """Compute per-event notional thresholds with the saved nearest rule."""
    values = events['notional'].to_numpy(dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('previous UTC hour has no valid economic events')
    return tuple(nearest_half_up(values, q) for q in (.5, .95, .995))


def events_from_trades(raw: pd.DataFrame) -> pd.DataFrame:
    """Group individual IDs by exchange millisecond and aggressor side."""
    required = ('id', 'timestamp_ms', 'price', 'qty', 'maker')
    if raw.empty or any(name not in raw for name in required) or raw[list(required)].isna().any().any():
        raise ValueError('empty or null individual trade input')
    if not np.isfinite(raw[['id', 'timestamp_ms', 'price', 'qty']].to_numpy()).all():
        raise ValueError('nonfinite trade input')
    zero = raw.price.eq(0) & raw.qty.eq(0)
    if (((raw.price <= 0) | (raw.qty <= 0)) & ~zero).any():
        raise ValueError('invalid economic trade')
    duplicate = raw[raw.duplicated('id', keep=False)]
    if not duplicate.empty and (duplicate.groupby('id')[list(required[1:])].nunique() > 1).any().any():
        raise ValueError('conflicting duplicate individual ID')
    raw = raw.drop_duplicates('id').sort_values('id', kind='stable')
    raw = raw.loc[~(raw.price.eq(0) & raw.qty.eq(0))].copy()
    if raw.empty:
        return pd.DataFrame(columns=('time', 'side', 'qty', 'notional'))
    raw['time'] = pd.to_datetime(raw.timestamp_ms, unit='ms', utc=True)
    raw['side'] = np.where(raw.maker, 'sell', 'buy')
    raw['notional'] = raw.price * raw.qty
    return raw.groupby(['time', 'side'], sort=True).agg(
        qty=('qty', 'sum'), notional=('notional', 'sum')).reset_index()


def minute_components(events: pd.DataFrame, thresholds: tuple[float, float, float],
                      *, first_minute: pd.Timestamp | None = None) -> dict[str, dict[str, float]]:
    """Classify one hour of events using its prior-hour fixed thresholds."""
    if events.empty:
        return {}
    e = events[['time', 'side', 'qty', 'notional']].copy()
    e['time'] = pd.to_datetime(e.time, utc=True)
    if (e.time.isna().any() or e.duplicated(['time', 'side']).any()
            or not e.side.isin(('buy', 'sell')).all()
            or not np.isfinite(e[['qty', 'notional']].to_numpy()).all()
            or (e[['qty', 'notional']] <= 0).any().any()):
        raise ValueError('invalid economic events')
    q50, q95, q995 = thresholds
    if not (0 < q50 <= q95 <= q995):
        raise ValueError('invalid prior-hour thresholds')
    e['small'] = e.notional.le(q50)
    e['big'] = e.notional.gt(q95) & e.notional.le(q995)
    e['ultra'] = e.notional.gt(q995)
    e['minute'] = e.time.dt.floor('min')
    boundary = e.loc[e.time.eq(e.minute)].copy()
    boundary['minute'] -= pd.Timedelta(minutes=1)
    if first_minute is not None:
        boundary = boundary.loc[boundary.minute >= first_minute]
    e = pd.concat([e, boundary], ignore_index=True).sort_values('time', kind='stable')
    reductions = {}
    for name in ('big', 'ultra'):
        bucket = e.loc[e[name]]
        for side in ('buy', 'sell'):
            reductions[name+'_'+side] = bucket.loc[bucket.side.eq(side)].groupby('minute').notional.sum()
        reductions[name+'_total'] = bucket.groupby('minute').notional.sum()
    small = e.loc[e.small]
    for side in ('buy', 'sell'):
        reductions['small_'+side] = small.loc[small.side.eq(side)].groupby('minute').qty.sum()
    return {minute.isoformat(): {name: float(reductions[name].get(minute, 0.0))
                                  for name in COMPONENTS}
            for minute in sorted(e.minute.unique())}


def minute_frame(parts: dict[str, dict[str, float]], day: str) -> pd.DataFrame:
    """Turn sparse compensated minute reductions into rolling inputs."""
    start = pd.Timestamp(day, tz='UTC')
    end = start + pd.Timedelta(days=1)
    keys = sorted(k for k in parts if start <= pd.Timestamp(k) < end)
    if not keys:
        raise ValueError('no economic minute in UTC day')
    values = []
    for key in keys:
        row = parts[key]
        pair = []
        for name in ('big', 'ultra'):
            pair.append((row[name+'_buy']-row[name+'_sell']) /
                        (row[name+'_total']+EPS))
        values.append((pair[0], pair[1], row['small_buy']+row['small_sell']))
    return pd.DataFrame(values,
                        index=pd.to_datetime(keys, utc=True),
                        columns=('big', 'ultra', 'small_qty'), dtype=float)


def add_components(target: dict[str, dict[str, float]],
                   source: dict[str, dict[str, float]]) -> None:
    """Combine the two hour contributions at an exact hour boundary."""
    for minute, row in source.items():
        current = target.setdefault(minute, {name: 0.0 for name in COMPONENTS})
        for name in COMPONENTS:
            current[name] += row[name]


def daily_from_events(events: pd.DataFrame, day: str,
                      previous_hour_quantiles: tuple[float, float, float]) -> dict:
    """Calculate a whole UTC day without using its own hour's future trades."""
    if events.empty:
        raise ValueError('empty economic trade day')
    e = events.copy()
    e['time'] = pd.to_datetime(e.time, utc=True)
    start = pd.Timestamp(day, tz='UTC')
    end = start + pd.Timedelta(days=1)
    if (e.time < start).any() or (e.time >= end).any():
        raise ValueError('events are outside requested UTC day')
    first_minute = e.time.min().floor('min')
    hour = e.time.dt.hour
    current = previous_hour_quantiles
    parts = {}
    for h in range(24):
        block = e.loc[hour.eq(h)]
        if block.empty:
            raise ValueError(f'no economic trades in UTC hour {h:02d}')
        add_components(parts, minute_components(block, current, first_minute=first_minute))
        current = quantiles(block)
    factors = daily_factors(minute_frame(parts, day), day)
    factors['formula_version'] = FORMULA_VERSION
    factors['prior_day_23h_quantiles'] = list(previous_hour_quantiles)
    factors['next_day_00h_quantiles'] = list(current)
    return factors
