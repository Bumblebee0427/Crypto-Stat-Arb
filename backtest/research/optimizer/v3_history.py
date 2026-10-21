"""Historical individual-trade replay for the live previous-hour formula.

The vectorized reductions follow the live formula, including compensated pandas
sums and closed-both minute boundaries. Historical capture quality is reported;
rebuilding the formula does not certify exchange-wide completeness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

V3 = 'ref_notional_previous_utc_hour_closed_both_v3'
MAIN = 'big_ultra_harmonic_rm240_mean_1d'
SMALL = 'small_qty_misalignment_rankcorr_lag5_1d'


# Same stable/leveraged-token exclusions as the production default universe.
STABLE_BASES = {'BKRW','USDC','USDP','TUSD','BUSD','FDUSD','DAI','EUR','GBP',
                'USBP','SUSD','PAXG','AEUR','EURI'}


def historical_contract_allowed(symbol, metadata):
    # Current trading status MUST NOT filter past dates (survivorship bias).
    row = metadata.get(symbol, {})
    if row and (row.get('contractType') != 'PERPETUAL' or row.get('quoteAsset') != 'USDT'):
        return False
    base = row.get('baseAsset', symbol[:-4])
    if symbol in {'BTCUSDT','ETHUSDT'} or base in STABLE_BASES:
        return False
    return not (base.endswith(('UP','DOWN','BEAR','BULL')) and base not in {'JUP','SYRUP'})


def economic_events(raw):
    cols = ['id', 'timestamp', 'side', 'price', 'amount']
    if raw.empty or raw[cols].isna().any().any():
        raise ValueError('empty or null individual trades')
    duplicate = raw.loc[raw.duplicated('id', keep=False)]
    if len(duplicate) and (duplicate.groupby('id')[cols[1:]].nunique() > 1).any().any():
        raise ValueError('conflicting duplicate IDs')
    raw = raw.drop_duplicates('id').copy()
    if (not raw.side.isin(['buy', 'sell']).all()
            or not np.isfinite(raw[['price', 'amount']].to_numpy()).all()
            or (raw[['price', 'amount']] <= 0).any().any()
            or (raw.timestamp % 1000 != 0).any()):
        raise ValueError('invalid economic trades or exchange timestamp')
    raw['notional'] = raw.price * raw.amount
    events = raw.groupby(['timestamp', 'side'], sort=True).agg(
        qty=('amount', 'sum'), notional=('notional', 'sum')).reset_index()
    return events, raw


def daily_fast(events, day, previous_hour_quantiles=None):
    """Events must include the preceding closed hour and the requested UTC day."""
    start = pd.Timestamp(day).value // 1000
    stamp = events.timestamp.to_numpy(np.int64)
    hours = (stamp-start) // 3_600_000_000
    blocks = [events.loc[hours == h, 'notional'].to_numpy(float) for h in range(-1, 24)]
    if any(not len(x) for x in blocks[1:]) or (previous_hour_quantiles is None and not len(blocks[0])):
        raise ValueError('missing previous-hour seed or an empty UTC hour')
    thresholds = []
    for j, x in enumerate(blocks[:-1]):
        if j == 0 and previous_hour_quantiles is not None:
            thresholds.append(np.asarray(previous_hour_quantiles)); continue
        ordered = np.sort(x)
        thresholds.append(ordered[np.floor((len(x)-1)*np.array([.5, .95, .995])+.5).astype(int)])
    e = events.loc[(hours >= 0) & (hours < 24)].copy()
    h = hours[(hours >= 0) & (hours < 24)]
    q = np.asarray(thresholds)[h]
    notion, qty = e.notional.to_numpy(float), e.qty.to_numpy(float)
    buy = e.side.to_numpy() == 'buy'
    big = (notion > q[:, 1]) & (notion <= q[:, 2])
    ultra = notion > q[:, 2]
    small = notion <= q[:, 0]
    values = {}
    for name, bucket in [('big', big), ('ultra', ultra)]:
        for side, flag in [('buy', buy), ('sell', ~buy)]:
            values[name+'_'+side] = np.where(bucket & flag, notion, 0.)
        values[name+'_total'] = np.where(bucket, notion, 0.)
    values['small_buy'] = np.where(small & buy, qty, 0.)
    values['small_sell'] = np.where(small & ~buy, qty, 0.)
    reduced = pd.DataFrame(values)
    reduced['timestamp'] = e.timestamp.to_numpy(np.int64)
    reduced['minute'] = reduced.timestamp // 60_000_000
    first_minute = reduced.minute.min()
    boundary = reduced.loc[reduced.timestamp % 60_000_000 == 0].copy()
    boundary['minute'] -= 1
    boundary = boundary.loc[boundary.minute >= first_minute]
    reduced = pd.concat([reduced, boundary], ignore_index=True).sort_values('timestamp', kind='stable')
    m = reduced.groupby('minute')[list(values)].sum()
    big = (m.big_buy-m.big_sell)/(m.big_total+1e-7)
    ultra = (m.ultra_buy-m.ultra_sell)/(m.ultra_total+1e-7)
    small = m.small_buy+m.small_sell
    harmonic = np.sign(big+ultra)*2*big.abs()*ultra.abs()/(big.abs()+ultra.abs()+1e-7)
    a = harmonic.rolling(240, min_periods=240).mean().mean()
    rank = small.rolling(240, min_periods=240).rank(method='average', pct=True)
    lag = small.shift(5).rolling(240, min_periods=240).rank(method='average', pct=True)
    b = -rank.corr(lag)
    last = np.sort(blocks[-1])
    return {MAIN: float(a) if np.isfinite(a) else None,
            SMALL: float(b) if np.isfinite(b) else None,
            'minute_count': len(m), 'formula_version': V3,
            'next_day_00h_quantiles': last[np.floor((len(last)-1)*np.array([.5,.95,.995])+.5).astype(int)].tolist()}


def temporal_panel_check(panel):
    required = ['asof', 'symbol', 'side', 'formula_version', 'main_rank',
                'small_rank', 'future_return', 'label_available_at', 'entry_at', 'exit_at', 'feature_available_at']
    if any(c not in panel for c in required):
        raise ValueError('incomplete historical alpha panel')
    if panel.duplicated(['asof', 'symbol']).any() or not panel.formula_version.eq(V3).all():
        raise ValueError('duplicate or wrong-version alpha samples')
    if (not panel.side.isin([-1, 1]).all()
            or (panel.entry_at < panel['asof']).any()
            or panel.feature_available_at.isna().any()
            or (panel.feature_available_at > panel.entry_at).any()
            or (panel.exit_at <= panel.entry_at).any()
            or (panel.label_available_at < panel.exit_at+pd.Timedelta(minutes=1)).any()):
        raise ValueError('invalid historical label timing')
    if 'exit_window_end' in panel:
        if ((panel.execution_window_end <= panel.entry_at).any()
                or (panel.exit_window_end <= panel.exit_at).any()
                or (panel.label_available_at < panel.exit_window_end+pd.Timedelta(minutes=1)).any()):
            raise ValueError('execution-window label not yet mature')
    return panel


def rolling_features(history):
    """Match publisher.py's valid-observation handling, including new listings."""
    main = [float(v) for v in history[MAIN] if pd.notna(v) and np.isfinite(v)]
    small = [float(v) for v in history[SMALL] if pd.notna(v) and np.isfinite(v)]
    if not main or not small or (len(history) == 12 and (len(main) < 12 or len(small) < 8)):
        return None
    weights = range(1, len(main)+1)
    mean = lambda n: sum(small[-n:])/len(small[-n:])
    return {'main':sum(v*w for v,w in zip(main,weights))/sum(weights),
            'small_long':mean(8),'small_short':mean(4)}
