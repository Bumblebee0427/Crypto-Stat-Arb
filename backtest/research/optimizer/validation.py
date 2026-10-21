from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_DEFINITIONS = {
    'frequency': 'complete UTC calendar days; net of all fees and funding',
    'annualization_days': 365,
    'minimum_acceptable_daily_return': 0.,
    'sortino': 'sqrt(365) * mean(r) / sqrt(mean(min(r, 0)^2)); denominator includes ALL days',
    'sharpe': 'sqrt(365) * mean(r) / sample_std(r, ddof=1); risk-free return assumed zero',
    'undefined_ratio': 'null when the denominator is zero',
}


def sortino_ratio(returns):
    r = np.asarray(returns, dtype=float)
    if not len(r) or not np.isfinite(r).all():
        return None
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.)**2)))
    return float(np.mean(r)/downside*np.sqrt(365)) if downside else None


def daily_returns(curve):
    x = curve.copy()
    x['date'] = pd.to_datetime(x.time).dt.floor('D')
    grouped = x.groupby('date')
    result = grouped.agg(equity=('equity', 'last'), before=('equity_before', 'first'),
        fee=('fee', 'sum'), funding_fee=('funding_fee', 'sum'), turnover=('turnover', 'sum'),
        rebalance_fee=('rebalance_fee', 'sum'), long_price_pnl=('long_price_pnl', 'sum'),
        short_price_pnl=('short_price_pnl', 'sum'), hours=('time', 'size'))
    result['net_return'] = result.equity/result.before-1
    result['fee_return'] = result.fee/result.before
    result['funding_return'] = result.funding_fee/result.before
    result['turnover_ratio'] = result.turnover/result.before
    return result.loc[result.hours == 24]


def summary(daily):
    r = daily.net_return
    if len(r) < 2:
        return {'days': len(r)}
    nav = (1+r).cumprod(); dd = nav/nav.cummax().clip(lower=1.)-1
    std = float(r.std())
    return {'days': len(r), 'start': str(r.index.min().date()), 'end': str(r.index.max().date()),
        'cumulative_return': float(nav.iloc[-1]-1),
        'annual_return': float(np.expm1(np.log1p(r).sum()*365/len(r))),
        'annual_volatility': float(std*np.sqrt(365)),
        'sharpe': float(r.mean()/std*np.sqrt(365)) if std else 0.,
        'sortino': sortino_ratio(r),
        'max_drawdown': float(dd.min()), 'worst_day': float(r.min()),
        'cvar_95': float(r[r <= r.quantile(.05)].mean()),
        'mean_daily_fee_bps': float(daily.fee_return.mean()*10000),
        'mean_daily_funding_bps': float(daily.funding_return.mean()*10000),
        'mean_daily_turnover': float(daily.turnover_ratio.mean()),
        'mean_daily_resize_fee_bps': float((daily.rebalance_fee/daily.before).mean()*10000)}


def paired_bootstrap(a, b, seed=20261019, block=7, draws=1000):
    both = pd.concat([a.net_return.rename('a'), b.net_return.rename('b')], axis=1).dropna()
    x, y = both.a.to_numpy(), both.b.to_numpy()
    if len(x) < block:
        return {'days': len(x)}
    rng = np.random.default_rng(seed); values = []
    def sharpe(r):
        return float(np.mean(r)/np.std(r, ddof=1)*np.sqrt(365))
    for _ in range(draws):
        starts = rng.integers(0, len(x)-block+1, int(np.ceil(len(x)/block)))
        idx = np.concatenate([np.arange(s, s+block) for s in starts])[:len(x)]
        sa, sb = sortino_ratio(x[idx]), sortino_ratio(y[idx])
        values.append([np.mean(x[idx]-y[idx])*10000, sharpe(x[idx])-sharpe(y[idx]),
                       sa-sb if sa is not None and sb is not None else np.nan])
    ci = np.nanquantile(values, [.025, .975], axis=0)
    return {'days': len(x), 'block_days': block, 'draws': draws,
        'mean_difference_bps': float(np.mean(x-y)*10000),
        'mean_difference_95ci_bps': ci[:, 0].tolist(), 'sharpe_difference_95ci': ci[:, 1].tolist(),
        'sortino_difference_95ci': [float(v) if np.isfinite(v) else None for v in ci[:, 2]],
        'selection_adjusted': False}


def select_schedule(candidates, config, last):
    first, end = pd.Timestamp(config.first_test), pd.Timestamp(last)
    schedule, splits = {}, []
    for boundary in pd.date_range(first, end, freq=f'{config.test_days}D'):
        start = boundary-pd.Timedelta(days=config.validation_days)
        scores = {}
        for gamma, curve in candidates.items():
            d = daily_returns(curve)
            valid = d.loc[(d.index >= start) & (d.index < boundary)]
            if len(valid) < 60:
                continue
            std = valid.net_return.std()
            scores[gamma] = float(valid.net_return.mean()/std*np.sqrt(365)) if std else 0.
        if scores:
            best = max(scores.values())
            stable = [g for g, s in scores.items() if s >= best-.05]
            selected = min(stable, key=lambda g: abs(np.log(g/config.gamma)))
        else:
            selected = config.gamma
        schedule[boundary] = selected
        splits.append({'test_start': boundary, 'test_end_exclusive': boundary+pd.Timedelta(days=config.test_days),
            'validation_start': start, 'validation_end_exclusive': boundary,
            'selected_gamma': selected, 'validation_sharpe': scores,
            'candidate_count': len(candidates), 'training': 'monthly trailing mature-label window',
            'account_reset': False})
    return schedule, splits
