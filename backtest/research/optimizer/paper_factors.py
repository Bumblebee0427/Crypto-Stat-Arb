"""Liu–Tsyvinski–Wu (2019 supplied version) risk-factor construction.

Weekly 30/40/30 independent sorts, capitalization-weighted legs, daily returns
between weekly formations. This is a Binance perpetual daily adaptation, not a
replication of the paper's 2014–2018 CoinMarketCap universe or Treasury returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .contracts import file_hash, write_json


CORE_FACTORS = ['CMKT', 'CSMB', 'CMOM']
EXTRA_FACTORS = ['CILLIQ10', 'CVOL30', 'CSKEW7']
STABLES = {'USDC', 'USDT', 'BUSD', 'TUSD', 'USDP', 'FDUSD', 'DAI', 'USDE', 'USDD', 'AEUR', 'EUR'}


@dataclass(frozen=True)
class PaperRiskConfig:
    window_days: int = 180
    min_observations: int = 20
    ewma_half_life_days: float = 30.
    cap_publication_delay_days: int = 2
    cap_max_stale_days: int = 3
    minimum_market_cap: float = 1_000_000.
    minimum_universe: int = 12
    tail_fraction: float = .3
    beta_prior_observations: float = 20.
    ridge_observations: float = 5.
    variance_floor: float = .01**2
    short_history_variance: float = .1**2

    def __post_init__(self):
        if self.window_days < self.min_observations or self.min_observations < 3:
            raise ValueError('invalid risk window')
        if self.ewma_half_life_days <= 0 or self.cap_publication_delay_days < 1:
            raise ValueError('invalid time weighting or publication delay')
        if not 0 < self.tail_fraction < .5 or self.minimum_universe < 6:
            raise ValueError('invalid factor sorts')


def normalized_symbol(symbol):
    return str(symbol).replace('-', '').replace('_', '').upper()


def load_capitalization(folder, closes, config, out):
    """Use actual historical circulating market caps; never infer cap from volume.

    Timestamp D is treated as end-of-day data available at D+2: one day to close
    the observation and one conservative publication day. Historical publication
    vintages are unavailable. Stale observations expire, and no backfill is used.
    """
    names = {normalized_symbol(s): s for s in closes.columns}
    frames, hashes, rejected, aliases = [], {}, [], []
    candidates = {}
    for path in sorted(Path(folder).glob('*.csv')):
        name = names.get(normalized_symbol(path.stem))
        if name is not None:
            candidates.setdefault(name, []).append(path)
    for name, paths in candidates.items():
        path = min(paths, key=lambda p: (p.stem != name, p.name))
        if len(paths) > 1:
            aliases.append({'symbol': name, 'selected_exact_or_lexical': str(path),
                'ignored_aliases': [str(p) for p in paths if p != path]})
        raw = pd.read_csv(path, encoding='gbk', skiprows=1,
            usecols=['candle_begin_time', 'id', 'circulating_mcap'])
        raw['source_date'] = pd.to_datetime(raw.candle_begin_time, errors='raise')
        raw['available_at'] = raw.source_date+pd.Timedelta(days=config.cap_publication_delay_days)
        raw['symbol'] = name
        valid = (raw.circulating_mcap > 0) & np.isfinite(raw.circulating_mcap) & raw.id.notna()
        rejected.append({'symbol': name, 'invalid_rows': int((~valid).sum())})
        frames.append(raw.loc[valid, ['source_date', 'available_at', 'symbol', 'id', 'circulating_mcap']])
        hashes[str(path)] = file_hash(path)
    if not frames:
        raise ValueError('no historical circulating-cap data matches market symbols')
    raw = pd.concat(frames, ignore_index=True).sort_values(['available_at', 'symbol'])
    if raw.duplicated(['available_at', 'symbol']).any():
        raise ValueError('ambiguous duplicate capitalization observations')
    raw.to_parquet(out/'capitalization_observations.parquet', index=False)
    calendar = closes.index
    cap = raw.pivot(index='available_at', columns='symbol', values='circulating_mcap')
    ids = raw.pivot(index='available_at', columns='symbol', values='id')
    dates = raw.assign(source_available_at=raw.available_at).pivot(
        index='available_at', columns='symbol', values='source_available_at')
    # Include pre-market observation dates before forward filling onto calendar.
    union = calendar.union(cap.index).sort_values()
    def aligned(frame):
        return frame.reindex(index=union, columns=closes.columns).ffill(
            limit=config.cap_max_stale_days).reindex(calendar)
    cap, ids, dates = (aligned(f) for f in [cap, ids, dates])
    latest = pd.Timestamp(raw.available_at.max())
    write_json(out/'capitalization_sources.json', {'source_sha256': hashes,
        'first_source_date': str(raw.source_date.min()), 'last_source_date': str(raw.source_date.max()),
        'latest_assumed_available_at': str(latest), 'files': len(hashes),
        'invalid_observations': rejected, 'file_alias_resolution': aliases,
        'publication_vintage_available': False,
        'publication_assumption_days_after_source_date': config.cap_publication_delay_days,
        'forward_fill_max_days': config.cap_max_stale_days})
    return cap, ids, dates, latest


def historical_features(closes, quotes):
    returns = closes.pct_change(fill_method=None)
    momentum = closes/closes.shift(21)-1.
    momentum = momentum.where(returns.rolling(21, min_periods=21).count() == 21)
    # Matches AddAmihud's ratio of rolling sums, using decimal returns instead of %.
    illiq = returns.abs().rolling(10, min_periods=10).sum()/quotes.rolling(10, min_periods=10).sum()
    vol = returns.rolling(30, min_periods=30).std(ddof=1)
    # polars rolling_skew defaults to biased g1; pandas returns corrected G1.
    skew = returns.rolling(7, min_periods=7).skew()*5/np.sqrt(7*6)
    skew = skew.where(returns.rolling(7, min_periods=7).std(ddof=0) > 0)
    return {'momentum21': momentum, 'illiquidity10': illiq, 'volatility30': vol, 'skewness7': skew}


def paper_week_boundaries(calendar):
    """52 weeks/year: first 51 have seven days, final week has eight/nine."""
    return pd.DatetimeIndex([t for t in calendar if (t.dayofyear-1) % 7 == 0 and (t.dayofyear-1)//7 <= 51])


def tail_portfolios(characteristic, cap, fraction=.3):
    good = characteristic.loc[np.isfinite(characteristic) & (cap > 0)]
    # Stable symbol tie-breaks; no future-dependent eligibility.
    order = good.sort_index().sort_values(kind='stable').index
    n = int(np.floor(len(order)*fraction))
    if n < 1:
        return None
    low, high = order[:n], order[-n:]
    return {'low': cap.loc[low]/cap.loc[low].sum(), 'high': cap.loc[high]/cap.loc[high].sum()}


def leg_returns(closes, formation, end, weights):
    """Fixed formation quantities, with lagged wealth weights within the week.

    A missing held constituent makes the leg unavailable. We never silently
    remove a future delisting or renormalize its weight onto surviving coins.
    """
    prices = closes.loc[(closes.index >= formation) & (closes.index <= end), weights.index]
    relative = prices.div(prices.iloc[0])
    wealth = relative.mul(weights, axis=1).sum(axis=1, min_count=len(weights))
    daily = wealth.pct_change(fill_method=None).iloc[1:]
    weekly = float(wealth.iloc[-1]-1) if np.isfinite(wealth.iloc[-1]) else np.nan
    return daily, weekly


def build_factors(closes, quotes, cap, ids, available, config, out=None):
    if not closes.index.is_unique or not closes.index.is_monotonic_increasing:
        raise ValueError('expected unique chronological daily closes')
    quotes = quotes.reindex(index=closes.index, columns=closes.columns)
    features = historical_features(closes, quotes)
    boundaries = paper_week_boundaries(closes.index)
    factor = pd.DataFrame(np.nan, index=closes.index, columns=CORE_FACTORS+EXTRA_FACTORS)
    members, weeks = [], []
    active_volume = quotes.rolling(7, min_periods=1).sum()
    crypto = pd.Series([str(s).removesuffix('-USDT') not in STABLES for s in closes.columns], index=closes.columns)
    for k, formation in enumerate(boundaries):
        end = boundaries[k+1] if k+1 < len(boundaries) else closes.index[-1]
        if end <= formation:
            continue
        caps = cap.loc[formation]
        mask = crypto & (caps >= config.minimum_market_cap) & (closes.loc[formation] > 0) & (quotes.loc[formation] > 0) & ids.loc[formation].notna()
        pool = pd.DataFrame({'cap': caps[mask], 'id': ids.loc[formation, mask], 'volume': active_volume.loc[formation, mask]})
        pool = pool.sort_index().sort_values('volume', ascending=False, kind='stable').drop_duplicates('id')
        if len(pool) < config.minimum_universe:
            continue
        ports = {'market': pool.cap/pool.cap.sum()}
        definitions = [('size', pool.cap),
            ('momentum', features['momentum21'].loc[formation, pool.index]),
            ('illiquidity', features['illiquidity10'].loc[formation, pool.index]),
            ('volatility', features['volatility30'].loc[formation, pool.index]),
            ('skewness', features['skewness7'].loc[formation, pool.index])]
        for name, characteristic in definitions:
            valid = characteristic.dropna()
            if len(valid) >= config.minimum_universe:
                tails = tail_portfolios(valid, pool.cap, config.tail_fraction)
                if tails:
                    ports.update({name+'_'+side: w for side, w in tails.items()})
        leg, weekly_leg = {}, {}
        for name, weights in ports.items():
            leg[name], weekly_leg[name] = leg_returns(closes, formation, end, weights)
            if out is not None:
                for symbol, weight in weights.items():
                    members.append({'formation': formation, 'holding_end': end, 'portfolio': name,
                        'symbol': symbol, 'token_id': int(pool.loc[symbol, 'id']), 'weight': float(weight),
                        'circulating_mcap': float(pool.loc[symbol, 'cap']),
                        'cap_available_at': available.loc[formation, symbol],
                        'momentum21': float(features['momentum21'].loc[formation, symbol])})
        definitions = {'CMKT': ('market', None), 'CSMB': ('size_low', 'size_high'),
            'CMOM': ('momentum_high', 'momentum_low'),
            'CILLIQ10': ('illiquidity_high', 'illiquidity_low'),
            'CVOL30': ('volatility_low', 'volatility_high'), 'CSKEW7': ('skewness_high', 'skewness_low')}
        for name, (a, b) in definitions.items():
            if a not in leg or (b is not None and b not in leg):
                continue
            result = leg[a] if b is None else leg[a]-leg[b]
            factor.loc[result.index, name] = result
            weeks.append({'formation': formation, 'holding_end': end, 'factor': name,
                'weekly_leg_spread': weekly_leg[a]-(weekly_leg[b] if b else 0.),
                'daily_observations': int(result.notna().sum()), 'universe_coins': len(pool),
                'long_leg_coins': len(ports[a]), 'short_leg_coins': len(ports[b]) if b else 0})
    if out is not None:
        factor.rename_axis('available_at').reset_index().to_parquet(out/'factor_returns_daily.parquet', index=False)
        pd.DataFrame(members).to_parquet(out/'factor_memberships.parquet', index=False)
        pd.DataFrame(weeks).to_csv(out/'factor_returns_weekly.csv', index=False)
        write_json(out/'factor_definition.json', {'paper': 'Liu, Tsyvinski & Wu, supplied NBER May 2019 version',
            'paper_url': 'https://www.nber.org/papers/w25882',
            'paper_sections': 'Section 2, Section 4, Table 2; PDF printed pages 4–5, 9, 15–16',
            'risk_config': asdict(config), 'core_factors': CORE_FACTORS, 'extension_factors': EXTRA_FACTORS,
            'definitions': {'CMKT': 'circulating-cap-weighted market price return; risk-free rate assumed zero',
                'CSMB': 'bottom 30% capitalization return minus top 30%',
                'CMOM': 'top 30% past 21-calendar-day simple return minus bottom 30%; no skipped week',
                'CILLIQ10': 'high minus low rolling sum(abs(price return),10)/sum(quote volume,10)',
                'CVOL30': 'low minus high past 30-day price-return sample standard deviation',
                'CSKEW7': 'high minus low past 7-day biased price-return skewness'},
            'formation': '52 paper-style weeks/year; independent univariate 30/40/30 sorts, floor rounding symmetric tails',
            'weights': 'raw historical circulating market caps; fixed constituent quantities within formation week',
            'factor_universe': 'historically available Binance perpetual prices and known fresh capitalization; excludes stablecoins; duplicate CMC IDs resolved using past seven-day quote volume',
            'missing_constituent': 'mark whole leg unavailable; no future selection or survivor renormalization',
            'adaptations': ['daily risk-factor observations from weekly formed legs, instead of weekly regression data',
                'Binance perpetual price universe rather than all historical CoinMarketCap assets',
                'zero risk-free-rate assumption; funding and trade fees excluded from theoretical factor returns',
                'formation weights use raw circulating cap; no return winsorization',
                'three extra style factors are reference-code extensions, not the paper three-factor model',
                'daily long-short spreads do not compound into the weekly difference of cumulative leg returns'],
            'historical_publication_vintage_available': False})
    return factor
