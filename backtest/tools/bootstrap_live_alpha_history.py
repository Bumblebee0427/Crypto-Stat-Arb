#!/usr/bin/env python3
"""Rebuild live-v3 historical features from existing individual-trade files.

Only compact derived rows are written; original raw files are never modified.
Run with the live computation package on PYTHONPATH. Source roots are explicit
CLI arguments, with layout YEAR/YYYY_MM_DD/SYMBOL.parquet, in priority order.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import inspect
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.contracts import file_hash, digest, write_json
from research.optimizer.v3_history import economic_events, daily_fast, rolling_features, MAIN, SMALL, V3, historical_contract_allowed

CONTEXT = {}


def init_worker(context):
    CONTEXT.clear()
    CONTEXT.update(context)


def source(roots, day, symbol):
    for root in roots:
        p = Path(root)/str(day.year)/day.strftime('%Y_%m_%d')/(symbol+'.parquet')
        if p.exists(): return p
    return None




def compute_day_task(task):
    """Date-ordered work keeps neighbouring receipt files in the disk cache."""
    day, symbol, seed = task; c = CONTEXT
    try:
        lo, hi = day.value//1000, (day+pd.Timedelta(days=1)).value//1000
        frames, sources = [], []
        days = [day, day+pd.Timedelta(days=1)] if seed else [day-pd.Timedelta(days=1), day, day+pd.Timedelta(days=1)]
        for d in days:
            path = source(c['roots'], d, symbol)
            if path is None: raise ValueError('missing neighbouring receipt partition')
            stat = path.stat()
            sources.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
            frames.append(pd.read_parquet(path, columns=['id', 'timestamp', 'local_timestamp', 'side', 'price', 'amount'],
                filters=[('timestamp', '>=', lo-3_600_000_000), ('timestamp', '<', hi),
                         ('local_timestamp', '<=', hi+8*60_000_000)]))
        raw = pd.concat(frames, ignore_index=True)
        # A genuine late revision to the seed hour requires a raw reread.
        revised_seed = seed and ((raw.timestamp < lo) & (raw.local_timestamp > lo+8*60_000_000)).any()
        if revised_seed:
            path = source(c['roots'], day-pd.Timedelta(days=1), symbol)
            if path is None: raise ValueError('missing revised seed receipt partition')
            stat = path.stat(); sources.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
            prior = pd.read_parquet(path, columns=['id','timestamp','local_timestamp','side','price','amount'],
                filters=[('timestamp','>=',lo-3_600_000_000),('timestamp','<',hi)])
            raw = pd.concat([prior, raw], ignore_index=True); seed = None
        events, raw = economic_events(raw)
        ids = raw.loc[raw.timestamp >= lo, 'id'].to_numpy(np.int64)
        if len(ids) < 2: raise ValueError('insufficient economic trade IDs')
        span = int(ids.max()-ids.min()+1); missing = span-len(ids)
        if missing < 0 or missing/span > .10: raise ValueError('internal ID missing bound exceeds 10 percent')
        row = daily_fast(events, str(day.date()), None if seed is None else seed['next_day_00h_quantiles'])
        row.update(day=str(day.date()), symbol=symbol, first_id=int(ids.min()), last_id=int(ids.max()),
            economic_rows=len(ids), uncovered_internal_id_slots=missing, internal_id_missing_fraction=missing/span,
            source_certified=False, latest_receipt_us=int(raw.local_timestamp.max()),
            feature_available_at=str(day+pd.Timedelta(days=1, minutes=8)))
        if seed: row['previous_seed_row_digest'] = digest(seed)
        return row, None, sources
    except Exception as exc:
        return None, {'symbol': symbol, 'day': str(day.date()), 'reason': type(exc).__name__+': '+str(exc)[:180]}, []


def market(args):
    """Historical active market, exact 599 preclose hours and one-hour BTC lag."""
    out = args.out
    marker = out/'market.json'
    identity = {'market': str(args.market), 'size': args.market.stat().st_size,
                'mtime_ns': args.market.stat().st_mtime_ns, 'version': 2,
                'market_adapter_sha256':digest(inspect.getsource(market))}
    if marker.exists() and json.loads(marker.read_text()).get('identity') == identity:
        return
    frames = pd.read_pickle(args.market)
    btc = frames['BTC-USDT'].set_index('candle_begin_time').close
    btc_trend = btc.shift(1).pct_change(720, fill_method=None)
    rows = []
    prices = []
    daily_close, daily_quotes, funding = {}, {}, []
    for symbol, original in frames.items():
        s = symbol.replace('-', '')
        if not s.endswith('USDT') or s in ['BTCUSDT', 'ETHUSDT']: continue
        x = original.set_index('candle_begin_time').sort_index()
        if x.index.has_duplicates: raise ValueError('duplicate historical market hour')
        # An absent historical hour stays absent and invalidates the window.
        x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq='h'))
        observed = x.quote_volume.notna()
        vol = x.quote_volume.shift(1).rolling(599, min_periods=1).mean()
        count = observed.shift(1, fill_value=False).rolling(599, min_periods=1).sum()
        age = pd.Series(np.arange(len(x)), index=x.index).clip(upper=599)
        selection = x.loc[x.index.hour == 23]
        for stamp, row in selection.loc[(selection.index >= pd.Timestamp(args.start)-pd.Timedelta(days=35)) &
                                         (selection.index < pd.Timestamp(args.end)+pd.Timedelta(days=2))].iterrows():
            if not bool(row.get('是否交易', 1) == 1) or not observed.loc[stamp] or count.loc[stamp] != age.loc[stamp] or age.loc[stamp] < 1:
                continue
            rows.append({'asof': stamp+pd.Timedelta(hours=1), 'symbol': s,
                'quote_volume_mean': vol.loc[stamp],
                'btc': btc_trend.get(stamp, np.nan) if x.index.min() <= stamp-pd.Timedelta(hours=721) else np.nan,
                'onboard_day': x.index.min().normalize()})
        entry = x.loc[x.index.hour == 1]
        prices.extend({'asof': t.normalize(), 'symbol': s, 'price': float(v)}
                      for t, v in entry.avg_price_1m.items() if np.isfinite(v) and v > 0)
        closes = x.loc[x.index.hour == 23, 'close'].copy(); closes.index += pd.Timedelta(hours=1)
        daily_close[s] = closes
        quotes = x.quote_volume.resample('D').sum(min_count=24); quotes.index += pd.Timedelta(days=1)
        daily_quotes[s] = quotes
        f = x.loc[x.funding_fee.notna() & x.funding_fee.ne(0), 'funding_fee']
        funding.extend({'symbol': s, 'time': t, 'rate': float(v)} for t, v in f.items())
    # Include BTC and ETH in market factors despite their exclusion from selections.
    for symbol in ['BTC-USDT', 'ETH-USDT']:
        x = frames[symbol].set_index('candle_begin_time').sort_index()
        y = x.loc[x.index.hour == 23, 'close'].copy(); y.index += pd.Timedelta(hours=1)
        daily_close[symbol.replace('-', '')] = y
        q = x.quote_volume.resample('D').sum(min_count=24); q.index += pd.Timedelta(days=1)
        daily_quotes[symbol.replace('-', '')] = q
    pd.DataFrame(rows).to_parquet(out/'market_selection.parquet', index=False)
    pd.DataFrame(prices).to_parquet(out/'entry_prices_01h.parquet', index=False)
    pd.DataFrame(daily_close).sort_index().to_parquet(out/'daily_closes.parquet')
    pd.DataFrame(daily_quotes).sort_index().to_parquet(out/'daily_quotes.parquet')
    funding = pd.DataFrame(funding).sort_values(['symbol', 'time'])
    delta = funding.groupby('symbol').time.diff().dt.total_seconds()/3600
    funding['interval_hours'] = delta.where(delta.isin([1, 4, 8]))
    funding['source'] = 'historical_hourly_settlement_nonzero_only'
    funding.to_parquet(out/'settlements.parquet', index=False)
    write_json(marker, {'identity': identity, 'entry_benchmark': '01:00 UTC first-minute VWAP',
        'entry_delay_minutes': 60, 'publication_vintages_available': False,
        'funding_limitation': 'hourly panel cannot distinguish zero settlements from no settlement',
        'selection_rows': len(rows), 'symbols': len(daily_close)})


def assemble(out, exchange_info=None):
    from data_center.top50_strategy import select
    from data_center import base_strategy, top50_strategy
    frames, errors = [], []
    for p in sorted((out/'days').glob('*.json')):
        r = json.loads(p.read_text()); frames.extend(r['daily']); errors.extend(r['errors'])
    raw = pd.DataFrame(frames)
    raw.to_parquet(out/'daily_v3.parquet', index=False)
    market = pd.read_parquet(out/'market_selection.parquet')
    if exchange_info is not None:
        metadata = {r['symbol']:r for r in json.loads(exchange_info.read_text())['symbols']}
        candidates = set(market.symbol)
        allowed = {s for s in candidates if historical_contract_allowed(s, metadata)}
        market = market.loc[market.symbol.isin(allowed)]
        raw = raw.loc[raw.symbol.isin(allowed)]
        write_json(out/'historical_scope.json', {'metadata_sha256':file_hash(exchange_info),
            'excluded_symbols':sorted(candidates-allowed), 'unknown_legacy_symbols_retained':sorted(candidates-set(metadata)),
            'current_status_not_used':True, 'rule':'PERPETUAL USDT; production stable/leveraged exclusions; historical lifecycle flags retained'})
    market = market.set_index(['symbol','asof'])
    features = []
    for symbol, part in raw.groupby('symbol'):
        part = part.set_index(pd.to_datetime(part.day)).sort_index()
        observed_days = part.index
        # Missing mature days invalidate all dependent 12-day features.
        for asof in pd.date_range(part.index.min()+pd.Timedelta(days=1), part.index.max()+pd.Timedelta(days=1)):
            if (symbol, asof) not in market.index: continue
            m = market.loc[(symbol, asof)]
            onboard = pd.Timestamp(m.onboard_day)
            required = pd.date_range(max(asof-pd.Timedelta(days=12), onboard+pd.Timedelta(days=1)), asof-pd.Timedelta(days=1))
            if not len(required) or not required.isin(observed_days).all(): continue
            history = part.loc[required]
            scores = rolling_features(history)
            if scores is None: continue
            features.append({'asof': asof, 'symbol': symbol, **scores,
                'btc': float(m.btc), 'quote_volume_mean': float(m.quote_volume_mean)})
    features = pd.DataFrame(features)
    features.to_parquet(out/'universe_features.parquet', index=False)
    panel = []
    for asof, part in features.groupby('asof'):
        rows = part.set_index('symbol').drop(columns='asof').to_dict('index')
        if len(rows) < 20: continue
        available = part.btc.dropna()
        if available.empty: continue
        chosen = select(rows, float(available.iloc[0]))
        for symbol, weight in chosen['allocations'].items():
            if not weight: continue
            side = 1 if weight > 0 else -1
            panel.append({'asof': asof, 'symbol': symbol, 'side': side,
                'baseline': weight*2, 'main': rows[symbol]['main'],
                'small': rows[symbol]['small_long' if side > 0 else 'small_short'],
                'formula_version': V3, 'entry_at': asof+pd.Timedelta(hours=1),
                'feature_available_at': asof+pd.Timedelta(minutes=8),
                'exit_at': asof+pd.Timedelta(days=1, hours=1),
                'label_available_at': asof+pd.Timedelta(days=1, hours=2),
                'sample_source': 'historical_v3_individual_trade_replay_01h'})
    panel = pd.DataFrame(panel)
    for f in ['main', 'small']:
        panel[f+'_rank'] = panel.groupby(['asof', 'side'])[f].rank(pct=True)*2-1
    prices = pd.read_parquet(out/'entry_prices_01h.parquet')
    panel = panel.merge(prices.rename(columns={'price': 'entry_price'}), on=['asof', 'symbol'], how='left')
    exits = prices.rename(columns={'price': 'exit_price'}).copy(); exits['asof'] -= pd.Timedelta(days=1)
    panel = panel.merge(exits, on=['asof', 'symbol'], how='left')
    panel['future_return'] = panel.exit_price/panel.entry_price-1
    panel.sort_values(['asof', 'side', 'symbol']).to_parquet(out/'alpha_panel.parquet', index=False)
    pd.DataFrame(errors).to_parquet(out/'source_exclusions.parquet', index=False)
    write_json(out/'panel_manifest.json', {'formula_version': V3, 'rows': len(panel),
        'days': panel['asof'].nunique(), 'mature_rows': int(panel.future_return.notna().sum()),
        'first': str(panel['asof'].min()), 'last': str(panel['asof'].max()),
        'panel_sha256': file_hash(out/'alpha_panel.parquet'), 'source_certified': False,
        'assembly_sha256':digest(inspect.getsource(assemble)+inspect.getsource(rolling_features)),
        'selector_sha256':file_hash(Path(base_strategy.__file__)),
        'selector_profile_sha256':file_hash(Path(top50_strategy.__file__)),
        'limitations': ['Historical capture is not exchange completeness certification.',
            'Historical market lifecycle flags and revisions are retained; original publication vintages unavailable.',
            'Conservative fixed 01:00 UTC benchmark differs from actual live execution minutes.'],
        'completed_at': datetime.now(timezone.utc).isoformat()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, action='append', required=True)
    p.add_argument('--market', type=Path, required=True)
    p.add_argument('--start', default='2025-06-18')
    p.add_argument('--end', default='2026-10-07')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--max-symbols', type=int, default=0)
    p.add_argument('--assemble-only', action='store_true')
    p.add_argument('--exchange-info', type=Path, required=True, help='official contract classifications; never filter by current trading status')
    p.add_argument('--force', action='store_true', help='recheck unchanged historical partitions and prior exclusions')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    market(a)
    symbols = sorted(set(pd.read_parquet(a.out/'market_selection.parquet').symbol))
    if a.max_symbols: symbols = symbols[:a.max_symbols]
    import inspect
    identity = digest({'start': a.start, 'end': a.end, 'roots': list(map(str, a.raw_root)),
        'computation': inspect.getsource(compute_day_task)+inspect.getsource(economic_events)+inspect.getsource(daily_fast)})
    CONTEXT.update(start=pd.Timestamp(a.start), end=pd.Timestamp(a.end), roots=a.raw_root,
                   identity=identity, out=a.out)
    started = time.monotonic()
    if not a.assemble_only:
        completed, count, errors = [], 0, 0
        prior_rows = {}; calendar = pd.date_range(a.start, a.end)
        # Spawn avoids copying the large market-preparation heap. Recycle the
        # pool weekly so pandas/Arrow allocator fragmentation stays bounded.
        for begin in range(0, len(calendar), 7):
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'),
                    initializer=init_worker, initargs=(dict(CONTEXT),)) as pool:
                for day in calendar[begin:begin+7]:
                    target = a.out/'days'/(str(day.date())+'.json')
                    if target.exists() and not a.force:
                        cached = json.loads(target.read_text())
                        valid = cached['identity'] == identity
                        if valid:
                            for item in cached['sources']:
                                try: stat = Path(item['path']).stat()
                                except OSError: valid = False; break
                                if stat.st_size != item['size'] or stat.st_mtime_ns != item['mtime_ns']:
                                    valid = False; break
                        if valid:
                            for row in cached['daily']:
                                if row.get('previous_seed_row_digest') and digest(prior_rows.get(row['symbol'])) != row['previous_seed_row_digest']:
                                    valid = False; break
                        if valid:
                            prior_rows = {r['symbol']:r for r in cached['daily']}
                            completed.append(str(day.date())); continue
                    available = set()
                    for raw_root in a.raw_root:
                        folder = raw_root/str(day.year)/day.strftime('%Y_%m_%d')
                        available.update(p.stem for p in folder.glob('*.parquet'))
                    wanted = sorted(available.intersection(symbols))
                    values, failures, sources = [], [], []
                    jobs = {pool.submit(compute_day_task, (day, s, prior_rows.get(s))): s for s in wanted}
                    for f in as_completed(jobs):
                        row, error, proof = f.result()
                        if row is not None: values.append(row)
                        if error is not None: failures.append(error)
                        sources.extend(proof)
                    write_json(target, {'identity': identity, 'day': str(day.date()), 'daily': values,
                        'errors': failures, 'sources': sources})
                    prior_rows = {r['symbol']:r for r in values}
                    completed.append(str(day.date())); count += len(values); errors += len(failures)
                    state = {'phase': 'rebuild', 'completed_days': len(completed),
                        'total_days': len(calendar), 'computed_rows': count,
                        'excluded_symbol_days': errors, 'last_day': str(day.date()),
                        'seconds': round(time.monotonic()-started, 1), 'identity': identity}
                    write_json(a.out/'job_status.json', state)
                    print(json.dumps(state), flush=True)
    assemble(a.out, a.exchange_info)
    print((a.out/'panel_manifest.json').read_text(), flush=True)


if __name__ == '__main__': main()
