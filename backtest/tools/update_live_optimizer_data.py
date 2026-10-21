#!/usr/bin/env python3
"""Refresh public Binance/CMC capitalization history and closed futures days.

Raw response timestamps and hashes are retained. Historical quote timestamp D
is recorded as the closing observation for D-1; the research model still applies
its conservative two-calendar-day availability assumption. No account access.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.paper_factors import load_capitalization, build_factors, PaperRiskConfig
from research.optimizer.contracts import write_json, file_hash

CAP_URL = 'https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/complianceSymbolList'
HISTORY_URL = 'https://www.binance.com/bapi/composite/v1/public/promo/cmc/cryptocurrency/quotes/historical'
FUTURES = 'https://fapi.binance.com/fapi/v1/'


def get(url, cache, refresh=False):
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'select-coin-public-research/1.0',
                                               'Accept': 'application/json'}), timeout=35) as r:
                value = json.load(r)
            cache.parent.mkdir(parents=True, exist_ok=True)
            write_json(cache, value)
            return value
        except Exception:
            if attempt == 2: raise
            time.sleep(attempt+1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--legacy-cap-root', type=Path)
    p.add_argument('--seed-closes', type=Path)
    p.add_argument('--seed-quotes', type=Path)
    p.add_argument('--asof', default=str(pd.Timestamp.now(tz='UTC').date()))
    p.add_argument('--history-start', default='2026-04-13')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--skip-market', action='store_true')
    p.add_argument('--max-symbols', type=int, default=0)
    args = p.parse_args()
    root = args.root; root.mkdir(parents=True, exist_ok=True)
    asof = pd.Timestamp(args.asof); stamp = asof.strftime('%Y-%m-%d')
    info = get(FUTURES+'exchangeInfo', root/'raw'/stamp/'exchangeInfo.json')
    products = get(CAP_URL, root/'raw'/stamp/'capitalization_current.json')
    if products.get('code') != '000000' or not isinstance(products.get('data'), list):
        raise ValueError('Binance capitalization listing unavailable')
    symbols = sorted({s['symbol'] for s in info['symbols']
                      if s.get('quoteAsset') == 'USDT' and s.get('contractType') == 'PERPETUAL'})
    active = {s['symbol'] for s in info['symbols'] if s.get('status') == 'TRADING'}
    official = {r['symbol']: int(r['cmcUniqueId']) for r in products['data']
                if r.get('cmcUniqueId') and r.get('symbol')}
    mapping = {}; rejected = []
    for symbol in symbols:
        ident = official.get(symbol)
        if ident is None and args.legacy_cap_root:
            legacy = args.legacy_cap_root/(symbol[:-4]+'-USDT.csv')
            if legacy.exists():
                try:
                    ids = pd.read_csv(legacy, encoding='gbk', skiprows=1, usecols=['id']).id.dropna()
                    if len(ids): ident = int(ids.iloc[-1])
                except (ValueError, UnicodeError): pass
        if ident is not None: mapping[symbol] = ident
        else: rejected.append(symbol)
    ids = sorted(set(mapping.values()))
    if args.max_symbols:
        ids = ids[:args.max_symbols]
        mapping = {s: i for s, i in mapping.items() if i in ids}
        symbols = [s for s in symbols if s in mapping]
    write_json(root/'mapping.json', {'symbols': mapping, 'missing_cmc_ids': rejected,
        'current_official_symbols': len(official), 'requested_futures_symbols': len(symbols)})
    print(json.dumps({'stage': 'mapping', 'futures_symbols': len(symbols), 'cmc_ids': len(ids),
                      'known_cap_symbols': len(mapping), 'missing_ids': len(rejected)}), flush=True)
    histories = {}; failures = []
    # Reuse the complete previous snapshot, refreshing a conservative overlap.
    prior = {}
    for f in sorted((root/'raw').glob('*/cap_batch_*.json')):
        try:
            body = json.loads(f.read_text()).get('data', {}).get('body', {})
            data = body.get('data', {})
            if 'quotes' in data: data = {str(data['id']): data}
            for ident, value in data.items():
                if isinstance(value, dict) and value.get('quotes'):
                    prior.setdefault(int(ident), {}).update({q['timestamp']: q for q in value['quotes']})
        except (ValueError, KeyError, TypeError): pass
    def cap_batch(batch):
        end = asof.tz_localize('UTC')
        requested_start = pd.Timestamp(args.history_start, tz='UTC')
        if all(i in prior for i in batch):
            oldest_latest = min(pd.Timestamp(max(prior[i])) for i in batch)
            requested_start = max(requested_start, oldest_latest-pd.Timedelta(days=3))
        params = {'id': ','.join(map(str, batch)), 'time_start': int(requested_start.timestamp()),
                  'time_end': int(end.timestamp()), 'interval': 'daily', 'convert': 'USD'}
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        path = root/'raw'/stamp/('cap_batch_'+key+'.json')
        value = get(HISTORY_URL+'?'+urlencode(params), path)
        body = value.get('data', {}).get('body', {})
        if body.get('status', {}).get('error_code') != 0: raise ValueError('historical cap query failed')
        data = body.get('data', {})
        if 'quotes' in data: data = {str(data['id']): data}
        return data
    batches = [ids[i:i+40] for i in range(0, len(ids), 40)]
    with ThreadPoolExecutor(min(args.workers, 2)) as pool:
        for f in as_completed({pool.submit(cap_batch, b): b for b in batches}):
            try:
                for ident, value in f.result().items():
                    ident = int(ident)
                    combined = dict(prior.get(ident, {}))
                    combined.update({q['timestamp']: q for q in value.get('quotes', [])})
                    histories[ident] = [combined[k] for k in sorted(combined)]
            except Exception as exc: failures.append({'stage': 'cap', 'type': type(exc).__name__})
    cap_root = root/'coin-cap'; cap_root.mkdir(exist_ok=True)
    cap_rows = 0
    for symbol, ident in mapping.items():
        rows = []
        for q in histories.get(ident, []):
            usd = q.get('quote', {}).get('USD', {}); timestamp = pd.Timestamp(q['timestamp'])
            if timestamp > asof.tz_localize('UTC'): continue
            cap = usd.get('market_cap')
            if cap is None or not np.isfinite(float(cap)) or float(cap) <= 0: continue
            rows.append({'candle_begin_time': str((timestamp-pd.Timedelta(days=1)).date()),
                'id': ident, 'circulating_mcap': cap, 'usd_price': usd.get('price'),
                'circulating_supply': usd.get('circulating_supply'), 'total_supply': usd.get('total_supply')})
        if rows:
            frame = pd.DataFrame(rows).drop_duplicates('candle_begin_time').sort_values('candle_begin_time')
            with (cap_root/(symbol[:-4]+'-USDT.csv')).open('w', encoding='gbk') as stream:
                stream.write('Binance public website / CMC historical USD quotes; timestamp D belongs to close of D-1\n')
                frame.to_csv(stream, index=False)
            cap_rows += len(frame)
    print(json.dumps({'stage': 'capitalization', 'rows': cap_rows, 'successful_ids': len(histories),
                      'failures': failures}), flush=True)
    if not histories: raise ValueError('no capitalization history downloaded')
    if args.skip_market: return
    # Include frozen delisted histories rather than deleting them from old formations.
    first = asof-pd.Timedelta(days=165)
    closes = pd.read_parquet(args.seed_closes).loc[first:asof] if args.seed_closes else pd.DataFrame()
    quotes = pd.read_parquet(args.seed_quotes).loc[first:asof] if args.seed_quotes else pd.DataFrame()
    for key in ['closes', 'quotes']:
        target = root/(key+'.parquet')
        if target.exists():
            old = pd.read_parquet(target).loc[first:asof]
            if key == 'closes': closes = old.combine_first(closes)
            else: quotes = old.combine_first(quotes)
    def market(symbol):
        name = symbol[:-4]+'-USDT'
        start = first
        if name in closes:
            valid = closes[name].dropna()
            if len(valid): start = max(first, valid.index.max()-pd.Timedelta(days=2))
        params = {'symbol': symbol, 'interval': '1d', 'startTime': int(start.tz_localize('UTC').timestamp()*1000),
                  'endTime': int(asof.tz_localize('UTC').timestamp()*1000)-1, 'limit': 200}
        raw = get(FUTURES+'klines?'+urlencode(params), root/'raw'/stamp/'klines'/(symbol+'.json'))
        if not isinstance(raw, list): raise ValueError('invalid klines')
        rows = {pd.to_datetime(v[0], unit='ms')+pd.Timedelta(days=1): (float(v[4]), float(v[7]))
                for v in raw if v[6] < int(asof.tz_localize('UTC').timestamp()*1000)}
        if symbol in active and asof not in rows:
            # A boundary request can reach Binance before its candle cache rolls.
            # An incomplete cached reply must not remain authoritative all day.
            raw = get(FUTURES+'klines?'+urlencode(params),
                      root/'raw'/stamp/'klines'/(symbol+'.json'), refresh=True)
            rows = {pd.to_datetime(v[0], unit='ms')+pd.Timedelta(days=1): (float(v[4]), float(v[7]))
                    for v in raw if v[6] < int(asof.tz_localize('UTC').timestamp()*1000)}
        return name, rows
    updates_c, updates_q = {}, {}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(market, s): s for s in symbols}
        for count, f in enumerate(as_completed(futures), 1):
            try:
                name, rows = f.result()
                updates_c[name] = pd.Series({t: v[0] for t, v in rows.items()}, dtype=float)
                updates_q[name] = pd.Series({t: v[1] for t, v in rows.items()}, dtype=float)
            except Exception as exc:
                failures.append({'stage': 'market', 'symbol': futures[f], 'type': type(exc).__name__})
            if count % 100 == 0: print(json.dumps({'stage': 'market', 'done': count, 'total': len(symbols)}), flush=True)
    closes = pd.DataFrame(updates_c).combine_first(closes).sort_index()
    quotes = pd.DataFrame(updates_q).combine_first(quotes).sort_index()
    calendar = pd.date_range(first, asof)
    closes = closes.reindex(calendar); quotes = quotes.reindex(index=calendar, columns=closes.columns)
    if asof not in closes.index or closes.loc[asof].notna().sum() < 100:
        raise ValueError('latest official daily market has insufficient coverage')
    closes.to_parquet(root/'closes.parquet'); quotes.to_parquet(root/'quotes.parquet')
    cfg = PaperRiskConfig(window_days=90, ewma_half_life_days=14)
    cap, cap_ids, available, _ = load_capitalization(cap_root, closes, cfg, root)
    factors = build_factors(closes, quotes, cap, cap_ids, available, cfg, out=root)
    returns = closes.pct_change(fill_method=None); returns.to_parquet(root/'returns.parquet')
    factors.to_parquet(root/'factors.parquet')
    eligible = pd.read_parquet(root/'capitalization_observations.parquet')
    eligible = eligible.loc[eligible.available_at <= asof]
    latest_source = eligible.source_date.max()
    provenance = {'provider': 'Binance public website / CoinMarketCap; futures public daily klines',
        'capitalization_latest_source_date': str(latest_source), 'cap_publication_delay_days': 2,
        'maximum_available_at': asof.isoformat(), 'asof': asof.isoformat(),
        'publication_vintages_available': False, 'quote_timestamp_rule': 'timestamp D mapped to source day D-1',
        'raw_history_start': args.history_start, 'missing_cmc_symbols': rejected,
        'latest_market_symbols': int(closes.loc[asof].notna().sum()),
        'latest_factor_complete': bool(factors.loc[asof].notna().all()),
        'factor_rows_last90': int(factors.loc[factors.index > asof-pd.Timedelta(days=90)].notna().all(axis=1).sum()),
        'failures': failures}
    write_json(root/'provenance.json', provenance)
    write_json(root/'inputs.json', {'returns': 'returns.parquet', 'factors': 'factors.parquet',
        'settlements': 'settlements.parquet', 'provenance': 'provenance.json'})
    print(json.dumps({'stage': 'complete', **provenance}, default=str), flush=True)


if __name__ == '__main__': main()
