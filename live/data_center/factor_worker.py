"""Production daily computation/backfill; quality gates remain independent.

Only public individual-trade archives are accepted. A final calculation with
unresolved source IDs is saved with source_complete=False, never relabelled
as a complete actionable input. No exchange/account configuration imports.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd

from data_center.base_strategy import FORMULA_VERSION, SOURCE_VERSION, symbol_name
from data_center.contract import atomic_json
from data_center.reference_formula_v2 import daily_factors, minute_features_from_events


def compute_archive(path, symbol, day, checksum, max_uncompressed_mb=700):
    started = time.monotonic()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].file_size > max_uncompressed_mb * 1024 * 1024:
            raise ValueError('unexpected or oversized raw archive')
        with archive.open(members[0]) as stream:
            d = pd.read_csv(stream, usecols=['id', 'price', 'qty', 'time', 'is_buyer_maker'])
    if d.empty or d.isna().any().any():
        raise ValueError('empty or null individual trades')
    if not d.is_buyer_maker.isin([True, False]).all():
        raise ValueError('invalid aggressor flag')
    if not np.isfinite(d[['id', 'time', 'price', 'qty']].to_numpy()).all() or (d[['price', 'qty']] <= 0).any().any():
        raise ValueError('invalid raw trade values')
    if ((d[['id', 'time']] % 1) != 0).any().any():
        raise ValueError('noninteger individual trade ID/time')
    if d.id.duplicated().any():
        raise ValueError('duplicate individual trade IDs')
    d = d.sort_values('id', kind='stable')
    jumps = np.diff(d.id.to_numpy(dtype=np.int64))
    missing_ids = int((jumps[jumps > 1] - 1).sum())
    d['event_time'] = pd.to_datetime(d.time, unit='ms', utc=True)
    if not d.event_time.dt.normalize().eq(pd.Timestamp(day, tz='UTC')).all():
        raise ValueError('public archive exchange-day mismatch')
    d['side'] = np.where(d.is_buyer_maker, 'sell', 'buy')
    d['notional'] = d.price * d.qty
    events = d.groupby(['event_time', 'side'], sort=True).agg(qty=('qty', 'sum'), notional=('notional', 'sum')).reset_index().rename(columns={'event_time': 'time'})
    # Compute even when incomplete, so diagnostics/coverage expose the actual
    # corrected factors. This does NOT bypass the publisher source quality gate.
    factors = daily_factors(minute_features_from_events(events), day)
    factors.update({'symbol': symbol, 'day': day, 'formula_version': FORMULA_VERSION,
                    'source_version': SOURCE_VERSION, 'status': 'final', 'source_complete': False,
                    'source_certified': False, 'quality_blockers': ['public_source_scope_not_certified'] +
                        (['unresolved_individual_trade_id_gaps'] if missing_ids else []),
                    'missing_trade_ids': missing_ids, 'trade_rows': len(d), 'event_rows': len(events),
                    'first_trade_id': int(d.id.iloc[0]), 'last_trade_id': int(d.id.iloc[-1]),
                    'archive_sha256': checksum, 'partition_basis': 'exchange_time_utc_day',
                    'available_at': datetime.now(timezone.utc).isoformat(),
                    'compute_seconds': round(time.monotonic() - started, 4)})
    return factors


def fetch_compute(root, symbol, day, max_archive_mb=80, max_uncompressed_mb=700):
    started = time.monotonic()
    name = f'{symbol}-trades-{day}.zip'
    url = f'https://data.binance.vision/data/futures/um/daily/trades/{quote(symbol, safe="")}/{quote(name, safe="")}'
    with urlopen(url + '.CHECKSUM', timeout=20) as response:
        expected = response.read(1024).decode().split()[0].lower()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('invalid archive checksum')
    work = Path(root) / 'archive_work'
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < 5 * 1024**3:
        raise RuntimeError('less than 5 GiB free; backfill stopped')
    # Only our freshly downloaded per-job temporary archive is removed; original
    # captures and existing archives are never touched. SHA256/URL are retained.
    with tempfile.TemporaryDirectory(prefix='raw-job-', dir=work) as directory:
        target = Path(directory) / name
        count, digest = 0, hashlib.sha256()
        with urlopen(Request(url, headers={'User-Agent': 'select-coin-base-factor/2'}), timeout=45) as response, target.open('wb') as handle:
            if int(response.headers.get('Content-Length', 0)) > max_archive_mb * 1024**2:
                raise ValueError('archive exceeds configured memory-safe size cap')
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > max_archive_mb * 1024**2:
                    raise ValueError('archive exceeds configured size cap')
                handle.write(chunk); digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('raw archive checksum mismatch')
        downloaded = time.monotonic()
        row = compute_archive(target, symbol, day, expected, max_uncompressed_mb=max_uncompressed_mb)
    row.update({'archive_url': url, 'archive_bytes': count,
                'download_seconds': round(downloaded - started, 4),
                'job_seconds': round(time.monotonic() - started, 4)})
    atomic_json(Path(root) / 'inputs/daily' / day / f'{symbol}.json', row)
    return row


def jobs(universe, end_day, days):
    end = datetime.fromisoformat(end_day).date()
    onboard = {symbol_name(k): v for k, v in universe.get('symbol_onboard_dates', {}).items()}
    symbols = sorted({symbol_name(s) for s in universe['tradable_symbols']} - {'BTCUSDT', 'ETHUSDT'})
    for offset in range(days - 1, -1, -1):
        day = (end - timedelta(days=offset)).isoformat()
        for symbol in symbols:
            if day >= onboard.get(symbol, day):
                yield symbol, day


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--universe', type=Path)
    p.add_argument('--symbols', default='')
    p.add_argument('--day')
    p.add_argument('--history-days', type=int, default=12)
    p.add_argument('--max-jobs', type=int, default=100)
    p.add_argument('--max-seconds', type=int, default=240)
    p.add_argument('--state-path', type=Path)
    args = p.parse_args()
    end_day = args.day or (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    if args.symbols:
        queue = [(symbol_name(s), end_day) for s in args.symbols.split(',')]
    else:
        queue = list(jobs(json.loads(args.universe.read_text()), end_day, args.history_days))
    state_path = args.state_path or args.root / 'compute_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    items = state.get('items', {})
    started, attempted = time.monotonic(), 0
    try:
        for symbol, day in queue:
            key = f'{day}/{symbol}'
            output = args.root / 'inputs/daily' / day / f'{symbol}.json'
            if output.exists():
                row = json.loads(output.read_text())
                if row.get('formula_version') == FORMULA_VERSION and row.get('source_version') == SOURCE_VERSION:
                    continue
            old = items.get(key, {})
            if old.get('retry_after_epoch', 0) > time.time():
                continue
            if attempted >= args.max_jobs or time.monotonic() - started >= args.max_seconds:
                break
            attempted += 1
            try:
                row = fetch_compute(args.root, symbol, day)
                items[key] = {k: row[k] for k in ('job_seconds', 'compute_seconds', 'download_seconds', 'archive_bytes', 'missing_trade_ids')}
                items[key]['status'] = 'computed_quality_blocked'
            except HTTPError as exc:
                items[key] = {'status': 'archive_not_published' if exc.code == 404 else 'download_error',
                              'http_status': exc.code, 'retry_after_epoch': time.time() + 3600}
                if exc.code in (418, 429):
                    break
            except Exception as exc:
                items[key] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}', 'retry_after_epoch': time.time() + 3600}
            print(json.dumps({'job': key, **items[key]}), flush=True)
    finally:
        now = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, {'formula_version': FORMULA_VERSION, 'source_version': SOURCE_VERSION,
                    'updated_at': now, 'items': items, 'attempted_this_run': attempted,
                    'statuses': dict(Counter(v['status'] for v in items.values())),
                    'submitted_to_exchange': False})


if __name__ == '__main__':
    main()
