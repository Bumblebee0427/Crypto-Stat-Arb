"""Bounded official-trade backfill for the isolated previous-hour factor v3."""
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

from data_center.base_strategy import SOURCE_VERSION
from data_center.contract import atomic_json
from data_center.factor_worker import jobs
from data_center.lagged_quantile_v3 import (
    FORMULA_CODE_SHA256, FORMULA_VERSION, daily_from_events, events_from_trades, quantiles,
)


def download(symbol: str, day: str, work: Path, max_mb=400) -> tuple[Path, str, int, str]:
    name = f'{symbol}-trades-{day}.zip'
    url = (f'https://data.binance.vision/data/futures/um/daily/trades/'
           f'{quote(symbol, safe="")}/{quote(name, safe="")}')
    with urlopen(url + '.CHECKSUM', timeout=20) as response:
        expected = response.read(1024).decode().split()[0].lower()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('invalid official archive checksum')
    path = work / name
    count, digest = 0, hashlib.sha256()
    with urlopen(Request(url, headers={'User-Agent': 'select-coin-lagged-factor/3'}), timeout=45) as response, path.open('wb') as out:
        if int(response.headers.get('Content-Length', 0)) > max_mb * 1024**2:
            raise ValueError('archive exceeds configured size cap')
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > max_mb * 1024**2:
                raise ValueError('archive exceeds configured size cap')
            out.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('official archive checksum mismatch')
    return path, expected, count, url


def read_events(path: Path, day: str, max_uncompressed_mb=2500) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].file_size > max_uncompressed_mb * 1024**2:
            raise ValueError('unexpected or oversized raw archive')
        with archive.open(members[0]) as stream:
            d = pd.read_csv(stream, usecols=['id', 'price', 'qty', 'time', 'is_buyer_maker'])
    if d.empty or d.isna().any().any() or not d.is_buyer_maker.isin([True, False]).all():
        raise ValueError('invalid official individual trades')
    if (not np.isfinite(d[['id', 'time', 'price', 'qty']].to_numpy()).all()
            or (d[['price', 'qty']] <= 0).any().any()
            or ((d[['id', 'time']] % 1) != 0).any().any()
            or d.id.duplicated().any()):
        raise ValueError('invalid official trade values or IDs')
    d = d.sort_values('id', kind='stable')
    times = pd.to_datetime(d.time, unit='ms', utc=True)
    if not times.dt.normalize().eq(pd.Timestamp(day, tz='UTC')).all():
        raise ValueError('official archive exchange-day mismatch')
    jumps = np.diff(d.id.to_numpy(dtype=np.int64))
    missing = int((jumps[jumps > 1] - 1).sum())
    raw = pd.DataFrame({'id': d.id, 'timestamp_ms': d.time,
                        'price': d.price, 'qty': d.qty,
                        'maker': d.is_buyer_maker})
    events = events_from_trades(raw)
    return events, {'missing_trade_ids': missing, 'trade_rows': len(d),
                    'event_rows': len(events),
                    'first_trade_id': int(d.id.iloc[0]),
                    'last_trade_id': int(d.id.iloc[-1])}


def previous_hour_seed(path: Path, day: str, max_uncompressed_mb=2500) -> tuple[float, float, float]:
    """Stream the preceding archive; retain only 23:00 events for the seed."""
    start = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=23)
    end = start + pd.Timedelta(hours=1)
    retained = []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].file_size > max_uncompressed_mb * 1024**2:
            raise ValueError('unexpected or oversized seed archive')
        with archive.open(members[0]) as stream:
            chunks = pd.read_csv(stream, usecols=['id', 'price', 'qty', 'time', 'is_buyer_maker'],
                                 chunksize=250_000)
            for d in chunks:
                if (d.empty or d.isna().any().any()
                        or not d.is_buyer_maker.isin([True, False]).all()
                        or not np.isfinite(d[['id', 'time', 'price', 'qty']].to_numpy()).all()
                        or (d[['price', 'qty']] <= 0).any().any()
                        or ((d[['id', 'time']] % 1) != 0).any().any()):
                    raise ValueError('invalid official seed archive values')
                times = pd.to_datetime(d.time, unit='ms', utc=True)
                if not times.dt.normalize().eq(pd.Timestamp(day, tz='UTC')).all():
                    raise ValueError('official seed archive exchange-day mismatch')
                block = d.loc[(times >= start) & (times < end)]
                if not block.empty:
                    retained.append(block)
    if not retained:
        raise ValueError('preceding archive has no 23:00 economic trades')
    d = pd.concat(retained, ignore_index=True)
    raw = pd.DataFrame({'id': d.id, 'timestamp_ms': d.time,
                        'price': d.price, 'qty': d.qty,
                        'maker': d.is_buyer_maker})
    return quantiles(events_from_trades(raw))


def compute(root: Path, symbol: str, day: str) -> dict:
    started = time.monotonic()
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise RuntimeError('less than 20GiB free; v3 backfill paused')
    previous = (datetime.fromisoformat(day).date() - timedelta(days=1)).isoformat()
    prior_row = root / 'inputs/daily' / previous / f'{symbol}.json'
    seed = None
    if prior_row.exists():
        info = json.loads(prior_row.read_text())
        if (info.get('formula_version') == FORMULA_VERSION
                and info.get('formula_code_sha256') == FORMULA_CODE_SHA256):
            seed = tuple(info['next_day_00h_quantiles'])
    archive_info = {}
    with tempfile.TemporaryDirectory(prefix='v3-archive-', dir=root / 'archive_work') as temporary:
        work = Path(temporary)
        if seed is None:
            path, checksum, _, _ = download(symbol, previous, work)
            seed = previous_hour_seed(path, previous)
            archive_info['previous_archive_sha256'] = checksum
            path.unlink()
        path, checksum, count, url = download(symbol, day, work)
        events, evidence = read_events(path, day)
        factors = daily_from_events(events, day, seed)
    factors.update(evidence)
    factors.update(archive_info)
    factors.update({'symbol': symbol, 'day': day, 'formula_version': FORMULA_VERSION,
                    'formula_code_sha256': FORMULA_CODE_SHA256,
                    'source_version': SOURCE_VERSION, 'status': 'final',
                    'candidate_only': True, 'source_complete': False,
                    'source_certified': False,
                    'quality_blockers': ['public_source_scope_not_certified'] +
                        (['unresolved_individual_trade_id_gaps'] if evidence['missing_trade_ids'] else []),
                    'archive_sha256': checksum, 'archive_url': url,
                    'archive_bytes': count,
                    'partition_basis': 'exchange_time_utc_day',
                    'available_at': datetime.now(timezone.utc).isoformat(),
                    'job_seconds': round(time.monotonic() - started, 4)})
    from data_center.quality_tolerance import assess
    quality = assess(factors, symbol, day, datetime.now(timezone.utc),
                     expected_formula=FORMULA_VERSION)
    factors['candidate_only'] = not quality['passes_tolerance']
    factors['live_quality_assessment'] = {
        'passes_tolerance': quality['passes_tolerance'],
        'missing_fraction_upper_bound': quality['missing_fraction_upper_bound'],
        'reference_kind': quality.get('reference_kind'),
        'blockers': quality['blockers'],
    }
    output = root / 'inputs/daily' / day / f'{symbol}.json'
    atomic_json(output, factors)
    return factors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--universe', type=Path, required=True)
    parser.add_argument('--day')
    parser.add_argument('--history-days', type=int, default=12)
    parser.add_argument('--max-jobs', type=int, default=8)
    parser.add_argument('--max-seconds', type=int, default=240)
    args = parser.parse_args()
    root = args.root
    (root / 'archive_work').mkdir(parents=True, exist_ok=True)
    state_path = root / 'backfill_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    items = state.get('items', {})
    day = args.day or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    first = state.get('history_start_day') or (
        datetime.fromisoformat(day).date() - timedelta(days=args.history_days-1)).isoformat()
    days_to_scan = (datetime.fromisoformat(day).date()
                    - datetime.fromisoformat(first).date()).days + 1
    if days_to_scan < 1:
        raise ValueError('backfill end precedes pinned history start')
    started = time.monotonic()
    attempted = 0
    try:
        for symbol, target in jobs(json.loads(args.universe.read_text()), day, days_to_scan):
            key = f'{target}/{symbol}'
            output = root / 'inputs/daily' / target / f'{symbol}.json'
            if output.exists():
                old = json.loads(output.read_text())
                if (old.get('formula_version') == FORMULA_VERSION
                        and old.get('formula_code_sha256') == FORMULA_CODE_SHA256):
                    if old.get('candidate_only') is True:
                        from data_center.quality_tolerance import assess
                        quality = assess(old, symbol, target, datetime.now(timezone.utc),
                                         expected_formula=FORMULA_VERSION)
                        if quality['passes_tolerance']:
                            old['candidate_only'] = False
                            old['live_quality_assessment'] = {
                                'passes_tolerance': True,
                                'missing_fraction_upper_bound': quality['missing_fraction_upper_bound'],
                                'reference_kind': quality.get('reference_kind'),
                                'blockers': [],
                            }
                            atomic_json(output, old)
                            items[key] = {**items.get(key, {}), 'status': 'promoted_final'}
                        else:
                            items[key] = {**items.get(key, {}), 'status': 'quality_blocked',
                                          'blockers': quality['blockers']}
                    continue
            if items.get(key, {}).get('retry_after_epoch', 0) > time.time():
                continue
            if attempted >= args.max_jobs or time.monotonic() - started >= args.max_seconds:
                break
            attempted += 1
            try:
                row = compute(root, symbol, target)
                items[key] = {'status': 'computed_final' if not row['candidate_only'] else 'quality_blocked',
                              'job_seconds': row['job_seconds'],
                              'missing_trade_ids': row['missing_trade_ids'],
                              'archive_bytes': row['archive_bytes']}
            except HTTPError as exc:
                items[key] = {'status': 'archive_not_published' if exc.code == 404 else 'download_error',
                              'http_status': exc.code,
                              'retry_after_epoch': time.time() + 3600}
                if exc.code in (418, 429):
                    break
            except Exception as exc:
                items[key] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                              'retry_after_epoch': time.time() + 3600}
            print(json.dumps({'job': key, **items[key]}), flush=True)
    finally:
        atomic_json(state_path, {'formula_version': FORMULA_VERSION,
                                 'history_start_day': first,
                                 'updated_at': datetime.now(timezone.utc).isoformat(),
                                 'attempted_this_run': attempted,
                                 'statuses': dict(Counter(v['status'] for v in items.values())),
                                 'items': items, 'submitted_to_exchange': False})


if __name__ == '__main__':
    main()
