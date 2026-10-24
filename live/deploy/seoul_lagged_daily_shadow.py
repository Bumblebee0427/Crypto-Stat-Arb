#!/usr/bin/env python3
"""Seal audited lagged-quantile daily rows from warm minutes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from data_center.contract import atomic_json, digest
from data_center.lagged_quantile_v3 import (
    FORMULA_CODE_SHA256, FORMULA_VERSION, add_components,
    events_from_trades, minute_components, minute_frame, quantiles,
)
from data_center.reference_formula_v2 import daily_factors
from deploy.seoul_lagged_minute_shadow import CODE_SHA256 as MINUTE_CODE_SHA256
from deploy.seoul_lagged_fast_metadata import metadata as fast_metadata
from deploy.seoul_lagged_daily_prefix import (finish_symbol, load as load_prefix,
                                             read_minute)


from data_center.paths import BASE as _PORTABLE_BASE

BASE = _PORTABLE_BASE
ROOT = BASE / 'data/lagged_quantile_v3'
STAGED = BASE / 'data/captured_daily_stream_v1/work'
MAIN = 'big_ultra_harmonic_rm240_mean_1d'
SMALL = 'small_qty_misalignment_rankcorr_lag5_1d'
COLS = ('id', 'timestamp_ms', 'price', 'qty', 'maker')
# Bound the largest pandas allocation before parsing. A symbol exceeding the
# replay budget is excluded through the existing qualified-pool policy.
MAX_REPLAY_HOUR_BYTES = 64 * 1024 * 1024


def read_hour(path: Path) -> pd.DataFrame:
    before = path.stat()
    if before.st_size > MAX_REPLAY_HOUR_BYTES:
        raise ValueError('hourly replay exceeds memory budget: '+str(path))
    result = pd.read_csv(path, header=None, names=COLS,
                       dtype={'id': 'int64', 'timestamp_ms': 'int64',
                              'price': 'float64', 'qty': 'float64', 'maker': 'bool'})
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('hourly source changed during replay: '+str(path))
    return result


def replay_hour(path: Path, start: pd.Timestamp):
    before = path.stat()
    if before.st_size > MAX_REPLAY_HOUR_BYTES:
        raise ValueError('hourly replay exceeds memory budget: '+str(path))
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    raw = read_hour(path)
    lo = start.value // 1_000_000
    hi = (start+pd.Timedelta(hours=1)).value // 1_000_000
    if raw.empty or not raw.timestamp_ms.between(lo, hi-1).all():
        raise ValueError('trade outside replay UTC hour: '+str(path))
    events = events_from_trades(raw)
    unique = raw.drop_duplicates('id')
    economic_count = int((~(unique.price.eq(0) & unique.qty.eq(0))).sum())
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('hourly source changed during replay: '+str(path))
    return events, economic_count, int(unique.id.min()), int(unique.id.max()), sha


def replay_symbol(day: str, symbol: str, trade_rows: int) -> tuple[dict, dict]:
    """Rebuild only a revised symbol from the captured complete UTC day."""
    prior = (datetime.fromisoformat(day).date() - timedelta(days=1)).isoformat()
    previous_source = STAGED / prior / 'hours' / '23' / f'{symbol}.csv'
    start = pd.Timestamp(day, tz='UTC')
    seed_events, _, _, _, seed_sha = replay_hour(previous_source, start-pd.Timedelta(hours=1))
    seed = quantiles(seed_events)
    del seed_events
    current = seed
    sources = {}
    parts = {}
    economic_count = 0
    previous_last_id = None
    first_minute = None
    for hour in range(24):
        path = STAGED / day / 'hours' / f'{hour:02d}' / f'{symbol}.csv'
        events, count, first_id, last_id, sha = replay_hour(path, start+pd.Timedelta(hours=hour))
        if previous_last_id is not None and first_id <= previous_last_id:
            raise ValueError('overlapping UTC hour IDs: '+symbol)
        if events.empty:
            raise ValueError(f'no economic trades in replay UTC hour {hour:02d}')
        previous_last_id = last_id
        economic_count += count
        sources[f'{hour:02d}'] = sha
        if first_minute is None:
            first_minute = events.time.min().floor('min')
        add_components(parts, minute_components(events, current, first_minute=first_minute))
        current = quantiles(events)
        del events
    if economic_count != trade_rows:
        raise ValueError(f'replay trade count differs from baseline: {symbol} '
                         f'{economic_count}/{trade_rows}')
    factors = daily_factors(minute_frame(parts, day), day)
    factors.update(formula_version=FORMULA_VERSION, prior_day_23h_quantiles=list(seed),
                   next_day_00h_quantiles=list(current))
    evidence = {'kind': 'replayed_complete_hourly_stage',
                'processing': 'one_utc_hour_at_a_time',
                'maximum_hour_bytes': MAX_REPLAY_HOUR_BYTES,
                'formula_code_sha256': FORMULA_CODE_SHA256,
                'previous_hour_seed': list(seed),
                'previous_hour_source_sha256': seed_sha,
                'hourly_source_sha256': sources,
                'trade_rows': economic_count}
    return factors, evidence


def emit(day: str, prepared_metadata: dict | None = None) -> dict:
    started = time.monotonic()
    phase_started = started
    timings = {}
    def phase(name):
        nonlocal phase_started
        current = time.monotonic()
        timings[name] = round(current-phase_started,3)
        phase_started = current
        print(json.dumps({'day':day,'stage':name,'seconds':timings[name]}), flush=True)
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) < start + timedelta(days=1):
        raise ValueError('UTC day boundary is not sealed yet')
    fast_rows = prepared_metadata if prepared_metadata is not None else fast_metadata(day, exclude_invalid=True)
    excluded_symbols = dict(getattr(fast_rows, "excluded_symbols", {}))
    symbols = sorted(fast_rows)
    baseline_kind = 'audited_hourly_id_metadata'
    if not symbols:
        raise ValueError('empty v3 metadata universe')
    current_stage = json.loads((STAGED/day/'stage_state.json').read_text())
    if digest(current_stage['processed']) != fast_rows[symbols[0]]['input_manifest_hash']:
        raise ValueError('Seoul staged source changed after metadata audit')
    phase('baseline_metadata_seconds')
    try:
        prefix = load_prefix(day)
        print(json.dumps({'day': day, 'daily_prefix': 'ready',
                          'minutes': len(prefix['minute_hashes']),
                          'symbols': len(prefix['symbols'])}), flush=True)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        prefix = None
        print(json.dumps({'day': day, 'daily_prefix': 'full_day_fallback',
                          'reason': type(exc).__name__ + ': ' + str(exc)[:140]}), flush=True)
    if prefix:excluded_symbols.update(prefix.get('excluded_symbols', {}))
    phase('prefix_load_seconds')
    by_symbol = {symbol: {} for symbol in symbols}
    trade_counts = {symbol: prefix['trade_counts'].get(symbol, 0) if prefix else 0
                    for symbol in symbols}
    threshold_hashes = dict(prefix['threshold_hashes']) if prefix else {}
    unchanged_quantile_revisions = []
    replay_symbols = {}
    minute_hashes = dict(prefix['minute_hashes']) if prefix else {}
    first_minute = prefix['last_minute'] + 1 if prefix else 0
    for n in range(first_minute, 1440):
        row, sha, _ = read_minute(day, n)
        excluded_symbols.update(row.get("excluded_symbols", {}))
        minute_hashes[f'{n // 60:02d}{n % 60:02d}.json'] = sha
        for symbol, components in row['components'].items():
            if symbol in by_symbol:
                by_symbol[symbol][row['minute']] = components
        for symbol, count in row.get('trade_counts', {}).items():
            if symbol in trade_counts:
                trade_counts[symbol] += count
        for key, source_sha in row.get('threshold_source_sha256', {}).items():
            if key.rsplit('/',1)[-1] not in by_symbol:continue
            old_sha = threshold_hashes.setdefault(key, source_sha)
            if old_sha != source_sha:
                excluded_symbols[key.rsplit('/', 1)[-1]]=['threshold_revised_during_utc_hour:'+key]
    phase('tail_minutes_seconds')
    for key, source_sha in threshold_hashes.items():
        threshold_day, hour, symbol = key.split('/')
        if symbol not in by_symbol:continue
        if symbol in excluded_symbols:continue
        try:
            cached = json.loads((ROOT / 'thresholds' / threshold_day / hour / f'{symbol}.json').read_text())
            source = Path(cached['source'])
            info = source.stat()
            if cached['source_sha256'] != source_sha:
                raise ValueError('threshold changed inside its active UTC hour: '+key)
            if cached['source_size'] != info.st_size or cached['source_mtime_ns'] != info.st_mtime_ns:
                revised = quantiles(events_from_trades(read_hour(source)))
                if revised != tuple(cached['quantiles']):
                    replay_symbols.setdefault(symbol, []).append('revised_quantile:'+key)
                else:
                    unchanged_quantile_revisions.append(key)
        except (OSError, ValueError, KeyError, TypeError, MemoryError) as exc:
            excluded_symbols[symbol]=[type(exc).__name__ + ': ' + str(exc)]
    phase('threshold_revision_checks_seconds')
    manifest = digest(minute_hashes)
    rows = {}
    prefix_symbols = 0
    for symbol in symbols:
        if symbol in excluded_symbols:continue
        try:
            old = dict(fast_rows[symbol])
            baseline_sha = digest(old)
            if trade_counts[symbol] != old['trade_rows']:
                replay_symbols.setdefault(symbol, []).append('minute_trade_count_mismatch')
            if symbol not in replay_symbols:
                used_prefix = bool(prefix and symbol in prefix['symbols'])
                if used_prefix:
                    factors = finish_symbol(prefix['symbols'][symbol], by_symbol[symbol], day)
                else:
                    factors = daily_factors(minute_frame(by_symbol[symbol], day), day)
                if old.get('minute_count') is not None and factors['minute_count'] != old['minute_count']:
                    replay_symbols.setdefault(symbol, []).append('minute_count_mismatch')
                if used_prefix and symbol not in replay_symbols:
                    prefix_symbols += 1
            if symbol in replay_symbols:
                replay_started=time.monotonic()
                print(json.dumps({'day':day,'stage':'symbol_replay_started','symbol':symbol,
                                  'reasons':replay_symbols[symbol]}), flush=True)
                factors, replay_evidence = replay_symbol(day, symbol, old['trade_rows'])
                timings['raw_replay_seconds']=round(timings.get('raw_replay_seconds',0)
                                                    +time.monotonic()-replay_started,3)
                print(json.dumps({'day':day,'stage':'symbol_replay_finished','symbol':symbol,
                                  'seconds':round(time.monotonic()-replay_started,3)}), flush=True)
                if old.get('minute_count') is not None and factors['minute_count'] != old['minute_count']:
                    raise ValueError('replayed minute coverage differs from baseline: '+symbol)
            else:
                replay_evidence = None
            old.update({MAIN: factors[MAIN], SMALL: factors[SMALL],
                        'minute_count': factors['minute_count'],
                        'boundary_minute_count': factors['boundary_minute_count'],
                        'formula_version': FORMULA_VERSION,
                        'formula_code_sha256': FORMULA_CODE_SHA256,
                        'status': 'final',
                        'producer': 'seoul_lagged_minute_v3_replay' if replay_evidence else 'seoul_lagged_minute_v3',
                        'candidate_only': False,
                        'provisional_minute_trade_rows': trade_counts[symbol],
                        'factor_input_trade_rows': old['trade_rows'],
                        'minute_manifest_hash': manifest,
                        'replay_reasons': replay_symbols.get(symbol, []),
                        'replay_evidence': replay_evidence,
                        'baseline_input_sha256': baseline_sha,
                        'baseline_evidence_kind': baseline_kind,
                        'available_at': datetime.now(timezone.utc).isoformat()})
            rows[symbol] = old
        except (OSError, ValueError, KeyError, TypeError, MemoryError) as exc:
            excluded_symbols[symbol]=[type(exc).__name__ + ': ' + str(exc)]
    symbols=sorted(rows)
    if not symbols:raise ValueError('no valid v3 daily symbols after exclusions')
    phase('symbol_factors_and_replays_seconds')
    output = ROOT / 'inputs/live_daily' / day
    output.mkdir(parents=True, exist_ok=True)
    for symbol, row in rows.items():
        atomic_json(output / f'{symbol}.json', row)
    phase('daily_rows_write_seconds')
    timings['total_before_marker_seconds']=round(time.monotonic()-started,3)
    atomic_json(output / 'complete.json', {
        'day': day, 'status': 'complete', 'formula_version': FORMULA_VERSION,
        'formula_code_sha256': FORMULA_CODE_SHA256,
        'minute_code_sha256': MINUTE_CODE_SHA256,
        'symbols': symbols, 'excluded_symbols': excluded_symbols, 'minute_manifest_hash': manifest,
        'baseline_evidence_kind': baseline_kind,
        'source_revisions_with_unchanged_quantiles': unchanged_quantile_revisions,
        'replayed_symbols': replay_symbols,
        'daily_prefix_symbols': prefix_symbols,
        'daily_prefix_minutes': first_minute,
        'timings': timings,
        'files_sha256': {symbol: hashlib.sha256((output/f'{symbol}.json').read_bytes()).hexdigest()
                         for symbol in symbols},
        'finished_at': datetime.now(timezone.utc).isoformat()})
    return {'day': day, 'status': 'complete', 'symbols': len(symbols), 'excluded_symbols': excluded_symbols}


if __name__ == '__main__':
    day = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    print(json.dumps(emit(day)), flush=True)
