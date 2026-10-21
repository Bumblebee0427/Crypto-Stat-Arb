"""Convert retained individual @trade segments into bounded UTC daily inputs.

The public stream is not source certified. Every segment hash, trade ID and
adjacent-day edge remains available for the TOP50 quality assessment.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from data_center.base_strategy import symbol_name
from data_center.contract import atomic_json, digest, timestamp
from data_center.live_daily import compute_daily


def edge(rows, symbol, day):
    row = rows.get(symbol)
    if row is None:
        return None
    if row.get('kind') == 'checkpointed_neighbor_trade':
        return row if row.get('symbol') == symbol and row.get('day') == day else None
    return {'kind': 'observed_neighbor_trade', 'symbol': symbol, 'day': day,
            'id': row['id'], 'timestamp_ms': row['timestamp_ms'],
            'raw_segment_sha256': row['raw_segment_sha256']}


def stage(capture, work, day, layout='daily', previous_seed=None, max_segments=None):
    capture, work = Path(capture), Path(work)
    if layout not in ('daily', 'hourly'):
        raise ValueError('unknown stage layout')
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / 'stage_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and state.get('day') != day:
        raise ValueError('stage directory belongs to another UTC day')
    if state and state.get('layout', 'daily') != layout:
        raise ValueError('stage layout changed')
    processed = dict(state.get('processed', {}))
    seeded = bool(state.get('previous_seeded')) or bool(not state and previous_seed)
    previous = dict(state.get('previous', previous_seed or {}))
    following = dict(state.get('following', {}))
    target_last = dict(state.get('target_last', {}))
    bad = int(state.get('invalid_target_records', 0))
    target = date.fromisoformat(day)
    prev_day, next_day = target-timedelta(days=1), target+timedelta(days=1)
    current_bucket = int(time.time() // 300)
    if layout == 'hourly':
        # Only the first next-day receive bucket can contain relevant delayed
        # previous-day trades. Its short boundary shard is sealed by the
        # collector's fsynced checkpoint inside the first trading minute.
        days = (target, next_day) if seeded else (prev_day, target, next_day)
    else:
        days = (target, next_day) if seeded else (prev_day, target, next_day)
    seal_state = json.loads((capture/'state.json').read_text())
    sealed = (set(seal_state.get('boundary_sealed_paths', []))
              if seal_state.get('boundary_sealed_day') == next_day.isoformat() else set())
    minute_sealed = set(seal_state.get('minute_sealed_paths', []))
    first_next_bucket = int(datetime.combine(next_day, datetime.min.time(), timezone.utc).timestamp() // 300)
    paths = set()
    for d in days:
        for path in (capture/'raw'/d.isoformat()).glob('*.jsonl.gz'):
            bucket = int(path.name.split('-', 1)[0])
            if layout == 'hourly' and d == next_day:
                if bucket != first_next_bucket:
                    continue
                if (bucket < current_bucket or str(path.relative_to(capture)) in sealed
                        or str(path.relative_to(capture)) in minute_sealed):
                    paths.add(path)
            elif bucket < current_bucket or str(path.relative_to(capture)) in minute_sealed:
                paths.add(path)
    # Archive repairs are written under their receipt day, which can be many
    # days after the exchange day. Include their sealed shards on a restage.
    paths.update(p for p in (capture/'raw').glob('*/*-repair.jsonl.gz')
                 if int(p.name.split('-', 1)[0]) < current_bucket)
    paths = sorted(paths)
    streams = {}
    scanned = 0
    unsealed = []
    try:
        for path in paths:
            name = str(path.relative_to(capture))
            if name in processed:
                continue
            with path.open('rb') as source:
                sha = hashlib.file_digest(source, 'sha256').hexdigest()
            try:
                with gzip.open(path, 'rt') as source:
                    for line in source:
                        record = json.loads(line)
                        d = record.get('data', {})
                        try:
                            symbol, ident, ms = d['s'], int(d['t']), int(d['T'])
                            event_day = datetime.fromtimestamp(ms/1000, timezone.utc).date()
                        except (KeyError, TypeError, ValueError, OverflowError):
                            bad += 1
                            continue
                        if event_day == target:
                            if record.get('validation_error'):
                                bad += 1
                                continue
                            key = (symbol, datetime.fromtimestamp(ms/1000, timezone.utc).hour) if layout == 'hourly' else symbol
                            if key not in streams:
                                if layout == 'hourly' and len(streams) >= 96:
                                    oldest = next(iter(streams))
                                    streams.pop(oldest).close()
                                dest = work/'hours'/f'{key[1]:02d}'/f'{symbol}.csv' if layout == 'hourly' else work/f'{symbol}.csv'
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                streams[key] = dest.open('a', newline='')
                            csv.writer(streams[key]).writerow(
                                (ident, ms, d['p'], d['q'], d['m']))
                            old = target_last.get(symbol)
                            if old is None or ident >= old['id']:
                                target_last[symbol] = {'id': ident, 'timestamp_ms': ms,
                                                       'raw_segment_sha256': sha}
                        elif event_day == prev_day or event_day == next_day:
                            if record.get('validation_error'):
                                continue
                            book = previous if event_day == prev_day else following
                            old = book.get(symbol)
                            if old is None or (ident >= old['id'] if event_day == prev_day else ident <= old['id']):
                                book[symbol] = {'id': ident, 'timestamp_ms': ms,
                                                'raw_segment_sha256': sha}
            except (EOFError, OSError):
                # A prior five-minute bucket may remain open until its shard
                # receives the first message of the new bucket. Retry it later.
                unsealed.append(name)
                continue
            processed[name] = sha
            scanned += 1
            if max_segments is not None and scanned >= max_segments:
                break
    finally:
        for stream in streams.values():
            stream.close()
    capture_state = json.loads((capture/'state.json').read_text())
    next_day = next_day.isoformat()
    for symbol, row in capture_state.get('symbols', {}).items():
        if row.get('first_trade_day') != next_day or symbol in following:
            continue
        ident, ms, sha = (row.get('first_trade_id'),
                          row.get('first_trade_timestamp_ms'),
                          row.get('first_trade_record_sha256'))
        if isinstance(ident, int) and isinstance(ms, int) and isinstance(sha, str):
            following[symbol] = {
                'kind': 'checkpointed_neighbor_trade', 'symbol': symbol,
                'day': next_day, 'id': ident, 'timestamp_ms': ms,
                'raw_record_sha256': sha,
                'capture_checkpoint_at': capture_state.get('updated_at'),
                'capture_status': capture_state.get('status')}
    pending=[p for p in paths if str(p.relative_to(capture)) not in processed]
    # Inputs were selected from the start-of-pass collector checkpoint.
    # A long catch-up must not claim trades received while it was running.
    checkpoint=timestamp(seal_state['updated_at'])
    if pending:
        first_pending=min(int(p.name.split('-',1)[0])*300 for p in pending)
        checkpoint=min(checkpoint,datetime.fromtimestamp(first_pending,timezone.utc))
    atomic_json(state_path, {'day': day, 'layout': layout, 'previous_seeded': seeded,
                             'processed': processed,
                             'previous': previous, 'target_last': target_last,
                             'following': following, 'invalid_target_records': bad,
                             'capture_started_at': capture_state.get('started_at'),
                             'capture_checkpoint_at': checkpoint.isoformat(),
                             'capture_observed_at':capture_state.get('updated_at'),
                             'pending_segments':len(pending),
                             'capture_status': capture_state.get('status'),
                             'unsealed_paths': unsealed,
                             'updated_at': datetime.now(timezone.utc).isoformat()})
    symbol_paths = work.glob('hours/*/*.csv') if layout == 'hourly' else work.glob('*.csv')
    return {'scanned_segments': scanned, 'total_segments': len(processed),
            'symbols_with_rows': len({p.stem for p in symbol_paths}),
            'previous_edges': len(previous), 'next_edges': len(following),
            'invalid_target_records': bad, 'unsealed_segments': len(unsealed),
            'pending_segments':len(pending)}


def emit(work, output, universe, day):
    work, output = Path(work), Path(output)
    if datetime.now(timezone.utc) < datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(days=1, minutes=5):
        raise ValueError('next UTC day boundary has not closed')
    state = json.loads((work/'stage_state.json').read_text())
    if state['day'] != day or state['invalid_target_records'] or state.get('unsealed_paths'):
        raise ValueError('wrong UTC day, invalid records, or unsealed raw segments')
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    if (state.get('capture_status') != 'running'
            or timestamp(state['capture_started_at']) > start
            or timestamp(state['capture_checkpoint_at']) < end + timedelta(minutes=5)):
        raise ValueError('capture was not running across the complete UTC day')
    symbols = sorted({symbol_name(s) for s in json.loads(Path(universe).read_text())['tradable_symbols']}
                     - {'BTCUSDT', 'ETHUSDT'})
    prev_day = (date.fromisoformat(day)-timedelta(days=1)).isoformat()
    next_day = (date.fromisoformat(day)+timedelta(days=1)).isoformat()
    manifest = digest(state['processed'])
    output.mkdir(parents=True, exist_ok=True)
    written, missing_edges = 0, []
    for symbol in symbols:
        source = work/f'{symbol}.csv'
        before = edge(state['previous'], symbol, prev_day)
        after = edge(state['following'], symbol, next_day)
        if not source.exists() or before is None or after is None:
            missing_edges.append(symbol)
            continue
        raw = pd.read_csv(source, header=None,
                          names=['id', 'timestamp_ms', 'price', 'qty', 'maker'],
                          dtype={'id': 'int64', 'timestamp_ms': 'int64',
                                 'price': 'float64', 'qty': 'float64', 'maker': 'bool'})
        row = compute_daily(raw, symbol, day, manifest)
        row['day_boundary_evidence'] = {'previous': before, 'next': after}
        atomic_json(output/f'{symbol}.json', row)
        written += 1
    return {'expected': len(symbols), 'written': written,
            'missing_rows_or_edges': missing_edges}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--day', required=True)
    p.add_argument('--emit-to', type=Path)
    p.add_argument('--universe', type=Path)
    a = p.parse_args()
    a.work.parent.mkdir(parents=True, exist_ok=True)
    with (a.work.parent/f'.{a.day}.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(stage(a.capture, a.work, a.day)), flush=True)
        if a.emit_to:
            if not a.universe:
                p.error('--emit-to requires --universe')
            print(json.dumps(emit(a.work, a.emit_to, a.universe, a.day)), flush=True)


if __name__ == '__main__':
    main()
