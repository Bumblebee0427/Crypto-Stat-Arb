#!/usr/bin/env python3
"""Seal source and ID evidence without waiting for v2 same-hour factors."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from data_center.base_strategy import SOURCE_VERSION, symbol_name
from data_center.captured_daily import edge
from data_center.contract import digest, timestamp
from data_center.hourly_trade_metadata import CODE_SHA256 as ID_METADATA_CODE_SHA256, summarize_hour
from data_center.quality_tolerance import anchor_id
from data_center.lagged_universe import split as split_universe


from data_center.paths import BASE as _PORTABLE_BASE

BASE = _PORTABLE_BASE
STAGED = BASE / 'data/captured_daily_stream_v1/work'
UNIVERSE = BASE / 'data/live_factor_storage/_state/universe.json'
COLS = ('id', 'timestamp_ms', 'price', 'qty', 'maker')


def closing_hour(source: Path, day: str) -> dict:
    return summarize_hour(source, day, 23)


def readonly_hour(work: Path, day: str, hour: int, symbol: str) -> dict:
    """Read a current ID cache or calculate in memory; never attempt a write."""
    source=work/'hours'/f'{hour:02d}'/(symbol+'.csv')
    cache=work/'id_hours'/f'{hour:02d}'/(symbol+'.json')
    info=source.stat()
    try:
        row=json.loads(cache.read_text())
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        row={}
    if (row.get('day')==day and row.get('hour')==hour
            and row.get('metadata_code_sha256')==ID_METADATA_CODE_SHA256
            and row.get('source_size')==info.st_size
            and row.get('source_mtime_ns')==info.st_mtime_ns):
        return row
    return summarize_hour(source,day,hour)


class MetadataRows(dict):
    def __init__(self, rows, excluded_symbols):
        super().__init__(rows)
        self.excluded_symbols = excluded_symbols


def metadata(day: str, symbols: set[str] | None = None, *, exclude_invalid=False) -> dict[str, dict]:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    work = STAGED / day
    state_path = work / 'stage_state.json'
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    if (state.get('day') != day or state.get('layout') != 'hourly'
            or state.get('invalid_target_records') or state.get('unsealed_paths')
            or state.get('capture_status') != 'running'
            or timestamp(state['capture_started_at']) > start
            or timestamp(state['capture_checkpoint_at']) < end):
        raise ValueError('Seoul stage has not closed the UTC day')
    prior = (start - timedelta(days=1)).date().isoformat()
    following = end.date().isoformat()
    if symbols is None:
        symbols, _ = split_universe(json.loads(UNIVERSE.read_text()), day, work)
    symbols = sorted({symbol_name(s) for s in symbols} - {'BTCUSDT', 'ETHUSDT'})
    # The collector checkpoints fsynced raw trades every ten seconds. Its
    # first next-day trade can arrive after the minute-stage pass at second
    # ten; read that immutable checkpoint directly instead of waiting for the
    # next stage pass at second ten of the following minute.
    capture_state = json.loads((BASE / 'data/individual_trades_v2/state.json').read_text())
    manifest = digest(state['processed'])
    output = {}
    excluded = {}
    for symbol in symbols:
        try:
            before = edge(state['previous'], symbol, prior)
            after = edge(state['following'], symbol, following)
            if after is None and capture_state.get('status') == 'running':
                first = capture_state.get('symbols', {}).get(symbol, {})
                if first.get('first_trade_day') == following:
                    candidate = {'kind': 'checkpointed_neighbor_trade',
                                 'symbol': symbol, 'day': following,
                                 'id': first.get('first_trade_id'),
                                 'timestamp_ms': first.get('first_trade_timestamp_ms'),
                                 'raw_record_sha256': first.get('first_trade_record_sha256'),
                                 'capture_checkpoint_at': capture_state.get('updated_at'),
                                 'capture_status': capture_state.get('status')}
                    if anchor_id(candidate, symbol, following, 'next') is not None:
                        after = candidate
            if before is None:
                raise ValueError('missing previous-day ID anchor: '+symbol)
            hours = [readonly_hour(work,day,hour,symbol) for hour in range(24)]
            sequence = trades = zeros = missing = 0
            monotonic = True
            previous_hour = None
            for row in hours:
                if previous_hour is not None:
                    if row['first_trade_id'] <= previous_hour['last_trade_id']:
                        raise ValueError('overlapping UTC hour IDs: '+symbol)
                    missing += row['first_trade_id'] - previous_hour['last_trade_id'] - 1
                    monotonic &= previous_hour['last_id_time_ms'] <= row['first_id_time_ms']
                sequence += row['sequence_rows']
                trades += row['trade_rows']
                zeros += row['zero_value_sequence_rows']
                missing += row['missing_trade_ids']
                monotonic &= row['time_monotonic_by_id']
                previous_hour = row
            if (sequence != trades + zeros or
                    hours[-1]['last_trade_id'] - hours[0]['first_trade_id'] + 1 != sequence + missing):
                raise ValueError('inconsistent whole-day sequence: '+symbol)
            output[symbol] = {'symbol': symbol, 'day': day, 'source_version': SOURCE_VERSION,
                              'status': 'final', 'source_complete': False,
                              'source_certified': False,
                              'quality_blockers': ['public_source_scope_not_certified'] +
                                  (['unresolved_individual_trade_id_gaps'] if missing else []),
                              'missing_trade_ids': missing, 'trade_rows': trades,
                              'sequence_rows': sequence, 'zero_value_sequence_rows': zeros,
                              'first_trade_id': hours[0]['first_trade_id'],
                              'last_trade_id': hours[-1]['last_trade_id'],
                              'first_exchange_time_ms': min(r['first_exchange_time_ms'] for r in hours),
                              'last_exchange_time_ms': max(r['last_exchange_time_ms'] for r in hours),
                              'time_monotonic_by_id': bool(monotonic),
                              'partition_basis': 'exchange_time_utc_day',
                              'available_at': datetime.now(timezone.utc).isoformat(),
                              'producer': 'seoul_lagged_fast_metadata_v1',
                              'input_manifest_hash': manifest,
                              'day_boundary_evidence': {'previous': before, 'next': after}}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not exclude_invalid:raise
            excluded[symbol] = [type(exc).__name__ + ': ' + str(exc)]
    if state_path.read_bytes() != state_bytes:
        raise ValueError('Seoul stage changed during metadata assembly')
    return MetadataRows(output, excluded) if exclude_invalid else output


if __name__ == '__main__':
    import sys
    result = metadata(sys.argv[1])
    print(json.dumps({'day': sys.argv[1], 'symbols': len(result)}))
