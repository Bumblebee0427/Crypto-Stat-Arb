#!/usr/bin/env python3
"""Read sealed Seoul trade shards and warm no-lookahead minute fields.

This writes only to the isolated v3 candidate tree. Production signals and
orders continue to use the existing formula until history is rebuilt.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from data_center.base_strategy import symbol_name
from data_center.contract import atomic_json
from data_center.lagged_quantile_v3 import FORMULA_VERSION, events_from_trades, quantiles


from data_center.paths import BASE as _PORTABLE_BASE

BASE = _PORTABLE_BASE
CAPTURE = BASE / 'data/individual_trades_v2'
STAGED = BASE / 'data/captured_daily_stream_v1/work'
ROOT = BASE / 'data/lagged_quantile_v3'
UNIVERSE = BASE / 'data/live_factor_storage/_state/universe.json'
COLS = ('id', 'timestamp_ms', 'price', 'qty', 'maker')
MIN_FREE = 20 * 1024**3
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def threshold_source(day: str, hour: int, symbol: str) -> Path:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(hours=hour)
    previous = start - timedelta(hours=1)
    return STAGED / previous.date().isoformat() / 'hours' / f'{previous.hour:02d}' / f'{symbol}.csv'


def threshold(day: str, hour: int, symbol: str) -> tuple[tuple[float, float, float], str]:
    source = threshold_source(day, hour, symbol)
    info = source.stat()
    output = ROOT / 'thresholds' / day / f'{hour:02d}' / f'{symbol}.json'
    if output.exists():
        row = json.loads(output.read_text())
        if row.get('formula_version') == FORMULA_VERSION:
            # The threshold is frozen for this UTC hour. A late repair to the
            # source hour is found by the daily fingerprint audit; it must
            # never change classifications halfway through the active hour.
            return tuple(row['quantiles']), row['source_sha256']
    raw = pd.read_csv(source, header=None, names=COLS,
                      dtype={'id': 'int64', 'timestamp_ms': 'int64',
                             'price': 'float64', 'qty': 'float64', 'maker': 'bool'})
    events = events_from_trades(raw)
    prior = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(hours=hour-1)
    if (events.empty or (events.time < pd.Timestamp(prior)).any()
            or (events.time >= pd.Timestamp(prior + timedelta(hours=1))).any()):
        raise ValueError('prior-hour stage is absent or contaminated')
    value = quantiles(events)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    atomic_json(output, {'day': day, 'hour': hour, 'symbol': symbol,
                         'formula_version': FORMULA_VERSION,
                         'quantiles': list(value), 'source': str(source),
                         'source_size': info.st_size,
                         'source_mtime_ns': info.st_mtime_ns,
                         'source_sha256': source_sha,
                         'event_rows': len(events),
                         'computed_at': datetime.now(timezone.utc).isoformat()})
    return value, source_sha


def sealed_paths(earliest: datetime, now: datetime) -> list[Path]:
    state = json.loads((CAPTURE / 'state.json').read_text())
    current_bucket = int(now.timestamp() // 300)
    sealed = set(state.get('minute_sealed_paths', []))
    first_bucket = int(earliest.timestamp() // 300)
    result = []
    for day in {earliest.date().isoformat(), now.date().isoformat()}:
        for path in (CAPTURE / 'raw' / day).glob('*.jsonl.gz'):
            try:
                bucket = int(path.name.split('-', 1)[0])
            except ValueError:
                continue
            if (first_bucket <= bucket <= current_bucket
                    and (bucket < current_bucket or str(path.relative_to(CAPTURE)) in sealed)):
                result.append(path)
    return sorted(result)


def recent_rows(paths: list[Path], targets: set[datetime]) -> tuple[dict, int]:
    # Keep a bounded receive-time lookback. Late exchange-time rows are
    # re-read on the following two minute runs; older repairs require a
    # deliberate replay and cannot silently change an issued candidate.
    rows: dict[tuple[datetime, str], dict[int, tuple]] = {}
    invalid = 0
    for path in paths:
        with gzip.open(path, 'rt') as source:
            for line in source:
                record = json.loads(line)
                trade = record.get('data', {})
                if record.get('validation_error'):
                    invalid += 1
                    continue
                try:
                    symbol = trade['s']
                    ident, ms = int(trade['t']), int(trade['T'])
                    price, qty = float(trade['p']), float(trade['q'])
                    maker = trade['m']
                    instant = datetime.fromtimestamp(ms / 1000, timezone.utc)
                except (KeyError, ValueError, TypeError, OverflowError):
                    invalid += 1
                    continue
                minute = instant.replace(second=0, microsecond=0)
                destinations = [minute]
                if ms % 60000 == 0:
                    before = minute - timedelta(minutes=1)
                    if before.date() == minute.date():
                        destinations.append(before)
                item = (ident, ms, price, qty, maker)
                for target in destinations:
                    if target in targets:
                        old = rows.setdefault((target, symbol), {}).setdefault(ident, item)
                        if old != item:
                            raise ValueError('conflicting duplicate trade ID in raw shards')
    return rows, invalid


def produce(now: datetime) -> dict:
    if shutil.disk_usage(ROOT.parent).free < MIN_FREE:
        raise RuntimeError('less than 20GiB free; v3 shadow paused')
    minute = now.replace(second=0, microsecond=0)
    targets = {minute - timedelta(minutes=n) for n in (1, 2, 3)}
    paths = sealed_paths(min(targets), now)
    rows, invalid = recent_rows(paths, targets)
    universe = ({symbol_name(s) for s in json.loads(UNIVERSE.read_text())['tradable_symbols']}
                - {'BTCUSDT', 'ETHUSDT'})
    trades = [(target, symbol, *trade)
              for (target, symbol), unique in rows.items() if symbol in universe
              for trade in unique.values()]
    frame = pd.DataFrame(trades, columns=('target', 'symbol', *COLS))
    components: dict[datetime, dict] = {target: {} for target in targets}
    counts: dict[datetime, dict] = {target: {} for target in targets}
    fingerprints: dict[datetime, dict] = {target: {} for target in targets}
    unavailable = []
    if not frame.empty:
        zero = frame.price.eq(0) & frame.qty.eq(0)
        if (((frame.price <= 0) | (frame.qty <= 0)) & ~zero).any():
            raise ValueError('invalid economic trade in sealed raw shard')
        actual_minute = pd.to_datetime(frame.timestamp_ms, unit='ms', utc=True).dt.floor('min')
        current = frame.loc[~zero & actual_minute.eq(frame.target)]
        for (target, symbol), count in current.groupby(['target', 'symbol']).size().items():
            target = target.to_pydatetime()
            counts[target][symbol] = int(count)
        frame = frame.loc[~zero].copy()
    if not frame.empty:
        frame['notional'] = frame.price * frame.qty
        events = frame.groupby(['target', 'symbol', 'timestamp_ms', 'maker'], sort=True).agg(
            qty=('qty', 'sum'), notional=('notional', 'sum')).reset_index()
        events['hour_ms'] = (events.timestamp_ms // 3600000) * 3600000
        pairs = events[['symbol', 'hour_ms']].drop_duplicates()
        seeds = []
        source_hashes = {}
        for pair in pairs.itertuples(index=False):
            hour = datetime.fromtimestamp(pair.hour_ms / 1000, timezone.utc)
            try:
                q, source_sha = threshold(hour.date().isoformat(), hour.hour, pair.symbol)
            except (FileNotFoundError, ValueError) as exc:
                unavailable.append({'symbol': pair.symbol, 'hour': hour.isoformat(),
                                    'reason': type(exc).__name__})
                continue
            seeds.append((pair.symbol, pair.hour_ms, *q))
            source_hashes[(pair.symbol, pair.hour_ms)] = source_sha
        if seeds:
            limits = pd.DataFrame(seeds, columns=('symbol', 'hour_ms', 'q50', 'q95', 'q995'))
            events = events.merge(limits, on=['symbol', 'hour_ms'], how='left', validate='many_to_one')
            events = events.loc[events.q50.notna()].copy()
            events['small'] = events.notional.le(events.q50)
            events['big'] = events.notional.gt(events.q95) & events.notional.le(events.q995)
            events['ultra'] = events.notional.gt(events.q995)
            reductions = {}
            for name in ('big', 'ultra'):
                bucket = events.loc[events[name]]
                for side, maker in (('buy', False), ('sell', True)):
                    reductions[name+'_'+side] = bucket.loc[bucket.maker.eq(maker)].groupby(
                        ['target', 'symbol']).notional.sum()
                reductions[name+'_total'] = bucket.groupby(['target', 'symbol']).notional.sum()
            small = events.loc[events.small]
            for side, maker in (('buy', False), ('sell', True)):
                reductions['small_'+side] = small.loc[small.maker.eq(maker)].groupby(
                    ['target', 'symbol']).qty.sum()
            for target, symbol in events[['target', 'symbol']].drop_duplicates().itertuples(index=False):
                key = (target, symbol)
                target = target.to_pydatetime()
                components[target][symbol] = {
                    name: float(series.get(key, 0.0)) for name, series in reductions.items()}
            for target, symbol, hour_ms in events[['target', 'symbol', 'hour_ms']].drop_duplicates().itertuples(index=False):
                hour = datetime.fromtimestamp(hour_ms / 1000, timezone.utc)
                key = f'{hour.date().isoformat()}/{hour.hour:02d}/{symbol}'
                fingerprints[target.to_pydatetime()][key] = source_hashes[(symbol, hour_ms)]
    written, waiting = 0, 0
    for target in sorted(targets):
        output = ROOT / 'minutes' / target.date().isoformat() / f'{target.hour:02d}{target.minute:02d}.json'
        # The most recent minute may be revised while more sealed receive
        # shards arrive. A candidate day is valid only after every minute has
        # a complete marker and its raw input evidence can be rechecked.
        atomic_json(output, {'minute': target.isoformat(),
                             'formula_version': FORMULA_VERSION,
                             'minute_code_sha256': CODE_SHA256,
                             'status': 'waiting' if unavailable or invalid else 'provisional',
                             'components': components[target], 'trade_counts': counts[target],
                             'threshold_source_sha256': fingerprints[target],
                             'missing_thresholds': unavailable,
                             'invalid_records_in_lookback': invalid,
                             'source_paths': [str(p.relative_to(CAPTURE)) for p in paths],
                             'produced_at': now.isoformat()})
        written += 1
        waiting += bool(unavailable or invalid)
    return {'minute': minute.isoformat(), 'written': written,
            'waiting': waiting, 'source_shards': len(paths)}


if __name__ == '__main__':
    print(json.dumps(produce(datetime.now(timezone.utc))), flush=True)
