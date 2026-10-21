#!/usr/bin/env python3
"""Prepare the stable v3 minute prefix before the UTC day boundary.

The minute producer revisits its last three targets. At 23:57:45 UTC,
23:54 is the latest minute that will no longer be rewritten by that producer.
The saved rolling outputs let the daily sealer process only 23:55–23:59.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from data_center.contract import atomic_json, digest
from data_center.lagged_quantile_v3 import FORMULA_VERSION, minute_frame
from data_center.reference_formula_v2 import EPS
from data_center.lagged_universe import split as split_universe
from deploy.seoul_lagged_minute_shadow import CODE_SHA256 as MINUTE_CODE_SHA256


from data_center.paths import BASE

ROOT = BASE / 'data/lagged_quantile_v3'
PREFIX_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LAST_PREFIX_MINUTE = 23 * 60 + 54
ROLLING = 240
LAG = 5


def minute_file(day: str, n: int) -> Path:
    return ROOT / 'minutes' / day / f'{n // 60:02d}{n % 60:02d}.json'


def read_minute(day: str, n: int) -> tuple[dict, str, dict]:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    minute = start + timedelta(minutes=n)
    path = minute_file(day, n)
    payload = path.read_bytes()
    row = json.loads(payload)
    usable_status = row.get('status') == 'provisional'
    if row.get('status') == 'waiting':
        missing = row.get('missing_thresholds', [])
        # Threshold errors identify the affected symbols. Preserve valid
        # components for every other symbol and exclude the bad symbols.
        usable_status = (bool(missing) and row.get('invalid_records_in_lookback') == 0
                         and all(isinstance(item.get('symbol'), str) and item['symbol']
                                 for item in missing))
        if usable_status:
            row = dict(row, excluded_symbols={item['symbol']:['missing_prior_hour_threshold']
                                               for item in missing})
    if (row.get('minute') != minute.isoformat()
            or row.get('formula_version') != FORMULA_VERSION
            or row.get('minute_code_sha256') != MINUTE_CODE_SHA256
            or not usable_status):
        raise ValueError('missing or blocked v3 minute: ' + minute.isoformat())
    if row['components'] and not row.get('threshold_source_sha256'):
        raise ValueError('minute lacks prior-hour threshold evidence: ' + minute.isoformat())
    info = path.stat()
    return row, hashlib.sha256(payload).hexdigest(), {
        'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
    }


def rolling_outputs(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Use the same Pandas operations and observation windows as daily_factors."""
    big, ultra = frame.big, frame.ultra
    harmonic = np.sign(big + ultra) * 2 * big.abs() * ultra.abs() / (big.abs() + ultra.abs() + EPS)
    main = harmonic.rolling(ROLLING, min_periods=ROLLING).mean()
    ranks = frame.small_qty.rolling(ROLLING, min_periods=ROLLING).rank(method='average', pct=True)
    lag = frame.small_qty.shift(LAG).rolling(ROLLING, min_periods=ROLLING).rank(
        method='average', pct=True)
    return main, ranks, lag


def values(series: pd.Series) -> list[float | None]:
    return [float(value) if pd.notna(value) else None for value in series]


def prepare_symbol(parts: dict, day: str) -> dict:
    frame = minute_frame(parts, day)
    main, ranks, lag = rolling_outputs(frame)
    history = frame.tail(ROLLING + LAG - 1)
    return {
        'minute_count': len(frame),
        'tail_inputs': [[index.isoformat(), float(row.big), float(row.ultra), float(row.small_qty)]
                        for index, row in history.iterrows()],
        'main_values': values(main),
        'rank_values': values(ranks),
        'lag_values': values(lag),
    }


def finish_symbol(state: dict, parts: dict, day: str) -> dict:
    """Append post-prefix sparse observations without rerunning prefix rolling work."""
    tail = minute_frame(parts, day) if parts else None
    if tail is None:
        new_main = new_rank = new_lag = []
        tail_count = 0
    else:
        history = pd.DataFrame(
            [(big, ultra, small) for _, big, ultra, small in state['tail_inputs']],
            index=pd.to_datetime([minute for minute, *_ in state['tail_inputs']], utc=True),
            columns=('big', 'ultra', 'small_qty'), dtype=float)
        combined = pd.concat((history, tail))
        if not combined.index.is_monotonic_increasing or not combined.index.is_unique:
            raise ValueError('prefix and tail minute order overlap')
        main, ranks, lag = rolling_outputs(combined)
        tail_count = len(tail)
        new_main, new_rank, new_lag = (values(series.iloc[-tail_count:])
                                       for series in (main, ranks, lag))
    main = pd.Series(state['main_values'] + new_main, dtype=float).mean()
    ranks = pd.Series(state['rank_values'] + new_rank, dtype=float)
    lag = pd.Series(state['lag_values'] + new_lag, dtype=float)
    small = -ranks.corr(lag)
    from data_center.base_strategy import MAIN, SMALL
    return {MAIN: float(main) if np.isfinite(main) else None,
            SMALL: float(small) if np.isfinite(small) else None,
            'formula_version': FORMULA_VERSION,
            'minute_count': state['minute_count'] + tail_count,
            'boundary_minute_count': 0, 'source_certified': False}


def prepare(day: str) -> dict:
    started = time.monotonic()
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) < start + timedelta(minutes=LAST_PREFIX_MINUTE + 3):
        raise ValueError('latest prefix minute has not exited its revision window')
    parts: dict[str, dict] = {}
    trade_counts: dict[str, int] = {}
    threshold_hashes: dict[str, str] = {}
    minute_hashes: dict[str, str] = {}
    minute_stats: dict[str, dict] = {}
    excluded_symbols = {}
    universe=json.loads((ROOT.parent/'live_factor_storage/_state/universe.json').read_text())
    eligible,_=split_universe(universe,day,ROOT.parent/'captured_daily_stream_v1/work'/day)
    for n in range(LAST_PREFIX_MINUTE + 1):
        row, sha, info = read_minute(day, n)
        excluded_symbols.update(row.get("excluded_symbols", {}))
        key = f'{n // 60:02d}{n % 60:02d}.json'
        minute_hashes[key] = sha
        minute_stats[key] = info
        for symbol, components in row['components'].items():
            if symbol not in eligible:continue
            parts.setdefault(symbol, {})[row['minute']] = components
        for symbol, count in row.get('trade_counts', {}).items():
            if symbol not in eligible:continue
            trade_counts[symbol] = trade_counts.get(symbol, 0) + count
        for key, source_sha in row.get('threshold_source_sha256', {}).items():
            if key.rsplit('/',1)[-1] not in eligible:continue
            if key in threshold_hashes and threshold_hashes[key] != source_sha:
                excluded_symbols[key.rsplit('/', 1)[-1]]=['threshold_revised_during_utc_hour:'+key]
            threshold_hashes[key] = source_sha
    symbols = {symbol: prepare_symbol(minutes, day) for symbol, minutes in parts.items()
               if symbol not in excluded_symbols}
    result = {
        'day': day, 'status': 'complete', 'last_minute': LAST_PREFIX_MINUTE,
        'formula_version': FORMULA_VERSION, 'prefix_code_sha256': PREFIX_CODE_SHA256,
        'minute_code_sha256': MINUTE_CODE_SHA256,
        'minute_hashes': minute_hashes, 'minute_stats': minute_stats,
        'threshold_hashes': threshold_hashes, 'trade_counts': trade_counts,
        'symbols': symbols, 'excluded_symbols': excluded_symbols,
        'prepared_at': datetime.now(timezone.utc).isoformat(),
    }
    result['checksum'] = digest(result)
    atomic_json(ROOT / 'prefix' / f'{day}.json', result)
    return {'day': day, 'status': 'complete', 'symbols': len(symbols),
            'prefix_minutes': len(minute_hashes),
            'seconds': round(time.monotonic() - started, 3)}


def load(day: str) -> dict:
    prefix = json.loads((ROOT / 'prefix' / f'{day}.json').read_text())
    if (prefix.get('day') != day or prefix.get('status') != 'complete'
            or prefix.get('last_minute') != LAST_PREFIX_MINUTE
            or prefix.get('formula_version') != FORMULA_VERSION
            or prefix.get('prefix_code_sha256') != PREFIX_CODE_SHA256
            or prefix.get('minute_code_sha256') != MINUTE_CODE_SHA256
            or len(prefix.get('minute_hashes', {})) != LAST_PREFIX_MINUTE + 1
            or len(prefix.get('minute_stats', {})) != LAST_PREFIX_MINUTE + 1
            or prefix.get('checksum') != digest({key: value for key, value in prefix.items()
                                                  if key != 'checksum'})):
        raise ValueError('invalid v3 daily prefix')
    expected = {f'{n // 60:02d}{n % 60:02d}.json' for n in range(LAST_PREFIX_MINUTE + 1)}
    if set(prefix['minute_hashes']) != expected or set(prefix['minute_stats']) != expected:
        raise ValueError('v3 daily prefix minute set is incomplete')
    for name, prior in prefix['minute_stats'].items():
        info = (ROOT / 'minutes' / day / name).stat()
        if info.st_size != prior['size'] or info.st_mtime_ns != prior['mtime_ns']:
            raise ValueError('stable prefix minute revised: ' + name)
    return prefix


if __name__ == '__main__':
    now = datetime.now(timezone.utc)
    print(json.dumps(prepare(now.date().isoformat())), flush=True)
