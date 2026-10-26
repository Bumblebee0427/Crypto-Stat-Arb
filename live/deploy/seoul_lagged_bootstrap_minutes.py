#!/usr/bin/env python3
"""Fill minutes before the v3 warm producer started from sealed hourly stage.

This is a one-time repair. Existing warm minute markers are never overwritten.
The daily audit still compares counts and threshold source revisions with the
independently completed Seoul daily input before a row can be final.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pandas as pd

from data_center.base_strategy import symbol_name
from data_center.contract import atomic_json
from data_center.lagged_quantile_v3 import (
    FORMULA_VERSION, events_from_trades, minute_components,
)
from deploy.seoul_lagged_minute_shadow import (
    CODE_SHA256, ROOT, STAGED, UNIVERSE, COLS, threshold, threshold_source,
)


MAX_RECOVERY_HOUR_BYTES = 64 * 1024**2


def fill(day: str, through_hour: int, max_hours: int = 24, through_minute: int = 59) -> dict:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    if not 0 <= through_hour <= 23:
        raise ValueError('invalid through hour')
    if not 0<=through_minute<=59:raise ValueError('invalid through minute')
    end = start + timedelta(hours=through_hour,minutes=through_minute+1)
    if datetime.now(timezone.utc) < end + timedelta(minutes=5):
        raise ValueError('source hours must be closed and staged')
    symbols = sorted({symbol_name(s) for s in json.loads(UNIVERSE.read_text())['tradable_symbols']}
                     - {'BTCUSDT', 'ETHUSDT'})
    markers = {}
    normalized=0
    for n in range(through_hour*60+through_minute+1):
        instant = start + timedelta(minutes=n)
        path = ROOT/'minutes'/day/f'{instant.hour:02d}{instant.minute:02d}.json'
        if path.exists():
            try:
                old = json.loads(path.read_text())
                if old.get('status') == 'provisional':continue
                errors=old.get('missing_thresholds',[])
                used={instant.replace(minute=0)}
                if instant.minute==59:used.add(instant.replace(minute=0)+timedelta(hours=1))
                if (old.get('status')=='waiting' and errors and old.get('components')
                        and old.get('minute')==instant.isoformat()
                        and old.get('formula_version')==FORMULA_VERSION
                        and old.get('minute_code_sha256')==CODE_SHA256
                        and old.get('invalid_records_in_lookback')==0
                        and all(datetime.fromisoformat(e['hour']) not in used for e in errors)):
                    atomic_json(path,{**old,'status':'provisional','missing_thresholds':[],
                                      'recovery_error_scope_normalized':True})
                    normalized+=1
                    continue
                # Named per-coin errors are already actionable exclusions in
                # read_minute(). Re-reading them on every bounded scan starves
                # later genuinely missing hours when one coin stays blocked.
                if (old.get('status')=='waiting' and errors
                        and isinstance(old.get('components'),dict) and old['components']
                        and old.get('threshold_source_sha256')
                        and old.get('minute')==instant.isoformat()
                        and old.get('formula_version')==FORMULA_VERSION
                        and old.get('minute_code_sha256')==CODE_SHA256
                        and old.get('invalid_records_in_lookback')==0
                        and all(isinstance(e.get('symbol'),str) and e['symbol'] for e in errors)):
                    continue
            except (OSError, ValueError):pass
        markers[instant] = {'components': {}, 'trade_counts': {}, 'threshold_source_sha256': {}}
    hours = sorted({t.hour for t in markers})[:max_hours]
    markers = {t:row for t,row in markers.items() if t.hour in hours}
    if not markers:return {'day':day,'symbols':len(symbols),'minutes_written':0,'normalized_minutes':normalized}
    required = {t.replace(minute=0) for t in markers}
    required |= {t.replace(minute=0)+timedelta(hours=1) for t in markers if t.minute == 59}
    # Boundary trades at HH:00:00.000 contribute to the previous minute.
    # Read the next staged hour too, with that hour's own frozen thresholds.
    # The full next hour need not be closed; the target hour must be sealed.
    missing_sources = []
    for hour_start in required:
        folder=STAGED/hour_start.date().isoformat()/'hours'/f'{hour_start.hour:02d}'
        if not folder.is_dir():missing_sources.append(str(folder))
    if missing_sources:raise ValueError('hourly stage not ready: '+missing_sources[0])
    unavailable=[]
    for symbol in symbols:
        for hour_start in sorted(required):
            source = STAGED/hour_start.date().isoformat()/'hours'/f'{hour_start.hour:02d}'/f'{symbol}.csv'
            if not source.exists():
                unavailable.append({'symbol':symbol,'hour':hour_start.isoformat(),'reason':'missing_hourly_stage'})
                continue
            if source.stat().st_size > MAX_RECOVERY_HOUR_BYTES:
                unavailable.append({'symbol':symbol,'hour':hour_start.isoformat(),
                                    'reason':'hourly_source_exceeds_recovery_memory_budget'})
                continue
            prior = threshold_source(hour_start.date().isoformat(), hour_start.hour, symbol)
            cached = ROOT/'thresholds'/hour_start.date().isoformat()/f'{hour_start.hour:02d}'/f'{symbol}.json'
            if not cached.exists() and prior.exists() and prior.stat().st_size > MAX_RECOVERY_HOUR_BYTES:
                unavailable.append({'symbol':symbol,'hour':hour_start.isoformat(),
                                    'reason':'prior_hour_exceeds_recovery_memory_budget'})
                continue
            raw = pd.read_csv(source, header=None, names=COLS,
                              dtype={'id': 'int64', 'timestamp_ms': 'int64',
                                     'price': 'float64', 'qty': 'float64', 'maker': 'bool'})
            economic = raw.loc[~(raw.price.eq(0) & raw.qty.eq(0))]
            actual = pd.to_datetime(economic.timestamp_ms, unit='ms', utc=True).dt.floor('min')
            for minute, count in actual.value_counts().items():
                instant = minute.to_pydatetime()
                if instant in markers:
                    markers[instant]['trade_counts'][symbol] = int(count)
            events = events_from_trades(raw)
            try:q, source_sha = threshold(hour_start.date().isoformat(), hour_start.hour, symbol)
            except (OSError, ValueError) as exc:
                unavailable.append({'symbol':symbol,'hour':hour_start.isoformat(),'reason':type(exc).__name__})
                continue
            parts = minute_components(events, q, first_minute=pd.Timestamp(start))
            for label, component in parts.items():
                instant = datetime.fromisoformat(label)
                if instant not in markers:
                    continue
                current = markers[instant]['components'].setdefault(symbol, {})
                for name, value in component.items():
                    current[name] = current.get(name, 0.0) + value
                key = f'{hour_start.date().isoformat()}/{hour_start.hour:02d}/{symbol}'
                markers[instant]['threshold_source_sha256'][key] = source_sha
    written = 0
    for instant, row in markers.items():
        output = ROOT / 'minutes' / day / f'{instant.hour:02d}{instant.minute:02d}.json'
        if output.exists():
            try:
                if json.loads(output.read_text()).get('status') == 'provisional':continue
            except (OSError, ValueError):pass
        used={instant.replace(minute=0)}
        if instant.minute==59:used.add(instant.replace(minute=0)+timedelta(hours=1))
        relevant=[e for e in unavailable if datetime.fromisoformat(e['hour']) in used]
        atomic_json(output, {'minute': instant.isoformat(), 'formula_version': FORMULA_VERSION,
                             'minute_code_sha256': CODE_SHA256, 'status': 'waiting' if relevant else 'provisional',
                             **row, 'missing_thresholds': relevant,
                             'invalid_records_in_lookback': 0,
                             'source_paths': [f'hourly_stage/{t.date().isoformat()}/{t.hour:02d}' for t in sorted(required)],
                             'produced_at': datetime.now(timezone.utc).isoformat(),
                             'bootstrap_from_hourly_stage': True})
        written += 1
    return {'day': day, 'symbols': len(symbols), 'minutes_written': written,'normalized_minutes':normalized}


if __name__ == '__main__':
    print(json.dumps(fill(sys.argv[1], int(sys.argv[2]))), flush=True)
