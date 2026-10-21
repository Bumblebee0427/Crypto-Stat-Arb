#!/usr/bin/env python3
"""At UTC midnight, publish audited previous-hour TOP50 targets by +10 minutes."""
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from data_center.base_strategy import finite, symbol_name
from data_center.contract import atomic_json, timestamp
from data_center.public_inputs import fetch_btc_closes
from data_center.publisher import build, publish
from data_center.top50_strategy import LAGGED_FORMULA_VERSION,LAGGED_STRATEGY_VERSION
from data_center.lagged_quantile_v3 import FORMULA_CODE_SHA256
from data_center.live_schedule import LIVE_WINDOW_MINUTES
from data_center.lagged_universe import readiness
from deploy.seoul_lagged_daily_shadow import emit
from deploy.seoul_lagged_fast_metadata import metadata as fast_metadata

from data_center.paths import BASE as _PORTABLE_BASE

BASE = _PORTABLE_BASE
ROOT = BASE/'data/base_signal_v1'
V3 = BASE/'data/lagged_quantile_v3'
UNIVERSE = BASE/'data/live_factor_storage/_state/universe.json'


def wait_metadata(day, deadline, symbols):
    last_error = None
    while datetime.now(timezone.utc) < deadline:
        try:
            return fast_metadata(day, symbols, exclude_invalid=True), None
        except (ValueError, OSError, KeyError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            if error != last_error:
                print(json.dumps({'day': day, 'metadata_wait': error[:180]}), flush=True)
                last_error = error
            time.sleep(2)
    return None, last_error


def ready_preboundary_btc(payload, end, near, far):
    if not isinstance(payload, dict):
        return False
    try:
        fetched_at = timestamp(payload['fetched_at'])
    except (KeyError, TypeError, ValueError):
        return False
    closes = payload.get('closes', {})
    if not isinstance(closes, dict):
        return False
    prices = (closes.get(near.isoformat()), closes.get(far.isoformat()))
    return (end-timedelta(hours=1) <= fetched_at < end
            and all(finite(value) and value > 0 for value in prices))


def main():
    now = datetime.now(timezone.utc)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if now.hour >= 23:
        end += timedelta(days=1)
    day = (end-timedelta(days=1)).date().isoformat()
    deadline = end+timedelta(minutes=LIVE_WINDOW_MINUTES)
    last = 'not_closed'
    emit_seconds = None
    build_seconds = None
    emit_attempts = 0
    last_exception = None
    universe = json.loads(UNIVERSE.read_text())
    cached_btc = json.loads((ROOT/'inputs/btc_hourly.json').read_text())
    symbols, deferred_new_symbols = readiness(universe, day, BASE/'data/captured_daily_stream_v1/work'/day)
    eligible = {s for s in symbols if s in cached_btc.get('first_candle_by_symbol', {})}
    print(json.dumps({'day':day,'eligible_symbols':len(eligible),
                      'deferred_new_symbols':deferred_new_symbols}), flush=True)
    near = end-timedelta(hours=2)  # Backtested BTC shift1: 06:00–07:00 CST.
    far = near-timedelta(days=30)
    pool = ThreadPoolExecutor(max_workers=2)
    metadata_future = None
    btc_future = None
    signal_btc = cached_btc if ready_preboundary_btc(cached_btc, end, near, far) else None
    if signal_btc is not None:
        print(json.dumps({'day': day, 'btc_input': 'preboundary_ready',
                          'fetched_at': cached_btc['fetched_at'],
                          'hours': [near.isoformat(), far.isoformat()]}), flush=True)
    while datetime.now(timezone.utc) < deadline:
        if datetime.now(timezone.utc) < end:
            time.sleep(2)
            continue
        if metadata_future is None:
            metadata_future = pool.submit(wait_metadata, day, deadline, eligible)
        if signal_btc is None and btc_future is None:
            btc_future = pool.submit(fetch_btc_closes, near)
        try:
            marker = V3/'inputs/live_daily'/day/'complete.json'
            if not marker.exists():
                minutes = V3/'minutes'/day
                stage_state = BASE/'data/captured_daily_stream_v1/work'/day/'stage_state.json'
                if stage_state.exists() and len(list(minutes.glob('*.json'))) == 1440:
                    if not metadata_future.done():
                        last = 'waiting_for_parallel_source_metadata'
                    else:
                        prepared, metadata_error = metadata_future.result()
                        if prepared is None:
                            raise ValueError('source metadata missed publication window: '
                                             + str(metadata_error))
                        started = time.monotonic()
                        emit_attempts += 1
                        print(json.dumps({'day':day,'stage':'emit_started',
                                          'attempt':emit_attempts}), flush=True)
                        try:
                            emit(day, prepared_metadata=prepared)
                        finally:
                            emit_seconds = round(time.monotonic()-started,3)
                else:
                    last = 'waiting_for_complete_v3_minutes_or_seoul_stage'
            if marker.exists():
                completed = json.loads(marker.read_text())
                if (completed.get('status') != 'complete'
                        or completed.get('formula_version') != LAGGED_FORMULA_VERSION
                        or not eligible.issubset(set(completed.get('symbols', []))
                                                | set(completed.get('excluded_symbols', {})))):
                    raise ValueError('v3 daily marker incomplete or wrong version')
                if signal_btc is None:
                    if btc_future.done():
                        try:
                            closes = btc_future.result()
                            signal_btc = {**cached_btc, 'closes': closes,
                                          'fetched_at': datetime.now(timezone.utc).isoformat(),
                                          'source': 'binance_public_fapi_1h_boundary_recovery'}
                            print(json.dumps({'day': day, 'btc_input': 'boundary_recovery',
                                              'hours': list(closes)}), flush=True)
                        except (ValueError, OSError) as exc:
                            btc_future = None
                            last = 'boundary_btc_fetch_failed:'+type(exc).__name__
                    if signal_btc is None:
                        last = 'waiting_for_boundary_btc_recovery'
                        time.sleep(2)
                        continue
                atomic_json(ROOT/'top50/inputs/btc_hourly'/f'{day}.json', signal_btc)
                started = time.monotonic()
                snapshot = build(ROOT, UNIVERSE, day, profile='top50', factor_root=V3,
                                 formula_version=LAGGED_FORMULA_VERSION,
                                 strategy_version=LAGGED_STRATEGY_VERSION,
                                 formula_code_sha256=FORMULA_CODE_SHA256,
                                 btc_input=signal_btc, universe_input=universe)
                build_seconds = round(time.monotonic()-started,3)
                last = snapshot['status']+':'+','.join(snapshot['blockers'][:3])
                if snapshot['status'] == 'complete' and datetime.now(timezone.utc) < deadline:
                    publish(ROOT/'top50', snapshot)
                    print(json.dumps({'day':day,'status':'published',
                                      'snapshot_id':snapshot['snapshot_id'],
                                      'generated_at':snapshot['generated_at'],
                                      'emit_seconds':emit_seconds,
                                      'emit_attempts':emit_attempts,
                                      'build_seconds':build_seconds,
                                      'eligible_symbols':snapshot['coverage']['eligible'],
                                      'excluded_symbols':snapshot['coverage']['excluded_symbols']}), flush=True)
                    return
        except Exception as exc:
            last = type(exc).__name__+': '+str(exc)[:180]
            print(json.dumps({'day':day,'stage':'publish_attempt_failed',
                              'attempt':emit_attempts,'error':last,
                              'emit_seconds':emit_seconds}), flush=True)
            last_exception = last
            # Rebuild metadata after a source revision; a completed Future
            # holds the original audit permanently unless explicitly replaced.
            metadata_future = None
        time.sleep(3)
    print(json.dumps({'day':day,'status':'missed_ten_minute_window','reason':last,
                      'last_exception':last_exception,'emit_attempts':emit_attempts,
                      'emit_seconds':emit_seconds,'build_seconds':build_seconds}), flush=True)


if __name__ == '__main__':
    main()
