"""Runtime gate before the fast minute pipeline can replace the daily batch."""
from datetime import date, timedelta, timezone, datetime
import hashlib
import json
import math
from pathlib import Path

from data_center.base_strategy import MAIN, SMALL
from data_center.contract import atomic_json

CODE_FILES = ('data_center/minute_stream.py', 'data_center/captured_daily.py',
              'data_center/individual_capture.py', 'data_center/reference_formula_v2.py')
EXACT = ('symbol', 'day', 'formula_version', 'source_version', 'status',
         'source_certified', 'source_complete', 'minute_count',
         'boundary_minute_count', 'trade_rows', 'sequence_rows',
         'zero_value_sequence_rows', 'event_rows', 'missing_trade_ids',
         'first_trade_id', 'last_trade_id', 'first_exchange_time_ms',
         'last_exchange_time_ms', 'time_monotonic_by_id')


def code_hashes(base: Path) -> dict:
    return {name: hashlib.sha256((base/name).read_bytes()).hexdigest()
            for name in CODE_FILES}


def compare(base: Path, day: str) -> dict:
    root = base/'data/captured_daily_stream_v1'
    fast = root/'inputs/live_daily'/day
    batch = base/'data/captured_daily_v1/inputs/live_daily'/day
    fm = json.loads((fast/'complete.json').read_text())
    bm = json.loads((batch/'complete.json').read_text())
    symbols = fm['symbols']
    errors = []
    if fm.get('minute_code_sha256') != code_hashes(base)['data_center/minute_stream.py']:
        errors.append('fast_marker_uses_older_minute_code')
    if fm['day'] != day or bm['day'] != day or symbols != bm['symbols']:
        errors.append('different completed symbol universe')
    for symbol in symbols:
        try:
            fpath, bpath = fast/(symbol+'.json'), batch/(symbol+'.json')
            fb, bb = fpath.read_bytes(), bpath.read_bytes()
            if (hashlib.sha256(fb).hexdigest() != fm['files_sha256'][symbol]
                    or hashlib.sha256(bb).hexdigest() != bm['files_sha256'][symbol]):
                errors.append(symbol+':marker_hash')
                continue
            f, b = json.loads(fb), json.loads(bb)
            for key in EXACT:
                if f.get(key) != b.get(key):
                    errors.append(symbol+':'+key)
            for key in (MAIN, SMALL):
                x, y = f.get(key), b.get(key)
                if x is None or y is None:
                    if x != y: errors.append(symbol+':'+key)
                elif not (math.isfinite(x) and math.isfinite(y)
                          and math.isclose(x, y, rel_tol=0, abs_tol=1e-10)):
                    errors.append(symbol+':'+key)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            errors.append(symbol+':'+type(exc).__name__)
    result = {'day':day,'status':'passed' if symbols and not errors else 'blocked',
              'symbols':len(symbols),'mismatches':len(errors),
              'examples':errors[:20], 'code_hashes':code_hashes(base),
              'checked_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(root/'validation.json',result)
    return result


def valid_for(base: Path, day: str) -> bool:
    """A prior full-day match validates the unchanged fast calculation code."""
    path = base/'data/captured_daily_stream_v1/validation.json'
    if not path.exists():return False
    row = json.loads(path.read_text())
    prior = (date.fromisoformat(day)-timedelta(days=1)).isoformat()
    return (row.get('day') == prior and row.get('status') == 'passed'
            and row.get('symbols', 0) >= 500
            and row.get('code_hashes') == code_hashes(base))
