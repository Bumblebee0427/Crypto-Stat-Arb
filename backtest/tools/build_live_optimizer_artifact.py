#!/usr/bin/env python3
"""Compute a daily optimizer artifact from audited, local, point-in-time inputs.

Input JSON keys: returns, factors, settlements, provenance (paths relative to
the input JSON), plus optional alpha_panel. The observation contains positions,
marks and equity; it contains no exchange credentials. No network or orders.
"""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.live import allocation, MODEL
# Support both the private source layout and the public export (live/ + backtest/).
for live_root in (ROOT.parent/'select-coin-hft-pro', ROOT.parent/'live'):
    if (live_root/'data_center/live_schedule.py').is_file():
        sys.path.insert(0, str(live_root))
        break
from data_center.live_schedule import LIVE_WINDOW_MINUTES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    body = {k: v for k, v in value.items() if k not in ('snapshot_id', 'checksum')}
    ident = digest(body)
    return {**body, 'snapshot_id': ident, 'checksum': ident}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--signal', type=Path, required=True)
    p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preview', action='store_true')
    args = p.parse_args()
    now = datetime.now(timezone.utc)
    config = json.loads(args.inputs.read_text())
    signal = json.loads(args.signal.read_text())
    obs = json.loads(args.observation.read_text())
    report = {'schema_version': 'select_coin.optimizer.v1', 'model': MODEL,
        'generated_at': now.isoformat(), 'base_snapshot_id': signal['snapshot_id'],
        'asof': signal['window_end'], 'diagnostic_only': args.preview}
    def read(key):
        path = Path(config[key])
        return path if path.is_absolute() else args.inputs.parent/path
    try:
        if seal(signal) != signal or signal.get('status') != 'complete' or signal.get('blockers'):
            raise ValueError('base signal is incomplete or has an invalid checksum')
        boundary = datetime.fromisoformat(signal['window_end'])
        deadline = boundary+timedelta(minutes=LIVE_WINDOW_MINUTES)
        if not args.preview and not boundary <= now < deadline:
            raise ValueError('outside the daily opening window')
        observed = datetime.fromisoformat(obs['checked_at'])
        if not observed <= now or now-observed > timedelta(seconds=60):
            raise ValueError('account observation is stale or in the future')
        if obs['base_snapshot_id'] != signal['snapshot_id']:
            raise ValueError('account observation refers to a different base signal')
        positions = {s: str(Decimal(str(q))) for s, q in obs['positions'].items() if Decimal(str(q))}
        budget = min(1500., float(obs['equity'])*.5)
        result = allocation(signal, pd.read_parquet(read('returns')),
            pd.read_parquet(read('factors')), pd.read_parquet(read('settlements')),
            positions, obs['marks'], budget, json.loads(read('provenance').read_text()),
            alpha_panel=pd.read_parquet(read('alpha_panel')) if config.get('alpha_panel') else None)
        report.update(result, status='ready', positions_digest=digest(positions),
                      budget_usdt=budget, valid_until=deadline.isoformat(),
                      max_observation_at=result['maximum_available_at'])
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        report.update(status='blocked', reason=str(exc), allocations={})
    report = seal(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.json.pending')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(json.dumps({k: report.get(k) for k in ['status', 'reason', 'model', 'asof', 'alpha_status']}))
    return 0 if report['status'] == 'ready' else 2


if __name__ == '__main__':
    raise SystemExit(main())
