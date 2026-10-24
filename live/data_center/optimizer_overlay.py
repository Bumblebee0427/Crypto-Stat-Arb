"""Validate a computation-only optimizer artifact before sizing live orders.

The executor remains dependency-free. Missing, stale or invalid research
artifacts preserve the original daily basket instead of flattening it.
"""
from datetime import timedelta
from decimal import Decimal
import json
import math
from pathlib import Path

from data_center.contract import digest, seal, timestamp
from data_center.live_schedule import LIVE_WINDOW_MINUTES, OPTIMIZER_WAIT_SECONDS

SCHEMA = 'select_coin.optimizer.v1'
MODEL = 'core_w90_f14_r14_cv_diagonal'
BLEND = 0.25


def wait_deadline(snapshot):
    # Three five-second polling ticks before the last decision. A failed
    # optimizer must still leave an opportunity to execute the baseline.
    return min(timestamp(snapshot['generated_at'])+timedelta(seconds=OPTIMIZER_WAIT_SECONDS),
               timestamp(snapshot['window_end'])+timedelta(minutes=LIVE_WINDOW_MINUTES,seconds=-15))


def should_wait(info, root, now, snapshot=None):
    """Only a missing computation gets a bounded grace period; failures fall back."""
    if info.get('status') in ('applied', 'baseline_fallback') or snapshot is None:
        return False
    try:
        config = json.loads((Path(root)/'optimizer/enabled.json').read_text())
        generated = timestamp(snapshot['generated_at'])
        boundary = timestamp(snapshot['window_end'])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not isinstance(config, dict):
        return False
    deadline = wait_deadline(snapshot)
    return (config.get('enabled') is True and config.get('model') == MODEL
            and boundary <= generated <= now < deadline)


def awaiting_artifact(snapshot, root, now):
    """Wait for missing/previous-signal artifacts; a failed current artifact is final."""
    if not should_wait({}, root, now, snapshot):
        return False
    try:
        path = Path(root)/'optimizer'/('target_'+snapshot['window_end'][:10]+'.json')
        if path.stat().st_size > 1_000_000:
            return False
        a = json.loads(path.read_text())
        if not isinstance(a, dict):
            return False
        # Files are atomically replaced. A malformed current file is a producer
        # failure, not a reason to postpone a valid original strategy signal.
        if seal(a) != a:
            return False
        return a.get('base_snapshot_id') != snapshot['snapshot_id']
    except FileNotFoundError:
        return True
    except (OSError, ValueError, TypeError):
        return False


def positions_digest(positions):
    return digest({s: str(Decimal(str(q))) for s, q in positions.items() if Decimal(str(q))})


def apply(snapshot, artifact, positions, budget, now):
    if not isinstance(artifact, dict) or seal(artifact) != artifact or artifact.get('schema_version') != SCHEMA:
        raise ValueError('optimizer artifact integrity/version mismatch')
    if artifact.get('model') != MODEL or artifact.get('blend') != BLEND:
        raise ValueError('optimizer model differs from authorized configuration')
    if artifact.get('status') != 'ready' or artifact.get('diagnostic_only') is not False:
        raise ValueError('optimizer artifact is not actionable')
    if artifact.get('base_snapshot_id') != snapshot['snapshot_id']:
        raise ValueError('optimizer artifact refers to another signal')
    boundary = timestamp(snapshot['window_end'])
    if timestamp(artifact['asof']) != boundary:
        raise ValueError('optimizer decision boundary mismatch')
    generated = timestamp(artifact['generated_at'])
    if not boundary <= generated <= now or now-generated > timedelta(minutes=10):
        raise ValueError('optimizer artifact future/stale')
    if timestamp(artifact['valid_until']) != boundary+timedelta(minutes=LIVE_WINDOW_MINUTES) or now >= timestamp(artifact['valid_until']):
        raise ValueError('optimizer opening window expired')
    if timestamp(artifact['max_observation_at']) > boundary:
        raise ValueError('optimizer contains future observations')
    if artifact.get('positions_digest') != positions_digest(positions):
        raise ValueError('optimizer current positions changed')
    basis = artifact.get('budget_usdt')
    if isinstance(basis, bool) or not isinstance(basis, (int, float)) or not math.isfinite(basis) or basis <= 0:
        raise ValueError('optimizer invalid budget')
    if abs(basis/float(budget)-1) > .01:
        raise ValueError('optimizer budget changed by more than one percent')
    baseline, weights = snapshot['allocations'], artifact['allocations']
    if not isinstance(weights, dict) or set(weights) != set(baseline):
        raise ValueError('optimizer changed the selected pool')
    # A raw 0.5--1.5 bound combined 25% with baseline permits 0.875--1.125.
    for symbol, base in baseline.items():
        value = weights[symbol]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('optimizer nonfinite allocation')
        if base == 0:
            if value != 0: raise ValueError('optimizer opened a zero-weight symbol')
        elif not .875-1e-8 <= value/base <= 1.125+1e-8:
            raise ValueError('optimizer direction/relative bound violation')
    for sign in [-1, 1]:
        wanted = sum(v for v in baseline.values() if v*sign > 0)
        actual = sum(v for v in weights.values() if v*sign > 0)
        if abs(wanted-actual) > 1e-9:
            raise ValueError('optimizer altered a side budget')
    result = seal({**snapshot, 'allocations': weights,
        'gross_allocation': sum(abs(v) for v in weights.values()),
        'base_snapshot_id': snapshot['snapshot_id'],
        'allocation_method': MODEL, 'optimizer_artifact_id': artifact['snapshot_id']})
    return result


def load_and_apply(snapshot, root, positions, budget, now):
    path = Path(root)/'optimizer'/('target_'+snapshot['window_end'][:10]+'.json')
    try:
        if path.stat().st_size > 1_000_000:
            raise ValueError('oversized optimizer artifact')
        artifact = json.loads(path.read_text())
        result = apply(snapshot, artifact, positions, budget, now)
        return result, {'status': 'applied', 'model': MODEL,
            'artifact_id': artifact['snapshot_id'], 'base_snapshot_id': snapshot['snapshot_id'],
            'blend': BLEND, 'solver_status': artifact.get('solver_status'),
            'alpha_status': artifact.get('alpha_status')}
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError):
        return snapshot, {'status': 'baseline_fallback', 'model': MODEL,
            'reason': 'missing_or_invalid_optimizer_artifact',
            'base_snapshot_id': snapshot['snapshot_id']}
