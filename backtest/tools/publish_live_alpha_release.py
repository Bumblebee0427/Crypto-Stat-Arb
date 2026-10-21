#!/usr/bin/env python3
"""Install a validated historical alpha panel into the local daily producer.

The installation is atomic and contains no order or account operation. A report
that fails any declared gate, or belongs to a different panel, is rejected.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.alpha_bootstrap import alpha_policy, validate_release
from research.optimizer.contracts import file_hash, digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--live-root', type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.validation.read_text())
    panel = a.history/'alpha_panel.parquet'
    if (report.get('accepted') is not True or report.get('alpha_policy') != alpha_policy() or not report.get('checks')
            or not all(report['checks'].values()) or report['panel_sha256'] != file_hash(panel)):
        raise ValueError('cannot publish: validation failed or training panel changed')
    destination = a.live_root/'alpha_bootstrap'
    if destination.exists(): raise ValueError('release exists; explicitly archive the prior release before replacement')
    temporary = Path(tempfile.mkdtemp(prefix='.alpha-bootstrap-', dir=a.live_root))
    try:
        shutil.copy2(panel, temporary/'historical_panel.parquet')
        shutil.copy2(a.validation, temporary/'validation.json')
        body = {'schema': 'select_coin.alpha_bootstrap.v1', 'policy': alpha_policy(),
            'panel_sha256': file_hash(temporary/'historical_panel.parquet'),
            'validation_sha256': file_hash(temporary/'validation.json'),
            'created_at': datetime.now(timezone.utc).isoformat()}
        write_json(temporary/'release.json', {**body, 'release_id': digest(body)})
        validate_release(temporary, datetime.now(timezone.utc))
        os.rename(temporary, destination)
    except Exception:
        if temporary.exists(): shutil.rmtree(temporary)
        raise
    print(json.dumps({'status': 'installed', 'release_id': digest(body), 'orders_submitted': False}))


if __name__ == '__main__': main()
