#!/usr/bin/env python3
"""Backfill actual published v3 features and post-publication price labels.

Source JSON files contain a snapshots mapping. Only sealed signal fields enter
the training store; account observations or deployment configuration do not.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_live_optimizer import archive_v3_alpha
from tools.build_live_optimizer_artifact import seal
from research.optimizer.contracts import write_json
from research.optimizer.v3_history import V3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--snapshot-file', type=Path, action='append', required=True)
    p.add_argument('--since', required=True)
    a = p.parse_args(); selected = {}
    lock = (a.root/'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path in a.snapshot_file:
        for value in json.loads(path.read_text()).get('snapshots', {}).values():
            if value.get('formula_version') != V3 or value.get('status') != 'complete': continue
            date = pd.Timestamp(value['window_end'])
            if date < pd.Timestamp(a.since, tz='UTC'): continue
            if seal(value) != value: raise ValueError('published historical snapshot checksum mismatch')
            old = selected.get(date)
            if old is None or pd.Timestamp(value['generated_at']) < pd.Timestamp(old['generated_at']):
                selected[date] = value
    for date, value in sorted(selected.items()):
        archive_v3_alpha(a.root, value, date.to_pydatetime())
        print(json.dumps({'imported_published_asof':str(date),'snapshot_id':value['snapshot_id']}),flush=True)
    panel = pd.read_parquet(a.root/'alpha_panel.parquet')
    write_json(a.root/'published_history_import.json', {'since':a.since,'days':len(selected),
        'first':str(min(selected)),'last':str(max(selected)),
        'mature_days':int(panel.loc[panel.future_return.notna(),'asof'].nunique()),
        'scope':'Actual published features; labels begin after publication; actual execution not asserted.',
        'orders_submitted':False})


if __name__ == '__main__': main()
