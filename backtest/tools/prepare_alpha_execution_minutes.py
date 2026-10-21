#!/usr/bin/env python3
"""Extract checksummed official 00:10–01:00 UTC window VWAPs from local archives."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.contracts import file_hash, write_json


def extract(task):
    symbol, path = task; path = Path(path)
    checksum = path.with_name(path.name+'.CHECKSUM')
    if not checksum.exists(): raise ValueError('official archive checksum absent: '+str(path))
    expected = checksum.read_text().split()[0]
    if file_hash(path) != expected: raise ValueError('official minute archive checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith('.csv')]
        if len(members) != 1: raise ValueError('ambiguous minute archive')
        frame = pd.read_csv(archive.open(members[0]), header=None, usecols=[0,5,7])
        frame.columns = ['time','qty','notional']
    frame = frame.apply(pd.to_numeric, errors='coerce').dropna()
    units = 1000 if frame.time.max() > 10**14 else 1
    ms = frame.time.astype('int64')//units
    minute = ms % 86_400_000
    chosen = frame.loc[(minute >= 600_000) & (minute < 3_600_000)].copy()
    chosen['asof'] = pd.to_datetime(ms.loc[chosen.index], unit='ms').dt.normalize()
    chosen = chosen.loc[(chosen.qty >= 0) & (chosen.notional >= 0)]
    chosen = chosen.groupby('asof', as_index=False)[['qty','notional']].sum()
    chosen['price'] = chosen.notional/chosen.qty
    chosen = chosen.loc[(chosen.qty > 0) & np.isfinite(chosen.price) & (chosen.price > 0)]
    return symbol, chosen[['asof','price']].to_dict('records'), {
        'path': str(path), 'sha256': expected, 'rows': len(chosen)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--minute-root', type=Path, action='append', required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(); root = a.history
    target = root/'execution_prices_10m.parquet'
    receipt = root/'execution_minutes_manifest.json'
    if (receipt.exists() and target.exists()
            and json.loads(receipt.read_text()).get('window_end_minutes') == 60
            and json.loads(receipt.read_text())['sha256'] == file_hash(target)):
        print('reuse audited official execution minutes', flush=True); return
    symbols = set(pd.read_parquet(root/'market_selection.parquet').symbol)
    scope=root/'historical_scope.json'
    if scope.exists():symbols-=set(json.loads(scope.read_text())['excluded_symbols'])
    symbols=sorted(symbols)
    months = pd.date_range('2025-07-01', '2026-10-01', freq='MS')
    tasks = []
    for symbol in symbols:
        for month in months:
            name = symbol+'-1m-'+month.strftime('%Y-%m')+'.zip'
            monthly = next((r/'monthly/klines'/symbol/'1m'/name for r in a.minute_root
                            if (r/'monthly/klines'/symbol/'1m'/name).exists()), None)
            if monthly is not None:
                tasks.append((symbol, monthly)); continue
            for day in pd.date_range(month, min(month+pd.offsets.MonthEnd(1), pd.Timestamp('2026-10-08'))):
                name = symbol+'-1m-'+str(day.date())+'.zip'
                daily = next((r/'daily/klines'/symbol/'1m'/name for r in a.minute_root
                              if (r/'daily/klines'/symbol/'1m'/name).exists()), None)
                if daily is not None: tasks.append((symbol, daily))
    data, proof, errors = [], [], []
    with ProcessPoolExecutor(a.workers) as pool:
        jobs = {pool.submit(extract, t): t for t in tasks}
        for k, f in enumerate(as_completed(jobs), 1):
            try:
                symbol, rows, evidence = f.result(); proof.append(evidence)
                data.extend({'symbol': symbol, **r} for r in rows)
            except Exception as exc: errors.append({'archive':str(jobs[f][1]),'reason':str(exc)[:180]})
            if k % 250 == 0: print(json.dumps({'phase':'execution_minutes','done':k,'total':len(tasks),'errors':len(errors)}),flush=True)
    if errors:
        write_json(root/'execution_minutes_errors.json', errors)
        raise ValueError('some existing official minute archives did not pass checksum/format checks')
    rows = pd.DataFrame(data)
    if rows.duplicated(['symbol','asof']).any(): raise ValueError('duplicate official execution minutes')
    wide = rows.pivot(index='asof',columns='symbol',values='price').sort_index()
    wide = wide.loc['2025-07-01':'2026-10-08']; wide.to_parquet(target)
    write_json(receipt, {'sha256':file_hash(target),'rows':int(wide.notna().sum().sum()),
        'benchmark_minutes':10,'window_end_minutes':60,'sources':proof,'missing_minutes_remain_missing':True})
    print(json.dumps({'phase':'execution_minutes_complete','archives':len(proof),'days':len(wide)}),flush=True)


if __name__ == '__main__': main()
