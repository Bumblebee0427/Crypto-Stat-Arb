#!/usr/bin/env python3
"""Refresh public inputs, compute the daily v3 overlay and atomically deliver it.

Machine addresses and SSH key paths are read from a private local JSON config.
Exchange credentials remain on the trade host. No order endpoint is called.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.contracts import write_json


def remote(config, host, program, *, sudo=False):
    command = ['ssh', '-i', config['ssh_key'], '-o', 'BatchMode=yes',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
        'ubuntu@'+host, 'sudo -n /usr/bin/python3 -' if sudo else 'python3 -']
    result = subprocess.run(command, input=program, text=True, capture_output=True, timeout=90)
    if result.returncode:
        # Do not serialize a signed-API exception URL or a private environment.
        raise RuntimeError('remote read/write failed: '+str(result.returncode))
    return json.loads(result.stdout)


def signal(config):
    path = config['seoul_root']+'/data/base_signal_v1/top50/latest.json'
    return remote(config, config['seoul_host'],
        'from pathlib import Path\nprint(Path('+repr(path)+').read_text())\n')


def observation(config, source):
    program = '''
from pathlib import Path
from datetime import datetime,timezone
import json,os,shlex,sys
root=Path(TRADE_ROOT)
sys.path[:0]=[str(root/'signal_extension'),str(root/'tools')]
for line in Path(ENV_FILE).read_text().splitlines():
 line=line.strip()
 if line and not line.startswith('#') and '=' in line:
  k,v=line.removeprefix('export ').split('=',1);v=shlex.split(v)
  if v:os.environ[k]=v[0]
from data_center.live_executor import BinancePortfolioTransport,account_check
from data_center.daily_rebalance import signed_positions
client=BinancePortfolioTransport();audit=client.audit()
ledger=json.loads((root/'data/top50_daily/ledger.json').read_text())
positions=signed_positions(ledger['positions']);account,_=account_check(audit,positions)
print(json.dumps({'checked_at':datetime.now(timezone.utc).isoformat(),
 'base_snapshot_id':BASE_SIGNAL,'equity':account['equity'],
 'positions':{s:str(q) for s,q in positions.items()},'marks':client.prices()}))
'''.replace('TRADE_ROOT', repr(config['trade_root'])).replace('ENV_FILE', repr(config['env_file'])).replace('BASE_SIGNAL', repr(source['snapshot_id']))
    return remote(config, config['trade_host'], program, sudo=True)


def settlements(root, symbols, boundary):
    rows = []
    def one(symbol):
        params = {'symbol': symbol, 'startTime': int((boundary-timedelta(days=7)).timestamp()*1000),
                  'endTime': int(boundary.timestamp()*1000)-1, 'limit': 1000}
        url = 'https://fapi.binance.com/fapi/v1/fundingRate?'+urlencode(params)
        with urlopen(Request(url, headers={'User-Agent': 'select-coin-live-research/1.0'}), timeout=25) as r:
            raw = json.load(r)
        if not isinstance(raw, list): raise ValueError('funding response unavailable')
        result = []
        raw.sort(key=lambda x: x['fundingTime'])
        for k, row in enumerate(raw):
            hours = (row['fundingTime']-raw[k-1]['fundingTime'])/3_600_000 if k else np.nan
            result.append({'symbol': symbol, 'time': pd.to_datetime(row['fundingTime'], unit='ms'),
                'rate': float(row['fundingRate']), 'interval_hours': hours if hours in [1, 4, 8] else np.nan,
                'source': 'official_binance_settled_funding'})
        return result
    missing = []
    with ThreadPoolExecutor(4) as pool:
        jobs = {pool.submit(one, s): s for s in symbols}
        for f in as_completed(jobs):
            try: rows.extend(f.result())
            except Exception: missing.append(jobs[f])
    frame = pd.DataFrame(rows, columns=['symbol', 'time', 'rate', 'interval_hours', 'source'])
    frame.to_parquet(root/'settlements.parquet', index=False)
    return missing


def archive_v3_alpha(root, source, boundary):
    """Collect online v3 features and first-minute VWAP labels; no v2 reuse."""
    features_path, prices_path = root/'v3_features.parquet', root/'v3_entry_prices.parquet'
    past = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
    date = pd.Timestamp(boundary).tz_convert('UTC').tz_localize(None)
    rows = []
    for symbol, weight in source['allocations'].items():
        if not weight: continue
        side = 1 if weight > 0 else -1; factors = source['factors'][symbol]
        rows.append({'asof':date, 'symbol':symbol, 'side':side,
            'entry_at':pd.Timestamp(source['generated_at']).tz_convert('UTC').tz_localize(None).ceil('min'),
            'main':factors['main'], 'small':factors['small_long' if side > 0 else 'small_short'],
            'formula_version':source['formula_version']})
    current = pd.DataFrame(rows)
    for feature in ['main', 'small']:
        current[feature+'_rank'] = current.groupby('side')[feature].rank(pct=True)*2-1
    features = pd.concat([past, current], ignore_index=True).drop_duplicates(['asof','symbol'], keep='last')
    features.to_parquet(features_path, index=False)
    prices = pd.read_parquet(prices_path) if prices_path.exists() else pd.DataFrame(columns=['asof','symbol','price'])
    # Labels start after the signal is published. Never use the earlier UTC
    # midnight price when the live factor only becomes available minutes later.
    known={(pd.Timestamp(r.asof),r.symbol) for r in prices.itertuples()}
    jobs=[];observed_now=pd.Timestamp.now(tz='UTC').tz_localize(None)
    for day, group in features.groupby('asof'):
        if day < date-pd.Timedelta(days=3): continue
        execution=pd.Timestamp(group.entry_at.iloc[0])
        if execution+pd.Timedelta(minutes=1) > observed_now: continue
        wanted=set(group.symbol)
        wanted.update(features.loc[features['asof'] == day-pd.Timedelta(days=1),'symbol'])
        jobs.extend((day,s,execution) for s in wanted if (pd.Timestamp(day),s) not in known)
    def entry(job):
        day,symbol,execution=job
        start=int(execution.tz_localize('UTC').timestamp()*1000)
        params = {'symbol':symbol, 'interval':'1m', 'startTime':start,
                  'endTime':start+59999, 'limit':1}
        with urlopen(Request('https://fapi.binance.com/fapi/v1/klines?'+urlencode(params),
            headers={'User-Agent':'select-coin-v3-labels/1.0'}),timeout=20) as r: value=json.load(r)
        if not value or value[0][0] != params['startTime'] or float(value[0][5]) <= 0:
            raise ValueError('first execution minute unavailable')
        return {'asof':day,'symbol':symbol,'price':float(value[0][7])/float(value[0][5])}
    entries=[]
    with ThreadPoolExecutor(4) as pool:
        for f in as_completed([pool.submit(entry,job) for job in jobs]):
            try: entries.append(f.result())
            except Exception: pass  # Missing labels stay missing and cannot train alpha.
    prices = pd.concat([prices,pd.DataFrame(entries)],ignore_index=True).drop_duplicates(['asof','symbol'],keep='last')
    prices['asof'] = pd.to_datetime(prices['asof'])
    prices['price'] = pd.to_numeric(prices['price'])
    prices.to_parquet(prices_path,index=False)
    panel = features.merge(prices.rename(columns={'price':'entry_price'}),on=['asof','symbol'],how='left')
    exits = prices.rename(columns={'price':'exit_price'}).copy();exits['asof']-=pd.Timedelta(days=1)
    panel = panel.merge(exits,on=['asof','symbol'],how='left')
    panel['future_return'] = panel.exit_price/panel.entry_price-1
    panel['label_available_at'] = panel['asof']+pd.Timedelta(days=1,hours=1)
    panel.to_parquet(root/'alpha_panel.parquet',index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--skip-refresh', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text()); root = Path(config['data_root'])
    lock = (root/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    now = datetime.now(timezone.utc); boundary = now.replace(hour=0, minute=0, second=0, microsecond=0)
    deadline = boundary+timedelta(minutes=8)
    if not args.preview and now >= deadline:
        raise ValueError('daily producer started outside the bounded opening window')
    if not args.skip_refresh:
        command = [sys.executable, str(ROOT/'tools/update_live_optimizer_data.py'), '--root', str(root),
                   '--asof', str(boundary.date()), '--workers', '4']
        for key, flag in [('legacy_cap_root', '--legacy-cap-root'), ('seed_closes', '--seed-closes'), ('seed_quotes', '--seed-quotes')]:
            if config.get(key): command += [flag, config[key]]
        subprocess.run(command, check=True, timeout=450)
    # The research frames retain the original dashed names; live symbols are exchange IDs.
    returns = pd.read_parquet(root/'returns.parquet')
    returns.columns = [str(s).replace('-', '') for s in returns.columns]
    if returns.columns.has_duplicates: raise ValueError('ambiguous live return symbols')
    returns.to_parquet(root/'returns_live.parquet')
    inputs = json.loads((root/'inputs.json').read_text()); inputs['returns'] = 'returns_live.parquet'
    write_json(root/'inputs_live.json', inputs)
    while True:
        source = signal(config)
        fresh = source.get('window_end') == boundary.isoformat() and source.get('status') == 'complete'
        if fresh: break
        if args.preview or datetime.now(timezone.utc) >= deadline: raise ValueError('fresh complete base signal unavailable')
        time.sleep(3)
    symbols = [s for s, w in source['allocations'].items() if w]
    missing = settlements(root, symbols, boundary)
    archive_v3_alpha(root, source, boundary)
    inputs['alpha_panel'] = 'alpha_panel.parquet'
    write_json(root/'inputs_live.json', inputs)
    # Fetch the account after the other data; its evidence expires after 60 seconds.
    obs = observation(config, source)
    write_json(root/'signal.json', source); write_json(root/'observation.json', obs)
    target = root/('target_'+str(boundary.date())+'.json')
    command = [sys.executable, str(ROOT/'tools/build_live_optimizer_artifact.py'),
        '--inputs', str(root/'inputs_live.json'), '--signal', str(root/'signal.json'),
        '--observation', str(root/'observation.json'), '--out', str(target)]
    if args.preview: command += ['--preview']
    subprocess.run(command, check=True, timeout=45)
    artifact = json.loads(target.read_text())
    if not args.preview:
        destination = config['trade_root']+'/data/top50_daily/optimizer/'+target.name
        program = '''
from pathlib import Path
import json,os,tempfile,pwd
path=Path(DESTINATION);path.parent.mkdir(parents=True,exist_ok=True)
payload=PAYLOAD
fd,tmp=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
owner=pwd.getpwnam('ubuntu');os.fchown(fd,owner.pw_uid,owner.pw_gid)
with os.fdopen(fd,'w') as f:json.dump(payload,f,allow_nan=False);f.flush();os.fsync(f.fileno())
os.replace(tmp,path)
print(json.dumps({'status':'delivered','artifact_id':payload['snapshot_id']}))
'''.replace('DESTINATION', repr(destination)).replace('PAYLOAD', repr(artifact))
        remote(config, config['trade_host'], program, sudo=True)
    summary = {'status': 'preview_ready' if args.preview else 'delivered',
        'model': artifact['model'], 'asof': source['window_end'], 'artifact_id': artifact['snapshot_id'],
        'allocations': len(artifact['allocations']), 'alpha_status': artifact['alpha_status'],
        'funding_missing_symbols': missing, 'checked_at': datetime.now(timezone.utc).isoformat(),
        'orders_submitted': False}
    write_json(root/'last_run.json', summary); print(json.dumps(summary))


if __name__ == '__main__': main()
