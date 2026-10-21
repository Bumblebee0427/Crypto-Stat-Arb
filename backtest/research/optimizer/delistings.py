"""Bound automatic settlement using official minute index OHLC, never last trades.

If the settlement is the mean of 1,800 one-second index observations, its value
lies between the means of the thirty minute lows/highs. This is an interval,
not a recovered official settlement price. Such samples stay out of alpha fit.
"""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from .contracts import file_hash,digest


def index_settlement_bounds(path, symbol, settled_at, announcement_url, rule_url):
    path=Path(path);expected=path.with_name(path.name+'.CHECKSUM').read_text().split()[0]
    if file_hash(path)!=expected:raise ValueError('index archive checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if n.endswith('.csv')]
        if len(names)!=1:raise ValueError('ambiguous index archive')
        f=pd.read_csv(archive.open(names[0]),header=None,usecols=[0,1,2,3,4,8])
    f.columns=['time','open','high','low','close','count'];f=f.apply(pd.to_numeric,errors='coerce').dropna()
    t=f.time.astype('int64');f['time']=pd.to_datetime(t,unit='us' if t.max()>10**14 else 'ms')
    end=pd.Timestamp(settled_at);start=end-pd.Timedelta(minutes=30)
    f=f.loc[f.time.ge(start)&f.time.lt(end)].set_index('time').sort_index()
    required=pd.date_range(start,end-pd.Timedelta(minutes=1),freq='min')
    if not f.index.equals(required) or not f['count'].eq(60).all():raise ValueError('incomplete 1800 index observations')
    if (not np.isfinite(f.to_numpy()).all() or (f[['open','high','low','close']]<=0).any().any()
            or (f.low>f[['open','close']].min(axis=1)).any() or (f.high<f[['open','close']].max(axis=1)).any()):
        raise ValueError('invalid index OHLC')
    body={'symbol':symbol,'settled_at':str(end),'window_start':str(start),'lower':float(f.low.mean()),
        'upper':float(f.high.mean()),'index_observations':1800,'archive_sha256':expected,
        'announcement_url':announcement_url,'rule_url':rule_url,'exact_settlement_price_known':False}
    return {**body,'event_id':digest(body)}


def load_delistings(root):
    path=Path(root)/'delisting_events.json'
    if not path.exists():return {}
    events={}
    for row in json.loads(path.read_text())['events']:
        body={k:v for k,v in row.items() if k!='event_id'}
        if row['event_id']!=digest(body) or not (0<row['lower']<=row['upper']):raise ValueError('invalid settlement interval')
        if row['index_observations']!=1800 or row['exact_settlement_price_known'] is not False:raise ValueError('unverified index settlement method')
        key=(pd.Timestamp(row['settled_at']).normalize(),row['symbol'])
        if key in events:raise ValueError('duplicate delisting event')
        events[key]=row
    return events
