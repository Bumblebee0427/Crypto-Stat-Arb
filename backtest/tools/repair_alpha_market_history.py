#!/usr/bin/env python3
"""Repair explicit historical hourly gaps from checksummed official minute archives.

This repairs observations, not historical universe membership. Existing good
observations are preserved. Current exchange status never filters past samples.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.optimizer.contracts import file_hash,write_json,digest


def hourly(task):
    symbol,path=task;path=Path(path)
    expected=path.with_name(path.name+'.CHECKSUM').read_text().split()[0]
    if file_hash(path)!=expected:raise ValueError('minute checksum mismatch: '+str(path))
    with zipfile.ZipFile(path) as archive:
        members=[n for n in archive.namelist() if n.endswith('.csv')]
        if len(members)!=1:raise ValueError('ambiguous minute archive')
        f=pd.read_csv(archive.open(members[0]),header=None,usecols=[0,4,5,7])
    f.columns=['time','close','qty','quote_volume'];f=f.apply(pd.to_numeric,errors='coerce').dropna()
    t=f.time.astype('int64');f['time']=pd.to_datetime(t,unit='us' if t.max()>10**14 else 'ms')
    if f.time.duplicated().any():raise ValueError('duplicate minute')
    f=f.sort_values('time');f['hour']=f.time.dt.floor('h')
    h=f.groupby('hour').agg(close=('close','last'),quote_volume=('quote_volume','sum'),count=('time','size'))
    h=h.loc[(h['count']==60)&h.close.gt(0)&h.quote_volume.ge(0)].drop(columns='count')
    return symbol,h,{'path':str(path),'sha256':expected}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True);p.add_argument('--exchange-info',type=Path,required=True)
    p.add_argument('--minute-root',type=Path,action='append',required=True);p.add_argument('--symbol',action='append',required=True)
    a=p.parse_args();root=a.history;metadata={r['symbol']:r for r in json.loads(a.exchange_info.read_text())['symbols']}
    tasks=[]
    for s in a.symbol:
        for m in pd.date_range('2025-06-01','2026-10-01',freq='MS'):
            name=s+'-1m-'+m.strftime('%Y-%m')+'.zip'
            path=next((r/'monthly/klines'/s/'1m'/name for r in a.minute_root if (r/'monthly/klines'/s/'1m'/name).exists()),None)
            if path is not None:tasks.append((s,path));continue
            for d in pd.date_range(m,min(m+pd.offsets.MonthEnd(1),pd.Timestamp('2026-10-08'))):
                name=s+'-1m-'+str(d.date())+'.zip'
                path=next((r/'daily/klines'/s/'1m'/name for r in a.minute_root if (r/'daily/klines'/s/'1m'/name).exists()),None)
                if path is not None:tasks.append((s,path))
    proof=[];frames={}
    with ProcessPoolExecutor(4) as pool:
        for s,h,e in pool.map(hourly,tasks):frames.setdefault(s,[]).append(h);proof.append(e)
    identity=digest({'sources':proof,'exchange_info':file_hash(a.exchange_info)})
    receipt=root/'market_repairs.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('identity')==identity and (root/'repaired_execution_close.parquet').exists():
        print('reuse audited market repairs',flush=True);return
    market=pd.read_parquet(root/'market_selection.parquet').set_index(['symbol','asof'])
    closes=pd.read_parquet(root/'daily_closes.parquet');quotes=pd.read_parquet(root/'daily_quotes.parquet')
    trend=market.reset_index().groupby('asof').btc.first()
    repaired=[];details=[];hourly_close={}
    for s,parts in frames.items():
        h=pd.concat(parts).sort_index()
        if h.index.has_duplicates:raise ValueError('duplicate official hour')
        h=h.reindex(pd.date_range(h.index.min(),h.index.max(),freq='h'))
        # A truncated minute archive is never treated as a newly listed asset.
        onboard=pd.to_datetime(metadata[s]['onboardDate'],unit='ms')
        first=onboard.ceil('h'); age=pd.Series((h.index-first)/pd.Timedelta(hours=1),index=h.index).clip(lower=0,upper=599)
        count=h.quote_volume.notna().shift(1,fill_value=False).rolling(599,min_periods=1).sum()
        vol=h.quote_volume.shift(1).rolling(599,min_periods=1).mean()
        for t in h.index[(h.index.hour==23)&(h.index<=pd.Timestamp('2026-10-07 23:00'))]:
            if pd.isna(h.at[t,'close']) or count.at[t]!=age.at[t] or age.at[t]<1:continue
            d=t+pd.Timedelta(hours=1)
            repaired.append({'symbol':s,'asof':d,'quote_volume_mean':vol.at[t],
                'btc':trend.get(d,np.nan) if first<=t-pd.Timedelta(hours=721) else np.nan,
                'onboard_day':onboard.normalize()})
        hourly_close[s]=h.close
        c=h.close.resample('D').last();c.index+=pd.Timedelta(days=1)
        q=h.quote_volume.resample('D').sum(min_count=24);q.index+=pd.Timedelta(days=1)
        closes[s]=closes[s].combine_first(c) if s in closes else c
        quotes[s]=quotes[s].combine_first(q) if s in quotes else q
        details.append({'symbol':s,'complete_hour_rows':int(h.close.notna().sum()),'onboard':str(onboard),
            'first_archive_hour':str(h.index.min()),'last_archive_hour':str(h.index.max())})
    fixed=pd.DataFrame(repaired).set_index(['symbol','asof'])
    # Official complete observations take precedence; unobserved gaps stay gaps.
    market=fixed.combine_first(market).sort_index()
    market.reset_index().to_parquet(root/'market_selection.parquet',index=False)
    closes.to_parquet(root/'daily_closes.parquet');quotes.to_parquet(root/'daily_quotes.parquet')
    pd.DataFrame(hourly_close).sort_index().to_parquet(root/'repaired_execution_close.parquet')
    write_json(receipt,{'identity':identity,'details':details,'repaired_selection_rows':len(fixed),'sources':proof,
        'current_status_not_used':True,'missing_prices_not_filled':True})
    print(json.dumps({'market_repair_symbols':len(frames),'complete_selection_rows':len(fixed)}),flush=True)


if __name__=='__main__':main()
