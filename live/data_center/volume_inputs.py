"""Freeze the TOP50 quote-volume ranking before the UTC day boundary."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from data_center.base_strategy import symbol_name
from data_center.contract import atomic_json,digest,timestamp


def volume_row(symbol,day,rows,first_candle,now):
    end=timestamp(day+'T00:00:00+00:00')+timedelta(days=1)
    begin=max(end-timedelta(hours=600),timestamp(first_candle))
    # Binance listing may begin inside an hour; metadata must be candle BEGIN.
    if begin.minute or begin.second or begin.microsecond:raise ValueError('unaligned first candle')
    expected=int((end-begin).total_seconds()/3600)
    if not 1<=expected<=600 or end>now:raise ValueError('invalid closed volume window')
    selected=[r for r in rows if begin.timestamp()*1000<=int(r[0])<end.timestamp()*1000]
    times=[int(r[0]) for r in selected]
    expected_times=[int((begin+timedelta(hours=i)).timestamp()*1000) for i in range(expected)]
    if times!=expected_times:raise ValueError('missing/duplicate/unordered hourly volume candles')
    values=[]
    for r in selected:
        if int(r[6])!=int(r[0])+3600000-1 or int(r[6])>=now.timestamp()*1000:raise ValueError('unclosed/invalid kline')
        v=float(r[7])
        if not math.isfinite(v) or v<0:raise ValueError('invalid quote volume')
        values.append(v)
    return {'symbol':symbol,'day':day,'hours':600,'observed_hours':len(values),
            'window_start':begin.isoformat(),'window_end':end.isoformat(),
            'quote_volume_mean':math.fsum(values)/len(values),'complete':True,
            'source':'binance_public_fapi_1h_quote_volume','available_at':now.isoformat(),
            'raw_candles_hash':digest(selected),'closed_candles':selected}


def public_klines(symbol,begin,end,limit):
    params={'symbol':symbol,'interval':'1h','startTime':int(begin.timestamp()*1000),
            'endTime':int(end.timestamp()*1000)-1,'limit':limit}
    with urlopen('https://fapi.binance.com/fapi/v1/klines?'+urlencode(params),timeout=20) as q:
        return json.load(q)


def warm_row(symbol,day,rows,first_candle,now):
    end=timestamp(day+'T00:00:00+00:00')+timedelta(days=1)
    through=end-timedelta(hours=1)
    begin=max(end-timedelta(hours=600),timestamp(first_candle))
    expected=int((through-begin).total_seconds()/3600)
    if not 1<=expected<=599 or now<through:raise ValueError('warm window unavailable')
    times=[int(r[0]) for r in rows]
    if times!=[int((begin+timedelta(hours=i)).timestamp()*1000) for i in range(expected)]:
        raise ValueError('missing warm volume hour')
    if any(int(r[6])!=int(r[0])+3600000-1 or int(r[6])>=now.timestamp()*1000
           or not math.isfinite(float(r[7])) or float(r[7])<0 for r in rows):
        raise ValueError('invalid/unclosed warm volume hour')
    return {'symbol':symbol,'day':day,'window_end':end.isoformat(),
            'first_candle':first_candle,'through':through.isoformat(),
            'closed_candles':rows,'raw_candles_hash':digest(rows),
            'available_at':now.isoformat(),'status':'warm_599_hours'}


def freeze_warm(symbol,day,warm,first,now):
    end=timestamp(day+'T00:00:00+00:00')+timedelta(days=1)
    through=end-timedelta(hours=1)
    if (warm.get('symbol')!=symbol or warm.get('day')!=day
            or warm.get('window_end')!=end.isoformat()
            or warm.get('through')!=through.isoformat()
            or warm.get('first_candle')!=first
            or warm.get('raw_candles_hash')!=digest(warm.get('closed_candles'))):
        raise ValueError('stale or corrupt warm volume input')
    checked=warm_row(symbol,day,warm['closed_candles'],first,now)
    if checked['raw_candles_hash']!=warm['raw_candles_hash']:
        raise ValueError('warm volume candles changed during validation')
    rows=checked['closed_candles']
    return {'symbol':symbol,'day':day,'hours':599,'observed_hours':len(rows),
            'window_start':max(end-timedelta(hours=600),timestamp(first)).isoformat(),
            'window_end':through.isoformat(),'target_window_end':end.isoformat(),
            'volume_lag_hours':1,
            'quote_volume_mean':math.fsum(float(r[7]) for r in rows)/len(rows),
            'complete':True,'source':'binance_public_fapi_1h_quote_volume_preclose_frozen',
            'available_at':now.isoformat(),'raw_candles_hash':checked['raw_candles_hash'],
            'closed_candles':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--universe',type=Path,required=True)
    p.add_argument('--max-jobs',type=int,default=30);a=p.parse_args()
    now=datetime.now(timezone.utc);end=now.replace(hour=0,minute=0,second=0,microsecond=0)
    day=(end-timedelta(days=1)).date().isoformat()
    universe=json.loads(a.universe.read_text());meta=json.loads((a.root/'inputs/btc_hourly.json').read_text())
    symbols=[symbol_name(raw) for raw in universe['tradable_symbols']
             if symbol_name(raw) not in ('BTCUSDT','ETHUSDT')]
    if now.hour==23:
        warm_day=end.date().isoformat();warm_end=end+timedelta(days=1)
        through=warm_end-timedelta(hours=1)
        warmed=0;warm_errors={}
        for s in symbols:
            path=a.root/'inputs/volume25_warm'/warm_day/(s+'.json')
            if path.exists():continue
            if warmed>=a.max_jobs:break
            warmed+=1
            try:
                first=meta['first_candle_by_symbol'][s]
                begin=max(warm_end-timedelta(hours=600),timestamp(first))
                row=warm_row(s,warm_day,public_klines(s,begin,through,1000),first,
                             datetime.now(timezone.utc))
                atomic_json(path,row)
            except Exception as exc:
                warm_errors[s]=type(exc).__name__+': '+str(exc)
                if getattr(exc,'code',0) in (418,429):break
            time.sleep(.15)
        frozen=0
        for s in symbols:
            path=a.root/'inputs/volume25'/warm_day/(s+'.json')
            warm_path=a.root/'inputs/volume25_warm'/warm_day/(s+'.json')
            if path.exists() or not warm_path.exists():continue
            try:
                first=meta['first_candle_by_symbol'][s]
                atomic_json(path,freeze_warm(s,warm_day,json.loads(warm_path.read_text()),
                                             first,datetime.now(timezone.utc)))
                frozen+=1
            except Exception as exc:
                warm_errors[s]=type(exc).__name__+': '+str(exc)
        available=len(list((a.root/'inputs/volume25'/warm_day).glob('*.json')))
        atomic_json(a.root/'volume25_status.json',{'updated_at':datetime.now(timezone.utc).isoformat(),
                    'day':warm_day,'attempted':warmed,'computed_this_run':frozen,
                    'available_files':available,'errors':warm_errors,'submitted_to_exchange':False})
        print(json.dumps({'day':warm_day,'warm_attempted':warmed,'frozen':frozen,
                          'frozen_available':available,
                          'warm_available':len(list((a.root/'inputs/volume25_warm'/warm_day).glob('*.json'))),
                          'errors':warm_errors}))
        return
    errors={};computed=0;attempted=0
    warm_jobs=[];cold_jobs=[]
    for s in symbols:
        path=a.root/'inputs/volume25'/day/(s+'.json')
        if path.exists():continue
        warm_path=a.root/'inputs/volume25_warm'/day/(s+'.json')
        (warm_jobs if warm_path.exists() else cold_jobs).append((s,path,warm_path))
    for s,path,warm_path in warm_jobs[:a.max_jobs]:
        attempted+=1
        try:
            first=meta['first_candle_by_symbol'][s]
            warm=json.loads(warm_path.read_text())
            atomic_json(path,freeze_warm(s,day,warm,first,datetime.now(timezone.utc)))
            computed+=1
        except Exception as exc:errors[s]=type(exc).__name__+': '+str(exc)
    for s,path,_ in cold_jobs[:max(0,a.max_jobs-attempted)]:
        attempted+=1
        try:
            first=meta['first_candle_by_symbol'][s]
            begin=max(end-timedelta(hours=600),timestamp(first))
            through=end-timedelta(hours=1)
            rows=public_klines(s,begin,through,1000)
            warm=warm_row(s,day,rows,first,datetime.now(timezone.utc))
            atomic_json(path,freeze_warm(s,day,warm,first,datetime.now(timezone.utc)))
            computed+=1
        except Exception as exc:
            errors[s]=type(exc).__name__+': '+str(exc)
            if getattr(exc,'code',0) in (418,429):break
        time.sleep(.15)
    atomic_json(a.root/'volume25_status.json',{'updated_at':now.isoformat(),'day':day,'attempted':attempted,'computed_this_run':computed,
                  'available_files':len(list((a.root/'inputs/volume25'/day).glob('*.json'))),'errors':errors,'submitted_to_exchange':False})
    print(json.dumps({'day':day,'computed':computed,'errors':errors}))


if __name__=='__main__':main()
