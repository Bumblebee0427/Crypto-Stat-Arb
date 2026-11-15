"""Public historical-hour Binance USDT spot membership for the daily selector.

Freeze one observation per UTC decision. Current exchange metadata only maps
names; membership is confirmed by the preceding closed hourly spot candle.
Network failures are unknown/excluded, never silently classified as no spot.
"""
import json,math
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from data_center.base_strategy import symbol_name
from data_center.contract import atomic_json,digest,timestamp

SCHEMA='select_coin.spot_membership.v1'
# Historical aliases verified against the research snapshot before this release.
ALIASES={'LUNA2USDT':'LUNAUSDT','RAYSOLUSDT':'RAYUSDT','DODOXUSDT':'DODOUSDT'}
API='https://api.binance.com/api/v3/'


def get(endpoint,params=None):
    url=API+endpoint+('?' + urlencode(params) if params else '')
    with urlopen(url,timeout=12) as response:
        return json.load(response)


def spot_symbol(symbol,listed):
    s=symbol_name(symbol)
    if s in listed:return s
    alias=ALIASES.get(s)
    if alias in listed:return alias
    # Binance lot-multiplier contracts retain the underlying spot base name.
    for prefix in ['1000000','10000','1000','100','1M']:
        if s.startswith(prefix) and s[len(prefix):] in listed:return s[len(prefix):]
    return None


def build(universe,day,now=None,fetch=get):
    now=now or datetime.now(timezone.utc)
    end=datetime.fromisoformat(day).replace(tzinfo=timezone.utc)+timedelta(days=1)
    if now<end:raise ValueError('spot membership hour is not closed')
    info=fetch('exchangeInfo')
    if not isinstance(info,dict) or not isinstance(info.get('symbols'),list) or not info['symbols']:
        raise ValueError('spot exchange metadata unavailable')
    # Include non-TRADING listings here: the closed-hour observation, rather
    # than a later halt/delisting status, determines formation membership.
    listed={r['symbol'] for r in info['symbols'] if r.get('quoteAsset')=='USDT' and r.get('isSpotTradingAllowed',True)}
    symbols=sorted({symbol_name(s) for s in universe['tradable_symbols']})
    mapping={s:spot_symbol(s,listed) for s in symbols};observed={};unknown={}
    near=end-timedelta(hours=1);start_ms=int(near.timestamp()*1000);end_ms=int(end.timestamp()*1000)-1
    def one(s):
        rows=fetch('klines',{'symbol':s,'interval':'1h','startTime':start_ms,'endTime':end_ms,'limit':1})
        if not isinstance(rows,list):raise ValueError('invalid spot hourly response')
        if not rows:return False
        row=rows[0]
        if len(row)<9 or int(row[0])!=start_ms or int(row[6])!=end_ms:
            raise ValueError('spot candle timestamp mismatch')
        close=float(row[4])
        return math.isfinite(close) and close>0 and int(row[8])>0
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs={pool.submit(one,s):s for s in sorted(set(mapping.values())-{None})}
        for future in as_completed(jobs):
            s=jobs[future]
            try:observed[s]=bool(future.result())
            except (ValueError,OSError,TypeError,KeyError,IndexError) as exc:unknown[s]=type(exc).__name__
    membership={s:(False if mapped is None else observed[mapped]) for s,mapped in mapping.items() if mapped not in unknown}
    excluded={s:'spot_hour_unavailable:'+unknown[mapped] for s,mapped in mapping.items() if mapped in unknown}
    return {'schema_version':SCHEMA,'signal_day':day,'asof':end.isoformat(),'observed_hour':near.isoformat(),
        'fetched_at':datetime.now(timezone.utc).isoformat(),'source':'binance_public_spot_exchangeInfo_and_closed_1h',
        'exchange_info_hash':digest(info),'mapping':mapping,'membership':membership,'excluded_symbols':excluded,
        'counts':{'requested':len(symbols),'spot':sum(membership.values()),'no_spot':sum(not v for v in membership.values()),'unknown':len(excluded)}}


def validate(value,day,now,symbols):
    end=datetime.fromisoformat(day).replace(tzinfo=timezone.utc)+timedelta(days=1)
    if (value.get('schema_version')!=SCHEMA or value.get('signal_day')!=day or timestamp(value['asof'])!=end
        or timestamp(value['observed_hour'])!=end-timedelta(hours=1)
        or not end<=timestamp(value['fetched_at'])<=now):raise ValueError('invalid spot membership observation')
    members=value.get('membership',{});excluded=value.get('excluded_symbols',{})
    if not isinstance(members,dict) or any(type(v)is not bool for v in members.values()):raise ValueError('invalid spot membership flags')
    if not set(symbols)<=set(members)|set(excluded):raise ValueError('incomplete spot membership coverage')
    return value


def prepare(root,universe,day):
    p=Path(root)/'inputs/spot_membership'/f'{day}.json'
    if p.exists():
        value=json.loads(p.read_text());return validate(value,day,datetime.now(timezone.utc),[symbol_name(s) for s in universe['tradable_symbols']])
    value=build(universe,day);atomic_json(p,value);return value
