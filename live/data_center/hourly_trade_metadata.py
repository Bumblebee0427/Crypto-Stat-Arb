"""ID-only closed-hour evidence for v3; no old factor calculation."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_center.contract import atomic_json

COLS = ('id', 'timestamp_ms', 'price', 'qty', 'maker')
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FIELDS = ('hour', 'source_size', 'source_mtime_ns', 'sequence_rows', 'trade_rows',
          'zero_value_sequence_rows', 'missing_trade_ids', 'first_trade_id',
          'last_trade_id', 'first_id_time_ms', 'last_id_time_ms',
          'first_exchange_time_ms', 'last_exchange_time_ms', 'time_monotonic_by_id')


def summarize_hour(source, day, hour):
    source=Path(source)
    before=source.stat()
    raw=pd.read_csv(source,header=None,names=COLS,
                    dtype={'id':'int64','timestamp_ms':'int64','price':'float64',
                           'qty':'float64','maker':'bool'})
    if raw.empty or raw[list(COLS)].isna().any().any():
        raise ValueError('empty or null hourly source')
    if not np.isfinite(raw[['id','timestamp_ms','price','qty']].to_numpy()).all():
        raise ValueError('nonfinite hourly source')
    zero=raw.price.eq(0)&raw.qty.eq(0)
    if (((raw.price<=0)|(raw.qty<=0))&~zero).any():
        raise ValueError('invalid economic hourly source')
    duplicate=raw[raw.duplicated('id',keep=False)]
    if not duplicate.empty and (duplicate.groupby('id')[list(COLS[1:])].nunique()>1).any().any():
        raise ValueError('conflicting duplicate hourly ID')
    raw=raw.drop_duplicates('id').sort_values('id',kind='stable')
    zero=raw.price.eq(0)&raw.qty.eq(0)
    start=pd.Timestamp(day,tz='UTC')+pd.Timedelta(hours=hour)
    times=pd.to_datetime(raw.timestamp_ms,unit='ms',utc=True)
    if not ((times>=start)&(times<start+pd.Timedelta(hours=1))).all():
        raise ValueError('source contains another UTC hour')
    ids=raw.id.to_numpy(dtype=np.int64);jumps=np.diff(ids)
    if (jumps<=0).any():raise ValueError('hourly IDs are not strictly increasing')
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('hourly source changed while reading')
    return {'day':day,'hour':hour,'metadata_code_sha256':CODE_SHA256,
            'source_size':after.st_size,'source_mtime_ns':after.st_mtime_ns,
            'sequence_rows':int(len(raw)),'trade_rows':int((~zero).sum()),
            'zero_value_sequence_rows':int(zero.sum()),
            'missing_trade_ids':int((jumps[jumps>1]-1).sum()),
            'first_trade_id':int(ids[0]),'last_trade_id':int(ids[-1]),
            'first_id_time_ms':int(raw.timestamp_ms.iloc[0]),
            'last_id_time_ms':int(raw.timestamp_ms.iloc[-1]),
            'first_exchange_time_ms':int(raw.timestamp_ms.min()),
            'last_exchange_time_ms':int(raw.timestamp_ms.max()),
            'time_monotonic_by_id':bool(raw.timestamp_ms.is_monotonic_increasing)}


def cached_hour(work, day, hour, symbol):
    work=Path(work);source=work/'hours'/f'{hour:02d}'/(symbol+'.csv')
    cache=work/'id_hours'/f'{hour:02d}'/(symbol+'.json')
    info=source.stat()
    if cache.exists():
        row=json.loads(cache.read_text())
        if (row.get('day')==day and row.get('hour')==hour
                and row.get('metadata_code_sha256')==CODE_SHA256
                and row.get('source_size')==info.st_size
                and row.get('source_mtime_ns')==info.st_mtime_ns):
            return row
    row=summarize_hour(source,day,hour)
    atomic_json(cache,row)
    return row


def refresh_hours(work, day, now):
    count=0
    for source in sorted((Path(work)/'hours').glob('*/*.csv')):
        hour=int(source.parent.name)
        end=datetime.fromisoformat(day).replace(tzinfo=timezone.utc)+timedelta(hours=hour+1)
        if now<end:continue
        cached_hour(work,day,hour,source.stem)
        count+=1
    return {'closed_hour_metadata':count,'factor_calculations':0}
