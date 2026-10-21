"""Raw individual sequence -> corrected daily factors; conservative quality.

Persist placeholders for sequence checks, exclude ONLY zero/zero values from
economic events. Source scope, full-day head/tail and unresolved IDs stay gated.
"""
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from data_center.base_strategy import FORMULA_VERSION,SOURCE_VERSION
from data_center.reference_formula_v2 import minute_features_from_events,daily_factors


def compute_daily(raw,symbol,day,manifest):
    d=raw.copy()
    required=['id','timestamp_ms','price','qty','maker']
    if d.empty or d[required].isna().any().any():raise ValueError('empty/null live raw input')
    if not d.maker.isin([True,False]).all():raise ValueError('invalid side')
    d['price']=pd.to_numeric(d.price);d['qty']=pd.to_numeric(d.qty)
    if not np.isfinite(d[['id','timestamp_ms','price','qty']].to_numpy()).all():raise ValueError('nonfinite raw values')
    if ((d[['id','timestamp_ms']]%1)!=0).any().any():raise ValueError('fractional ID/time')
    zeros=(d.price==0)&(d.qty==0)
    if (((d.price<=0)|(d.qty<=0))&~zeros).any():raise ValueError('invalid non-placeholder trade')
    duplicate=d[d.duplicated('id',keep=False)]
    if not duplicate.empty and (duplicate.groupby('id')[required[1:]].nunique()>1).any().any():raise ValueError('conflicting trade ID')
    d=d.drop_duplicates('id').sort_values('id',kind='stable')
    times=pd.to_datetime(d.timestamp_ms,unit='ms',utc=True)
    if not times.dt.normalize().eq(pd.Timestamp(day,tz='UTC')).all():raise ValueError('exchange-day mismatch')
    gaps=np.diff(d.id.to_numpy(dtype=np.int64));missing=int((gaps[gaps>1]-1).sum())
    economic=d[(d.price>0)&(d.qty>0)].copy()
    economic['time']=pd.to_datetime(economic.timestamp_ms,unit='ms',utc=True)
    economic['side']=np.where(economic.maker,'sell','buy');economic['notional']=economic.price*economic.qty
    events=economic.groupby(['time','side'],sort=True).agg(qty=('qty','sum'),notional=('notional','sum')).reset_index()
    row=daily_factors(minute_features_from_events(events),day)
    row.update(symbol=symbol,day=day,formula_version=FORMULA_VERSION,source_version=SOURCE_VERSION,status='final',
        source_complete=False,source_certified=False,quality_blockers=['public_source_scope_not_certified','full_day_boundaries_not_certified']+
            (['unresolved_individual_trade_id_gaps'] if missing else []),missing_trade_ids=missing,
        trade_rows=len(economic),sequence_rows=len(d),zero_value_sequence_rows=len(d)-len(economic),event_rows=len(events),
        first_trade_id=int(d.id.iloc[0]),last_trade_id=int(d.id.iloc[-1]),
        first_exchange_time_ms=int(d.timestamp_ms.min()),last_exchange_time_ms=int(d.timestamp_ms.max()),
        time_monotonic_by_id=bool(d.timestamp_ms.is_monotonic_increasing),
        partition_basis='exchange_time_utc_day',available_at=datetime.now(timezone.utc).isoformat(),
        producer='individual_live_daily_v1',input_manifest_hash=manifest)
    return row
