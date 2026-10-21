"""Audit user-authorized <=10% tolerance; never enables execution or edits factors.

Unknown ID slots are pessimistically counted as possible economic trades.
This interval bound is NOT a full-day rate unless the day has a bounded source.
Published checksummed official daily files anchor the archive reference, not
agreement with historical Tardis capture or proof of exchange-wide completeness.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import re
from urllib.parse import quote

from data_center.base_strategy import FORMULA_VERSION,SOURCE_VERSION,symbol_name
from data_center.contract import atomic_json,digest,timestamp

MAX_MISSING_FRACTION=.10
POLICY_VERSION='observed_interval_missing_id_lte_10pct_no_next_anchor_20260927_v2'


def integer(value):
    return isinstance(value,int) and not isinstance(value,bool) and value>=0


def archive_reference(row,symbol,day):
    expected='https://data.binance.vision/data/futures/um/daily/trades/'+quote(symbol,safe='')+'/'+quote(f'{symbol}-trades-{day}.zip',safe='')
    return (row.get('symbol')==symbol and row.get('day')==day and row.get('archive_url')==expected
            and bool(re.fullmatch('[a-f0-9]{64}',str(row.get('archive_sha256',''))))
            and row.get('partition_basis')=='exchange_time_utc_day')


def anchor_id(anchor,symbol,day,edge):
    if not isinstance(anchor,dict) or anchor.get('symbol')!=symbol or anchor.get('day')!=day:return None
    ident=anchor.get('id')
    if not integer(ident):return None
    if anchor.get('kind')=='checksummed_archive_edge':
        evidence=anchor.get('archive_evidence',{})
        if not archive_reference(evidence,symbol,day):return None
        if evidence.get('last_trade_id' if edge=='previous' else 'first_trade_id')!=ident:return None
    elif anchor.get('kind') in ('observed_neighbor_trade','checkpointed_neighbor_trade'):
        ms=anchor.get('timestamp_ms')
        try:
            if not integer(ms) or datetime.fromtimestamp(ms/1000,timezone.utc).date().isoformat()!=day:return None
        except (OverflowError,OSError,ValueError):return None
        if anchor['kind']=='observed_neighbor_trade':
            if not re.fullmatch('[a-f0-9]{64}',str(anchor.get('raw_segment_sha256',''))):return None
        else:
            if not re.fullmatch('[a-f0-9]{64}',str(anchor.get('raw_record_sha256',''))):return None
            try:
                if (anchor.get('capture_status')!='running'
                        or timestamp(anchor['capture_checkpoint_at']).timestamp()*1000 < ms):return None
            except (KeyError,ValueError,TypeError):return None
    else:return None
    return ident


def assess(row,symbol,day,now,expected_formula=FORMULA_VERSION):
    blockers=[]
    result={'symbol':symbol,'day':day,'missing_fraction_upper_bound':None,
            'decision_interval_fraction_upper_bound':None,
            'internal_id_fraction_upper_bound':None,'bounded_day_reference':False,
            'passes_tolerance':False,'live_ready':False}
    if not row:return {**result,'blockers':['missing_daily_input']}
    if (row.get('symbol')!=symbol or row.get('day')!=day or row.get('status')!='final'
        or row.get('formula_version')!=expected_formula or row.get('source_version')!=SOURCE_VERSION):
        blockers.append('daily_identity_version_invalid')
    try:
        if timestamp(row['available_at'])>now:blockers.append('future_input')
    except (KeyError,ValueError,TypeError):blockers.append('invalid_input_timestamp')
    n,gap=row.get('trade_rows'),row.get('missing_trade_ids')
    count_ok=integer(n) and n>0 and integer(gap)
    if count_ok:
        sequence=row.get('sequence_rows',n)
        first,last=row.get('first_trade_id'),row.get('last_trade_id')
        zero=row.get('zero_value_sequence_rows',0)
        if (not all(integer(v) for v in (sequence,first,last,zero)) or last<first
            or last-first+1!=sequence+gap or sequence!=n+zero):count_ok=False
    if not count_ok:blockers.append('invalid_or_inconsistent_sequence_counts')
    else:
        result.update(observed_economic_trades=n,unexplained_id_slots=gap,
                      internal_id_fraction_upper_bound=gap/(n+gap))
        if gap/(n+gap)>MAX_MISSING_FRACTION:blockers.append('possible_missing_fraction_above_10pct')
    archived=archive_reference(row,symbol,day)
    # An absent next-day trade ID leaves the day's trailing edge unbounded.
    # The user permits a live decision from a fully sealed capture interval;
    # report its bound separately and never call it a full-day loss rate.
    result['bounded_day_reference']=archived
    result['reference_kind']='checksummed_official_daily_archive' if archived else 'unbounded_live_interval'
    if archived and count_ok:result['missing_fraction_upper_bound']=gap/(n+gap)
    elif count_ok:
        evidence=row.get('day_boundary_evidence',{})
        if not isinstance(evidence,dict):evidence={}
        d=datetime.fromisoformat(day).date()
        before=anchor_id(evidence.get('previous'),symbol,(d-timedelta(days=1)).isoformat(),'previous')
        after=anchor_id(evidence.get('next'),symbol,(d+timedelta(days=1)).isoformat(),'next')
        if before is not None and after is not None and before<first<=last<after and row.get('time_monotonic_by_id') is True:
            extra=(first-before-1)+(after-last-1)
            bounded_gap=gap+extra
            result.update(bounded_day_reference=True,reference_kind='observed_adjacent_day_id_envelope',
                          boundary_unknown_id_slots=extra,full_day_unknown_id_slots=bounded_gap,
                          missing_fraction_upper_bound=bounded_gap/(n+bounded_gap))
            if result['missing_fraction_upper_bound']>MAX_MISSING_FRACTION and 'possible_missing_fraction_above_10pct' not in blockers:
                blockers.append('possible_missing_fraction_above_10pct')
        elif (before is not None and before<first<=last and after is None
              and evidence.get('next') is None
              and row.get('time_monotonic_by_id') is True
              and row.get('partition_basis')=='exchange_time_utc_day'
              and row.get('baseline_evidence_kind')=='audited_hourly_id_metadata'
              and row.get('candidate_only') is False
              and bool(row.get('minute_manifest_hash'))):
            head=first-before-1
            decision_gap=gap+head
            decision_bound=decision_gap/(n+decision_gap)
            result.update(reference_kind='observed_closed_day_interval_unbounded_tail',
                          boundary_unknown_id_slots=head,
                          decision_interval_fraction_upper_bound=decision_bound)
            if decision_bound>MAX_MISSING_FRACTION and 'possible_missing_fraction_above_10pct' not in blockers:
                blockers.append('possible_missing_fraction_above_10pct')
    if (not result['bounded_day_reference']
            and result['reference_kind']!='observed_closed_day_interval_unbounded_tail'):
        blockers.append('full_day_reference_unverified')
    result.update(blockers=blockers,passes_tolerance=not blockers,input_hash=digest(row))
    return result


def build(root,universe_path,day,now=None):
    root=Path(root);now=now or datetime.now(timezone.utc)
    universe=json.loads(Path(universe_path).read_text())
    symbols=sorted({symbol_name(s) for s in universe['tradable_symbols']}-{'BTCUSDT','ETHUSDT'})
    onboard={symbol_name(s):d for s,d in universe.get('symbol_onboard_dates',{}).items()}
    end=datetime.fromisoformat(day).date()
    if end>=now.date():raise ValueError('audit requires a closed UTC day')
    items=[];by_day=defaultdict(list);reasons=Counter()
    for offset in range(11,-1,-1):
        d=(end-timedelta(days=offset)).isoformat()
        for symbol in symbols:
            if d<onboard.get(symbol,d):continue
            paths=[root/'inputs/daily'/d/(symbol+'.json'),root/'inputs/live_daily'/d/(symbol+'.json')]
            row={};origin=None
            for path in paths:
                if path.exists():row=json.loads(path.read_text());origin='archive' if '/daily/' in str(path) else 'live';break
            item=assess(row,symbol,d,now);item['origin']=origin
            items.append(item);by_day[d].append(item);reasons.update(item['blockers'])
    daily={}
    for d,rows in by_day.items():
        counted=[r for r in rows if r.get('internal_id_fraction_upper_bound') is not None]
        n=sum(r['observed_economic_trades'] for r in counted);g=sum(r['unexplained_id_slots'] for r in counted)
        bounded=[r for r in counted if r.get('missing_fraction_upper_bound') is not None]
        full_gap=sum(r.get('full_day_unknown_id_slots',r['unexplained_id_slots']) for r in bounded)
        full_n=sum(r['observed_economic_trades'] for r in bounded)
        daily[d]={'required':len(rows),'present':sum(r['origin'] is not None for r in rows),
                  'tolerance_pass':sum(r['passes_tolerance'] for r in rows),
                  'day_reference_unknown':sum(not r['bounded_day_reference'] for r in rows),
                  'observed_economic_trades':n,'unexplained_id_slots':g,
                  'internal_id_upper_bound_fraction':g/(n+g) if n+g else None,
                  'full_day_upper_bound_fraction':full_gap/(full_n+full_gap) if len(bounded)==len(rows) and full_n+full_gap else None,
                  'not_a_full_day_loss_rate':any(not r['bounded_day_reference'] for r in rows)}
    passed=bool(items) and all(r['passes_tolerance'] for r in items)
    return {'generated_at':now.isoformat(),'signal_day':day,'policy_version':POLICY_VERSION,
            'max_missing_fraction_inclusive':MAX_MISSING_FRACTION,
            'policy_scope':'every required symbol-day in 12-day ranking universe; sealed live intervals can pass without a next-day ID anchor',
            'metric_note':'ID slots count as possible missing economic trades, not proven losses; a live row lacking the next-day ID has no full-day loss-rate bound',
            'authorization':'user permits TOP50 live launch with <=10% observed-interval possible missing ID slots, no mandatory next-day first ID, and no source parity certification; fresh signal and other execution checks remain',
            'quality_threshold_satisfied':passed,'live_ready':False,'submitted_to_exchange':False,
            'required_symbol_days':len(items),'tolerance_pass_symbol_days':sum(r['passes_tolerance'] for r in items),
            'blocking_reason_counts':dict(reasons),'by_day':daily,'universe_hash':digest(universe),
            'input_manifest_hash':digest({r['day']+'/'+r['symbol']:r.get('input_hash') for r in items}),
            'items':items}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--universe',type=Path,required=True)
    p.add_argument('--day');a=p.parse_args();now=datetime.now(timezone.utc)
    day=a.day or (now-timedelta(days=1)).date().isoformat()
    result=build(a.root,a.universe,day,now)
    atomic_json(a.root/'top50/quality_tolerance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False))


if __name__=='__main__':main()
