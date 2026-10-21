#!/usr/bin/env python3
"""Maintain v3 raw hourly staging and ID evidence; calculate no old factors."""
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from data_center.base_strategy import symbol_name
from data_center.captured_daily import stage
from data_center.hourly_trade_metadata import refresh_hours
from data_center.lagged_quantile_v3 import FORMULA_VERSION

from data_center.paths import BASE as _PORTABLE_BASE

BASE = _PORTABLE_BASE
CAPTURE=BASE/'data/individual_trades_v2'
ROOT=BASE/'data/captured_daily_stream_v1'
UNIVERSE=BASE/'data/live_factor_storage/_state/universe.json'
V3=BASE/'data/lagged_quantile_v3'
MIN_FREE=19*1024**3


def tail_seed(day):
    previous=(datetime.fromisoformat(day).date()-timedelta(days=1)).isoformat()
    wanted=({symbol_name(s) for s in json.loads(UNIVERSE.read_text())['tradable_symbols']}
            -{'BTCUSDT','ETHUSDT'})
    buckets={}
    for path in (CAPTURE/'raw'/previous).glob('*.jsonl.gz'):
        try:buckets.setdefault(int(path.name.split('-',1)[0]),[]).append(path)
        except ValueError:continue
    found={}
    for bucket in sorted(buckets,reverse=True):
        for path in buckets[bucket]:
            with path.open('rb') as source:sha=hashlib.file_digest(source,'sha256').hexdigest()
            with gzip.open(path,'rt') as source:
                for line in source:
                    record=json.loads(line)
                    if record.get('validation_error'):continue
                    trade=record.get('data',{});symbol=trade.get('s')
                    if symbol not in wanted:continue
                    ident,ms=trade.get('t'),trade.get('T')
                    if not isinstance(ident,int) or not isinstance(ms,int):continue
                    if datetime.fromtimestamp(ms/1000,timezone.utc).date().isoformat()!=previous:continue
                    old=found.get(symbol)
                    if old is None or ident>old['id']:
                        found[symbol]={'id':ident,'timestamp_ms':ms,'raw_segment_sha256':sha}
        if len(found)==len(wanted):return found
    return None


def seed_for(day):
    previous=(datetime.fromisoformat(day).date()-timedelta(days=1)).isoformat()
    path=ROOT/'work'/previous/'stage_state.json'
    if path.exists():
        row=json.loads(path.read_text())
        wanted=({symbol_name(s) for s in json.loads(UNIVERSE.read_text())['tradable_symbols']}
                -{'BTCUSDT','ETHUSDT'})
        if row.get('layout')=='hourly' and wanted.issubset(row.get('target_last',{})):
            return row['target_last']
    return tail_seed(day)


def process(day,now):
    work=ROOT/'work'/day
    staged=stage(CAPTURE,work,day,layout='hourly',
                 previous_seed=None if (work/'stage_state.json').exists() else seed_for(day),
                 max_segments=1000)
    metadata=refresh_hours(work,day,now)
    return {'day':day,'stage':staged,'metadata':metadata,'formula':'v3','old_factor_jobs':0}


def verified_daily(day):
    """Require intact sealed v3 outputs before discarding derived source CSVs."""
    output=V3/'inputs/live_daily'/day
    try:
        row=json.loads((output/'complete.json').read_text())
        symbols=row['symbols']; hashes=row['files_sha256']
        if (row.get('day')!=day or row.get('status')!='complete'
                or row.get('formula_version')!=FORMULA_VERSION
                or not symbols or set(symbols)!=set(hashes)):
            return False
        for symbol in symbols:
            if Path(symbol).name!=symbol or symbol in ('.','..'):
                return False
            if hashlib.sha256((output/(symbol+'.json')).read_bytes()).hexdigest()!=hashes[symbol]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def cleanup(now):
    # Yesterday needs all 24 hours for metadata and late-revision replay.
    # The preceding day is only read for its 23:00 threshold/replay seed.
    # Keep that seed and all ID/stage evidence; raw gzip retention is separate.
    expiry=(now.date()-timedelta(days=2)).isoformat()
    removed=[]
    for path in sorted((ROOT/'work').glob('*')):
        if not path.is_dir() or path.is_symlink():continue
        try:
            day=datetime.fromisoformat(path.name).date()
        except ValueError:
            continue
        if day.isoformat()!=path.name or path.name>expiry:continue
        hours=[path/'hours'/f'{hour:02d}' for hour in range(23)]
        if path.name==expiry and not any(hour.exists() for hour in hours):continue
        following=(day+timedelta(days=1)).isoformat()
        if not all(verified_daily(d) for d in (path.name,following)):continue
        if path.name<expiry:
            shutil.rmtree(path)
            removed.append({'day':path.name,'removed':'whole_stage'})
        else:
            deleted=[]
            for hour in hours:
                if hour.is_dir() and not hour.is_symlink():
                    shutil.rmtree(hour)
                    deleted.append(hour.name)
            if deleted:
                removed.append({'day':path.name,'removed_hours':deleted,'retained_seed_hour':'23'})
    return removed


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc)
    removed=cleanup(now)
    if removed:print(json.dumps({'stage_cleanup':removed}),flush=True)
    if shutil.disk_usage(ROOT).free<MIN_FREE:
        raise RuntimeError('less than 19GiB free; v3 source producer paused')
    today=now.date().isoformat();yesterday=(now.date()-timedelta(days=1)).isoformat()
    days=([yesterday] if (ROOT/'work'/yesterday).exists() else [])+[today]
    for day in days:
        print(json.dumps(process(day,datetime.now(timezone.utc))),flush=True)


if __name__=='__main__':main()
