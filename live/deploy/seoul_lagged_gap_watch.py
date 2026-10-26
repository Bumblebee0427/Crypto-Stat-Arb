"""Bounded recovery of missing/blocked minute markers from retained hourly stage."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import shutil

from data_center.contract import atomic_json
from deploy.seoul_lagged_bootstrap_minutes import fill
from deploy.seoul_lagged_minute_shadow import ROOT, MIN_FREE, STAGED
from data_center.lagged_quantile_v3 import FORMULA_VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--day')
    p.add_argument('--max-hours',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.max_hours<=24:raise ValueError('invalid repair bound')
    if shutil.disk_usage(ROOT).free<MIN_FREE:raise RuntimeError('minute repair waits for 20GiB free')
    with (ROOT/'.gap-repair.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        now=datetime.now(timezone.utc)
        days=[args.day] if args.day else [(now-timedelta(days=1)).date().isoformat(),now.date().isoformat()]
        results=[]
        for day in days:
            start=datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
            if start.date()<now.date()-timedelta(days=1) or start>now:raise ValueError('repair day outside recent window')
            marker=ROOT/'inputs/live_daily'/day/'complete.json'
            if marker.exists():
                sealed=json.loads(marker.read_text())
                if sealed.get('status')=='complete' and sealed.get('day')==day and sealed.get('formula_version')==FORMULA_VERSION:
                    result={'day':day,'status':'sealed_skipped','minutes_written':0}
                    results.append(result);print(json.dumps(result),flush=True);continue
            state=STAGED/day/'stage_state.json'
            if not state.exists():continue
            checkpoint=datetime.fromisoformat(json.loads(state.read_text())['capture_checkpoint_at'])
            stable=min(now,checkpoint)-timedelta(minutes=5)
            last=min(1439,int((stable-start).total_seconds()//60)-1)
            if last<0:continue
            through,minute=divmod(last,60)
            print(json.dumps({'day':day,'phase':'repair_scan','through_hour':through,'through_minute':minute}),flush=True)
            result=fill(day,through,max_hours=args.max_hours,through_minute=minute)
            results.append(result)
            print(json.dumps(result),flush=True)
        atomic_json(ROOT/'gap_repair_status.json',{'checked_at':now.isoformat(),'results':results})


if __name__=='__main__':main()
