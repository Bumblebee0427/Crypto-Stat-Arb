"""Independent pandas ranking against a published TOP50 preview; never certifies source."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_center.contract import atomic_json,validate
from data_center.top50_strategy import STRATEGY_VERSION,LAGGED_STRATEGY_VERSION,LEGACY_LAGGED_STRATEGY_VERSION


def check(snapshot,now=None):
    now=now or datetime.now(timezone.utc)
    version=snapshot['strategy_version']
    if version not in (STRATEGY_VERSION,LAGGED_STRATEGY_VERSION,LEGACY_LAGGED_STRATEGY_VERSION):raise ValueError('unknown strategy')
    validate(snapshot,now,actionable=False,expected_strategy=version,expected_formula=snapshot['formula_version'])
    is_spot=version==LAGGED_STRATEGY_VERSION
    columns=['main','small_long','small_short','btc','quote_volume_mean']
    frame=pd.DataFrame.from_dict(snapshot['factors'],orient='index').reindex(columns=columns)
    frame=frame.replace([np.inf,-np.inf],np.nan)
    expected={};leg_results={}
    for field,cut,high,fraction,ascending,budget,label,extra in (
        ('small_long',.4,True,.2,False,.5,'longs',False),
        ('small_short',.35,False,.15,True,-.225,'shorts',False),
        ('small_short',.35,False,.15,True,-.275,'extra_shorts',True)):
        q=frame.dropna(subset=['main',field,'quote_volume_mean']+(['btc'] if extra else []))
        q=q.loc[q.quote_volume_mean>=0]
        small=q[field].rank(method='average',pct=True)
        volume=q.quote_volume_mean.rank(method='average',pct=True,ascending=False)
        q=q[(small.ge(cut) if high else small.le(cut)) & volume.le(.5)]
        if extra:q=q[q.btc.le(.2)]
        if is_spot:
            members=snapshot['spot_membership']['membership']
            leg_results[label]=[]
            for has_spot in (True,False):
                group=q.loc[[s for s in q.index if members[s]==has_spot]]
                k=min(len(group),max(int(len(group)*(.4 if label=='longs' else .3)),10))
                selected=group[group.main.rank(method='min',ascending=ascending).le(k)]
                leg_results[label]+=list(selected.index)
                for symbol in selected.index:expected[symbol]=expected.get(symbol,0.)+budget*.5/len(selected)
            leg_results[label].sort()
        else:
            selected=q[q.main.rank(method='min',ascending=ascending).le(len(q)*fraction)]
            leg_results[label]=sorted(selected.index)
            for symbol in selected.index:expected[symbol]=expected.get(symbol,0.)+budget/len(selected)
    actual=snapshot['allocations']
    mismatch=[s for s in set(actual)|set(expected) if abs(actual.get(s,0)-expected.get(s,0))>1e-12]
    leg_mismatch=[k for k,v in leg_results.items() if v!=sorted(snapshot['selection'][k])]
    return {'checked_at':now.isoformat(),'snapshot_id':snapshot['snapshot_id'],'signal_day':snapshot['signal_day'],
            'strategy_version':version,'selection_parity_pass':not mismatch and not leg_mismatch,
            'diagnostic_universe_complete':snapshot.get('diagnostic_universe_complete') is True,
            'compared_factor_symbols':len(frame),'target_count':len(actual),
            'mismatch_symbols':sorted(mismatch),'mismatch_legs':leg_mismatch,
            'source_certified':False,'live_ready':False,
            'scope':'independent selection/ranking from published factors; not raw/daily factor or source certification'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:
        snapshot=json.loads((a.root/'preview.json').read_text());result=check(snapshot)
    except Exception as exc:
        result={'checked_at':datetime.now(timezone.utc).isoformat(),'selection_parity_pass':False,
                'error_type':type(exc).__name__,'live_ready':False,'source_certified':False}
    atomic_json(a.root/'runtime_selection_audit.json',result);print(json.dumps(result))
    return 0 if result['selection_parity_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
