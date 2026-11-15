from datetime import datetime,timedelta,timezone
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from data_center.spot_split_strategy import select,STRATEGY_VERSION,ALLOCATION_METHOD
from data_center.spot_membership import build,validate,spot_symbol
from data_center.optimizer_overlay import load_and_apply,awaiting_artifact,MODEL
from data_center.contract import seal
from data_center.runtime_selection_audit import check


class SpotSplitTest(unittest.TestCase):
    def fixture(self,n=240):
        rows={f'C{i}USDT':{'main':float(i%39),'small_long':float(i%17),'small_short':float(i%13),
                         'quote_volume_mean':float(300-i),'btc':None if i%19==0 else .1} for i in range(n)}
        members={s:i%3!=0 for i,s in enumerate(rows)}
        return rows,members

    def test_independent_pandas_selection_and_budgets_with_ties(self):
        rows,members=self.fixture();actual=select(rows,.1,members)
        frame=pd.DataFrame.from_dict(rows,orient='index');expected={}
        for field,cut,high,fraction,budget,label,extra in [('small_long',.4,True,.4,.5,'longs',False),
            ('small_short',.35,False,.3,-.225,'shorts',False),('small_short',.35,False,.3,-.275,'extra_shorts',True)]:
            q=frame.dropna(subset=['main',field,'quote_volume_mean']+(['btc'] if extra else []))
            p=q[field].rank(pct=True);v=q.quote_volume_mean.rank(pct=True,ascending=False)
            q=q[((p>=cut)if high else(p<=cut))&(v<=.5)]
            if extra:q=q[q.btc<=.2]
            selected=[]
            for flag in [True,False]:
                g=q.loc[[s for s in q.index if members[s]==flag]];k=min(len(g),max(int(len(g)*fraction),10))
                ids=g[g.main.rank(method='min',ascending=not high)<=k].index
                selected+=list(ids)
                for s in ids:expected[s]=expected.get(s,0.)+.5*budget/len(ids)
            self.assertEqual(sorted(selected),actual[label])
        self.assertEqual(expected.keys(),actual['allocations'].keys())
        for s in expected:self.assertAlmostEqual(expected[s],actual['allocations'][s],places=14)
        # Missing membership cannot silently mean no spot.
        members.pop(next(iter(members)))
        with self.assertRaises(ValueError):select(rows,.1,members)

    def test_empty_group_keeps_cash_and_btc_protection(self):
        rows,members=self.fixture();members={s:True for s in rows}
        for r in rows.values():r['btc']=.3
        result=select(rows,.3,members)
        self.assertFalse(result['extra_shorts'])
        self.assertAlmostEqual(sum(result['allocations'].values()),.25-.1125)
        self.assertLessEqual(sum(abs(v) for v in result['allocations'].values()),.3625+1e-12)

    def test_closed_hour_membership_aliases_and_partial_network_failures(self):
        end=datetime(2026,11,15,tzinfo=timezone.utc);near=end-timedelta(hours=1)
        symbols=['PEPEUSDT','LUNAUSDT','IDLEUSDT','ERRUSDT']
        def fetch(endpoint,params=None):
            if endpoint=='exchangeInfo':return {'symbols':[{'symbol':s,'quoteAsset':'USDT','isSpotTradingAllowed':True} for s in symbols]}
            if params['symbol']=='ERRUSDT':raise OSError('network')
            return [[int(near.timestamp()*1000),0,0,0,'1',0,int(end.timestamp()*1000)-1,0,0 if params['symbol']=='IDLEUSDT' else 1]]
        universe={'tradable_symbols':['1000PEPEUSDT','LUNA2USDT','IDLEUSDT','ERRUSDT','ONLYPERPUSDT']}
        result=build(universe,'2026-11-14',fetch=fetch)
        self.assertEqual(result['membership'],{'1000PEPEUSDT':True,'LUNA2USDT':True,'IDLEUSDT':False,'ONLYPERPUSDT':False})
        self.assertEqual(result['excluded_symbols'],{'ERRUSDT':'spot_hour_unavailable:OSError'})
        self.assertEqual(validate(result,'2026-11-14',datetime.now(timezone.utc),universe['tradable_symbols']),result)
        with self.assertRaises(ValueError):validate(result,'2026-11-13',datetime.now(timezone.utc),[])
        with self.assertRaises(ValueError):build(universe,'2026-11-14',now=near,fetch=fetch)
        self.assertEqual(spot_symbol('1000PEPEUSDT',{'1000PEPEUSDT','PEPEUSDT'}),'1000PEPEUSDT')

    def test_b_bypasses_optimizer_even_with_matching_artifact(self):
        now=datetime(2026,11,15,0,4,tzinfo=timezone.utc)
        s=seal({'allocation_method':ALLOCATION_METHOD,'window_end':now.replace(minute=0).isoformat(),'generated_at':now.isoformat(),'allocations':{'AUSDT':.25,'BUSDT':-.25}})
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'optimizer';p.mkdir();(p/'enabled.json').write_text(json.dumps({'enabled':True,'model':MODEL}))
            out,info=load_and_apply(s,root,{},1500,now)
            self.assertEqual(out,s);self.assertEqual(info['status'],'disabled_by_strategy')
            self.assertFalse(awaiting_artifact(s,root,now))


if __name__=='__main__':unittest.main()
