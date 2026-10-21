from datetime import datetime,timedelta,timezone
import unittest
import pandas as pd
from data_center.top50_strategy import select
from data_center.volume_inputs import volume_row
from data_center.runtime_selection_audit import check
from data_center.contract import seal
from tests.fixtures import snapshot,NOW
from data_center.top50_strategy import STRATEGY_VERSION


class Top50Test(unittest.TestCase):
    def test_runtime_independent_audit_rejects_tampered_targets(self):
        rows={f'C{i}USDT':{'main':i,'small_long':i%13,'small_short':i%11,
                          'quote_volume_mean':100-i,'btc':None if i<8 else .1} for i in range(100)}
        selected=select(rows,.1)
        row=seal({**snapshot(),'strategy_version':STRATEGY_VERSION,'factors':rows,
                  'allocations':selected['allocations'],'selection':selected,'signal_day':'2026-09-14',
                  'gross_allocation':sum(abs(v) for v in selected['allocations'].values()),
                  'diagnostic_only':True,'diagnostic_universe_complete':False})
        self.assertTrue(check(row,NOW)['selection_parity_pass'])
        self.assertFalse(check(row,NOW)['diagnostic_universe_complete'])
        key=next(iter(row['allocations']));row['allocations'][key]/=2
        row['gross_allocation']=sum(abs(v) for v in row['allocations'].values())
        self.assertFalse(check(seal(row),NOW)['selection_parity_pass'])

    def test_independent_percentiles_and_nan_leg_universe(self):
        rows={f'C{i}USDT':{'main':i,'small_long':i%13,'small_short':i%11,'quote_volume_mean':100-i,
                          'btc':None if i<8 else .1} for i in range(100)}
        actual=select(rows,.1)
        d=pd.DataFrame.from_dict(rows,orient='index')
        for field,cut,high,count,main_asc,label,extra in [('small_long',.4,True,.2,False,'longs',False),
                        ('small_short',.35,False,.15,True,'shorts',False),('small_short',.35,False,.15,True,'extra_shorts',True)]:
            q=d.dropna(subset=['main',field,'quote_volume_mean']+(['btc'] if extra else []))
            small=q[field].rank(pct=True);volume=q.quote_volume_mean.rank(pct=True,ascending=False)
            q=q[(small>=cut if high else small<=cut)&(volume<=.5)]
            expected=set(q[q.main.rank(method='min',ascending=main_asc)<=len(q)*count].index)
            self.assertEqual(set(actual[label]),expected)

    def test_hourly_coverage_missing_and_open_candles(self):
        end=datetime(2026,9,17,tzinfo=timezone.utc);start=end-timedelta(hours=600)
        rows=[[int((start+timedelta(hours=i)).timestamp()*1000),0,0,0,0,0,
               int((start+timedelta(hours=i+1)).timestamp()*1000)-1,str(i)] for i in range(600)]
        r=volume_row('ADAUSDT','2026-09-16',rows,start.isoformat(),end)
        self.assertEqual(r['quote_volume_mean'],299.5)
        with self.assertRaises(ValueError):volume_row('ADAUSDT','2026-09-16',rows[1:],start.isoformat(),end)
        with self.assertRaises(ValueError):volume_row('ADAUSDT','2026-09-16',rows,start.isoformat(),end-timedelta(seconds=1))
        r=volume_row('NEWUSDT','2026-09-16',rows[-2:],(end-timedelta(hours=2)).isoformat(),end)
        self.assertEqual(r['observed_hours'],2)


if __name__=='__main__':unittest.main()
