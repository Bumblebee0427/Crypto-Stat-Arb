"""End-to-end synthetic accounting/chronology test, not performance evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.optimizer.v3_history import V3
from research.optimizer.paper_factors import CORE_FACTORS, EXTRA_FACTORS
from tools.validate_live_alpha_history import execution_inputs, evaluate


class AlphaWalkForwardTest(unittest.TestCase):
    def test_complete_monthly_walk_and_repeatable_execution_alignment(self):
        rng = np.random.default_rng(810)
        dates = pd.date_range('2025-07-01','2026-10-09')
        symbols = ['AUSDT','BUSDT','CUSDT','DUSDT']
        realized = rng.normal(0,.002,(len(dates),4))+rng.normal(0,.006,(len(dates),1))
        realized += np.array([-.0005,.0005,-.0005,.0005])
        prices = pd.DataFrame(100*np.exp(np.cumsum(realized,axis=0)),index=dates,columns=symbols)
        future = prices.shift(-1)/prices-1
        rows = []
        for day in dates[:-1]:
            for j, symbol in enumerate(symbols):
                rows.append({'asof':day,'symbol':symbol,'side':1 if j<2 else -1,
                    'baseline':.5 if j<2 else -.5,'main_rank':float(j%2),'small_rank':float(j%2),
                    'formula_version':V3,'feature_available_at':day+pd.Timedelta(minutes=8),
                    'entry_at':day+pd.Timedelta(hours=1),'exit_at':day+pd.Timedelta(days=1,hours=1),
                    'label_available_at':day+pd.Timedelta(days=1,hours=2),
                    'future_return':future.at[day,symbol]})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'validation';(root/'risk').mkdir()
            original=pd.DataFrame(rows);original.to_parquet(root/'alpha_panel.parquet',index=False)
            prices.to_parquet(root/'execution_prices_10m.parquet')
            prices.stack().rename('price').rename_axis(['asof','symbol']).reset_index().to_parquet(root/'entry_prices_01h.parquet',index=False)
            (root/'panel_manifest.json').write_text(json.dumps({'limitations':['synthetic test only']}))
            (root/'execution_minutes_manifest.json').write_text(json.dumps({'synthetic':True}))
            (root/'validation_design.json').write_text(json.dumps({'test_start':'2026-07-01','test_end_inclusive':'2026-10-07'}))
            risk_dates=pd.date_range('2026-03-01','2026-10-08')
            factors=pd.DataFrame(rng.normal(0,.006,(len(risk_dates),6)),index=risk_dates,columns=CORE_FACTORS+EXTRA_FACTORS)
            factors.to_parquet(root/'risk/factors.parquet')
            prices.pct_change(fill_method=None).to_parquet(root/'risk/returns.parquet')
            (root/'risk/manifest.json').write_text('{}')
            times=pd.date_range('2026-06-20','2026-10-09',freq='8h')
            events=pd.DataFrame([{'symbol':s,'time':t+pd.Timedelta(milliseconds=2),'event_hour':t,'rate':.0001,'interval_hours':8.,'source':'synthetic'} for s in symbols for t in times])
            events.to_parquet(root/'settlements.parquet',index=False)
            hours=pd.date_range('2026-06-30','2026-10-09',freq='h')
            market={s[:-4]+'-USDT':pd.DataFrame({'candle_begin_time':hours,'close':prices[s].reindex(hours,method='ffill').to_numpy(),
                'avg_price_1m':prices[s].reindex(hours,method='ffill').to_numpy(),'funding_fee':0.}) for s in symbols}
            path=root/'market.pkl';pd.to_pickle(market,path)
            args=SimpleNamespace(history=root,out=out,market=path,market_tail=None,settlements=None)
            execution_inputs(args)
            expected=pd.read_parquet(root/'alpha_panel.parquet')
            original.to_parquet(root/'alpha_panel.parquet',index=False)
            execution_inputs(args)
            pd.testing.assert_frame_equal(expected,pd.read_parquet(root/'alpha_panel.parquet'))
            with contextlib.redirect_stdout(io.StringIO()): evaluate(args)
            report=json.loads((out/'validation.json').read_text())
            self.assertTrue(report['checks']['accounting_and_budget_audit'])
            self.assertTrue(report['checks']['both_sides_all_refits'])
            self.assertEqual(len(report['refits']),4)
            self.assertEqual(report['funding_max_boundary_jitter_ms'],2.)
            self.assertTrue(pd.read_parquet(out/'accounts.parquet').funding_pnl.abs().gt(0).any())
            self.assertEqual(len(pd.read_parquet(out/'accounts.parquet')),99*6)
            for record in report['refits']:
                self.assertLessEqual(pd.Timestamp(record['label_cutoff']),pd.Timestamp(record['fitted_at']))


            # Delisting has no next-day execution price: settle on the actual
            # cessation day, charge exit fees and do not carry phantom lots.
            from research.optimizer.contracts import digest
            modified=original.loc[~(original.symbol.eq('CUSDT') & original['asof'].gt(pd.Timestamp('2026-07-02')))].copy()
            modified.loc[modified['asof'].gt(pd.Timestamp('2026-07-02')) & modified.symbol.eq('DUSDT'),'baseline']=-1.
            modified.to_parquet(root/'alpha_panel.parquet',index=False)
            missing_prices=prices.copy();missing_prices.loc['2026-07-03':,'CUSDT']=np.nan
            missing_prices.to_parquet(root/'execution_prices_10m.parquet')
            body={'symbol':'CUSDT','settled_at':'2026-07-02 09:00:00','window_start':'2026-07-02 08:30:00',
                'lower':90.,'upper':110.,'index_observations':1800,'archive_sha256':'synthetic',
                'announcement_url':'synthetic','rule_url':'synthetic','exact_settlement_price_known':False}
            (root/'delisting_events.json').write_text(json.dumps({'events':[{**body,'event_id':digest(body)}]}))
            execution_inputs(args)
            with contextlib.redirect_stdout(io.StringIO()):evaluate(args)
            bounded=json.loads((out/'validation.json').read_text())
            self.assertTrue(bounded['checks']['accounting_and_budget_audit'])
            accounts=pd.read_parquet(out/'accounts.parquet')
            self.assertTrue(accounts.loc[accounts['asof'].eq(pd.Timestamp('2026-07-02')),'settlement_fee'].gt(0).all())
            decisions=pd.read_parquet(out/'decisions.parquet')
            self.assertFalse((decisions['asof'].gt(pd.Timestamp('2026-07-02')) & decisions.symbol.eq('CUSDT')).any())
            self.assertEqual(set(bounded['settlement_interval_sensitivity']),{'adverse','favorable'})


if __name__ == '__main__': unittest.main()
