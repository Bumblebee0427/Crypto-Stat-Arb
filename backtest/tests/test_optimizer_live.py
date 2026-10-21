import unittest
import numpy as np
import pandas as pd

from research.optimizer.live import allocation, MODEL, SPEC, V3
from research.optimizer.factor_risk_tuning import estimate_grid
from research.optimizer.paper_factors import CORE_FACTORS, EXTRA_FACTORS


class LiveOptimizerAdapterTest(unittest.TestCase):
    def test_online_labels_start_after_publication_and_link_next_day_prices(self):
        import io,json,tempfile
        from pathlib import Path
        from urllib.parse import urlsplit,parse_qs
        from unittest.mock import patch
        from tools.run_live_optimizer import archive_v3_alpha
        calls=[]
        def response(request,timeout):
            query=parse_qs(urlsplit(request.full_url).query);start=int(query['startTime'][0])
            calls.append(start);day=pd.to_datetime(start,unit='ms').day
            return io.BytesIO(json.dumps([[start,0,0,0,0,'10',start+59999,str(day*100)]]).encode())
        with tempfile.TemporaryDirectory() as folder, patch('tools.run_live_optimizer.urlopen',side_effect=response):
            root=Path(folder)
            for day in [1,2]:
                boundary=pd.Timestamp(f'2024-01-{day:02}',tz='UTC').to_pydatetime()
                generated=pd.Timestamp(boundary)+pd.Timedelta(minutes=4,seconds=20)
                source={'generated_at':generated.isoformat(),'formula_version':V3,
                    'allocations':{'AUSDT':.5,'BUSDT':-.5},
                    'factors':{s:{'main':1.,'small_long':.2,'small_short':.3} for s in ['AUSDT','BUSDT']}}
                archive_v3_alpha(root,source,boundary)
            panel=pd.read_parquet(root/'alpha_panel.parquet')
            first=panel.loc[panel['asof']==pd.Timestamp('2024-01-01')]
            np.testing.assert_allclose(first.future_return,1.)
            self.assertTrue((pd.to_datetime(calls,unit='ms').minute==5).all())
            self.assertEqual(first.label_available_at.iloc[0],pd.Timestamp('2024-01-02 01:00:00'))

    def inputs(self):
        rng = np.random.default_rng(27)
        dates = pd.date_range('2026-06-01', periods=130)
        factors = pd.DataFrame(rng.normal(0, .015, (130, 6)), index=dates,
                               columns=CORE_FACTORS+EXTRA_FACTORS)
        symbols = ['AUSDT', 'BUSDT', 'CUSDT', 'DUSDT']
        history = pd.DataFrame(factors[CORE_FACTORS].to_numpy()@rng.normal(size=(3, 4))
            +rng.normal(0, .015, (130, 4)), index=dates, columns=symbols)
        asof = dates[-1]
        signal = {'formula_version': V3, 'window_end': asof.isoformat()+'+00:00',
            'allocations': dict(zip(symbols, [.25, .25, -.25, -.25])), 'leverage': 2.,
            'factors': {s: {'main': j, 'small_long': j, 'small_short': j} for j, s in enumerate(symbols)}}
        events = pd.DataFrame([{'symbol': s, 'time': asof-pd.Timedelta(hours=8),
            'rate': .0001, 'interval_hours': 8., 'source': 'settled_raw_archive'} for s in symbols])
        provenance = {'capitalization_latest_source_date': str(asof-pd.Timedelta(days=2)),
            'cap_publication_delay_days': 2, 'maximum_available_at': asof.isoformat()}
        return signal, history, factors, events, {s: 0 for s in symbols}, {s: 1 for s in symbols}, 1500., provenance

    def test_same_risk_math_as_research_and_bounded_blend(self):
        args = self.inputs(); result = allocation(*args)
        expected = estimate_grid(args[1], args[2], sorted(args[0]['allocations']),
                                 args[1].index[-1], [SPEC])[MODEL]
        self.assertEqual(result['risk'], expected['info'])
        self.assertEqual(result['alpha_status'], 'v3_training_warmup_zero_alpha')
        self.assertEqual(result['solver_status'], 'optimal')
        w = result['allocations']
        self.assertAlmostEqual(sum(v for v in w.values() if v > 0), .5)
        self.assertAlmostEqual(sum(v for v in w.values() if v < 0), -.5)
        for s, b in args[0]['allocations'].items():
            self.assertGreaterEqual(w[s]/b, .875-1e-7)
            self.assertLessEqual(w[s]/b, 1.125+1e-7)

    def test_old_capitalization_cannot_silently_activate_live_model(self):
        args = list(self.inputs())
        args[-1] = {**args[-1], 'capitalization_latest_source_date': '2026-04-13'}
        with self.assertRaisesRegex(ValueError, 'capitalization source is stale'):
            allocation(*args)

    def test_future_input_and_wrong_formula_rejected(self):
        args = list(self.inputs())
        args[0] = {**args[0], 'formula_version': 'v2'}
        with self.assertRaisesRegex(ValueError, 'v3 base signal'): allocation(*args)
        args = list(self.inputs())
        args[1] = pd.concat([args[1], args[1].iloc[-1:].set_axis([args[1].index[-1]+pd.Timedelta(days=1)])])
        with self.assertRaisesRegex(ValueError, 'decision boundary'): allocation(*args)


if __name__ == '__main__':
    unittest.main()
