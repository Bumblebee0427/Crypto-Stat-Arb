"""Financial construction, no-future-information and covariance-estimator checks."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.covariance import ledoit_wolf

from research.optimizer.paper_factors import (PaperRiskConfig, historical_features,
    tail_portfolios, paper_week_boundaries, leg_returns, build_factors, load_capitalization)
from research.optimizer.factor_covariance import (covariance_weights, weighted_covariance,
    estimate_covariance_block, estimate_variants, VARIANTS)
from research.optimizer.factor_risk_reporting import gaussian_covariance_score, risk_calibration


class FactorConstructionTests(unittest.TestCase):
    def test_size_sort_value_weights_and_symmetric_tails(self):
        names = [f'C{k:02}' for k in range(20)]
        cap = pd.Series(np.arange(1, 21)*1e6, index=names)
        ports = tail_portfolios(cap, cap)
        self.assertEqual(ports['low'].index.tolist(), names[:6])
        self.assertEqual(ports['high'].index.tolist(), names[-6:])
        self.assertAlmostEqual(ports['low'].sum(), 1.)
        self.assertAlmostEqual(ports['high'].sum(), 1.)
        self.assertAlmostEqual(ports['low'].iloc[-1]/ports['low'].iloc[0], 6.)

    def test_exact_paper_year_end_week_lengths(self):
        for year, final_length in [(2023, 8), (2024, 9)]:
            calendar = pd.date_range(f'{year}-01-01', f'{year+1}-01-01')
            boundaries = paper_week_boundaries(calendar)
            self.assertEqual(len(boundaries), 53)
            lengths = np.diff(boundaries.to_numpy()).astype('timedelta64[D]').astype(int)
            np.testing.assert_array_equal(lengths[:51], np.full(51, 7))
            self.assertEqual(lengths[-1], final_length)

    def test_formation_quantities_drift_with_prices(self):
        t = pd.date_range('2024-01-01', periods=3)
        c = pd.DataFrame({'A': [100., 200., 200.], 'B': [100., 100., 200.]}, index=t)
        r, weekly = leg_returns(c, t[0], t[-1], pd.Series({'A': .5, 'B': .5}))
        np.testing.assert_allclose(r, [.5, 1/3])
        self.assertAlmostEqual(weekly, 1.)

    def test_missing_held_coin_is_not_removed_from_factor(self):
        t = pd.date_range('2024-01-01', periods=3)
        c = pd.DataFrame({'A': [100., 110., 120.], 'B': [100., np.nan, 100.]}, index=t)
        r, _ = leg_returns(c, t[0], t[-1], pd.Series({'A': .5, 'B': .5}))
        self.assertTrue(r.isna().all())

    def test_reference_momentum_amihud_volatility_and_skew(self):
        t = pd.date_range('2024-01-01', periods=50)
        r = np.linspace(-.03, .06, 50)+np.sin(np.arange(50))*.01
        prices = pd.DataFrame({'A': 100*np.cumprod(1+r)}, index=t)
        quotes = pd.DataFrame({'A': np.linspace(1e6, 2e6, 50)}, index=t)
        f = historical_features(prices, quotes)
        observed = prices.A.pct_change(fill_method=None)
        self.assertAlmostEqual(f['momentum21'].A.iloc[-1], prices.A.iloc[-1]/prices.A.iloc[-22]-1)
        self.assertAlmostEqual(f['illiquidity10'].A.iloc[-1], observed.iloc[-10:].abs().sum()/quotes.A.iloc[-10:].sum())
        self.assertAlmostEqual(f['volatility30'].A.iloc[-1], observed.iloc[-30:].std(ddof=1))
        self.assertAlmostEqual(f['skewness7'].A.iloc[-1], skew(observed.iloc[-7:], bias=True), places=10)

    def test_capitalization_delay_expiry_and_no_backfill(self):
        t = pd.date_range('2024-01-01', periods=9)
        c = pd.DataFrame({'A-USDT': np.full(9, 100.)}, index=t)
        cfg = PaperRiskConfig()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder); capfile = p/'A-USDT.csv'
            capfile.write_text('metadata\ncandle_begin_time,id,circulating_mcap\n2024-01-01,1,10000000\n2024-01-08,1,99999999\n')
            cap, ids, available, _ = load_capitalization(p, c, cfg, p)
            self.assertTrue(cap.loc['2024-01-02'].isna().all())
            self.assertEqual(cap.loc['2024-01-03', 'A-USDT'], 1e7)
            self.assertEqual(cap.loc['2024-01-06', 'A-USDT'], 1e7)
            self.assertTrue(cap.loc['2024-01-07'].isna().all())
            self.assertEqual(available.loc['2024-01-03', 'A-USDT'], pd.Timestamp('2024-01-03'))

    def test_capitalization_alias_prefers_exact_instrument_name(self):
        t=pd.date_range('2024-01-01',periods=5)
        c=pd.DataFrame({'A_B-USDT':100.},index=t)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for name,value in [('A_B-USDT',5e7),('A-B-USDT',1e7)]:
                (p/(name+'.csv')).write_text('metadata\ncandle_begin_time,id,circulating_mcap\n2024-01-01,1,'+str(value)+'\n')
            cap,_,_,_=load_capitalization(p,c,PaperRiskConfig(),p)
            self.assertEqual(cap.loc['2024-01-03','A_B-USDT'],5e7)

    def test_future_prices_cannot_change_completed_factors_or_formation(self):
        t = pd.date_range('2023-10-01', periods=130)
        rng = np.random.default_rng(6); names = [f'C{k:02}-USDT' for k in range(30)]
        c = pd.DataFrame(100*np.cumprod(1+rng.normal(.001, .02, (130, 30)), axis=0), index=t, columns=names)
        q = pd.DataFrame(1e6, index=t, columns=names)
        cap = pd.DataFrame(np.tile(np.arange(1,31)*1e6,(130,1)), index=t, columns=names)
        ids = pd.DataFrame(np.tile(np.arange(1,31),(130,1)), index=t, columns=names)
        available = pd.DataFrame({s: t for s in names}, index=t)
        cutoff = pd.Timestamp('2024-01-08')
        a = build_factors(c,q,cap,ids,available,PaperRiskConfig())
        c.loc[c.index > cutoff] *= np.linspace(1.,10.,30)
        cap.loc[cap.index > cutoff] *= 1000
        b = build_factors(c,q,cap,ids,available,PaperRiskConfig())
        pd.testing.assert_frame_equal(a.loc[:cutoff],b.loc[:cutoff])


class CovarianceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PaperRiskConfig()

    def fixture(self):
        rng = np.random.default_rng(42)
        t = pd.date_range('2023-01-01', periods=240)
        names = ['CMKT','CSMB','CMOM','CILLIQ10','CVOL30','CSKEW7']
        f = pd.DataFrame(rng.normal(0,.02,(240,6)), index=t, columns=names)
        beta = np.array([[1.,.5,.3,0.,0.,0.],[.8,-.3,.1,.1,.2,.1]])
        y = f.to_numpy()@beta.T+rng.normal(0,.01,(240,2))
        history = pd.DataFrame(y,index=t,columns=['A','B'])
        history['NEW'] = np.nan; history.loc[t[-4]:,'NEW'] = [.01,.02,-.01,.03]
        return history,f

    def test_ewma_uses_calendar_age_with_missing_dates(self):
        t = pd.to_datetime(['2024-01-01','2024-01-31'])
        np.testing.assert_allclose(covariance_weights(t,t[-1],30),[1/3,2/3])

    def test_future_observation_rejected(self):
        with self.assertRaisesRegex(ValueError,'future observation'):
            covariance_weights(pd.to_datetime(['2024-01-02']),'2024-01-01',30)

    def test_recent_shock_has_more_weight_under_ewma(self):
        t = pd.date_range('2024-01-01',periods=180)
        x = np.zeros((180,2)); x[-1] = [.2,.1]
        sample,_ = estimate_covariance_block(x,t,t[-1],'fixed',self.cfg)
        ewma,_ = estimate_covariance_block(x,t,t[-1],'ewma_fixed',self.cfg)
        self.assertGreater(ewma[0,0],sample[0,0]*3)

    def test_standard_lw_matches_published_library_estimator(self):
        rng = np.random.default_rng(4); x = rng.normal(size=(50,5))
        t = pd.date_range('2024-01-01',periods=50)
        actual,info = estimate_covariance_block(x,t,t[-1],'lw',self.cfg)
        expected,shrink = ledoit_wolf(x)
        np.testing.assert_allclose(actual,expected,rtol=1e-13)
        self.assertEqual(info['shrinkage'],shrink)

    def test_ewma_lw_is_explicit_convex_hybrid(self):
        rng = np.random.default_rng(8); x = rng.normal(size=(80,4))*[1,2,3,4]
        t = pd.date_range('2024-01-01',periods=80)
        actual,info = estimate_covariance_block(x,t,t[-1],'ewma_lw',self.cfg)
        _,delta = ledoit_wolf(x)
        s = weighted_covariance(x,covariance_weights(t,t[-1],30))
        expected = (1-delta)*s+delta*np.trace(s)/4*np.eye(4)
        np.testing.assert_allclose(actual,expected)
        self.assertFalse(info['weighted_lw_optimality_claimed'])

    def test_all_variants_psd_and_short_history_is_conservative(self):
        history,f = self.fixture(); asof = history.index[-1]
        results = estimate_variants(history,f,history.columns.tolist(),asof,self.cfg)
        self.assertEqual(set(results),set(VARIANTS))
        for name,value in results.items():
            self.assertGreaterEqual(np.linalg.eigvalsh(value['covariance']).min(),1e-8-1e-10,name)
            self.assertGreaterEqual(value['covariance'][2,2],.1**2-1e-10,name)
            self.assertEqual(value['info']['samples'][2],4,name)

    def test_future_returns_and_factors_do_not_change_current_covariance(self):
        history,f = self.fixture(); cutoff = history.index[200]
        first = estimate_variants(history,f,['A','B'],cutoff,self.cfg)
        history.loc[history.index > cutoff] = 1000.
        f.loc[f.index > cutoff] = -1000.
        second = estimate_variants(history,f,['A','B'],cutoff,self.cfg)
        for name in first:
            np.testing.assert_array_equal(first[name]['covariance'],second[name]['covariance'])

    def test_nonmarket_factor_sign_changes_do_not_change_covariance(self):
        history,f = self.fixture(); cutoff = history.index[-1]
        a = estimate_variants(history,f,['A','B'],cutoff,self.cfg)
        f['CMOM'] *= -1
        b = estimate_variants(history,f,['A','B'],cutoff,self.cfg)
        for name in a:
            np.testing.assert_allclose(a[name]['covariance'],b[name]['covariance'],rtol=1e-10,atol=1e-12)

    def test_gaussian_score_uses_log_determinant_and_inverse(self):
        cov=np.diag([.04,.09]); z=np.array([.1,-.2])
        expected=(np.log(.04*.09)+.1**2/.04+.2**2/.09)/2
        self.assertAlmostEqual(gaussian_covariance_score(cov,z),expected)

    def test_calibration_uses_next_day_outcome_and_common_past_mean(self):
        from types import SimpleNamespace
        t=pd.date_range('2024-01-01',periods=3); asof=t[1]
        history=pd.DataFrame({'A':[0.,.02,.1],'B':[0.,-.02,-.04]},index=t)
        old=SimpleNamespace(rows=pd.DataFrame({'symbol':['A','B'],'w0':[.5,-.5]}),
            covariance=np.eye(2)*.01,risk_info={'samples':[20,20]})
        risks={asof:{'paper3_diag':{'covariance':np.eye(2)*.02,'info':{'samples':[20,20]}}}}
        result=risk_calibration(SimpleNamespace(daily_returns=history),{asof:old},risks,PaperRiskConfig(),asof,t[2])
        self.assertTrue((result.outcome_available_at==t[2]).all())
        np.testing.assert_allclose(result.portfolio_return,.07)
        # Past means are +.01/-.01: portfolio innovation .07-.01=.06.
        np.testing.assert_allclose(result.portfolio_innovation,.06)


if __name__=='__main__':
    unittest.main()
