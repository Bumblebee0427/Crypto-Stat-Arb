import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf
from research.optimizer.factor_risk_tuning import (experiment_specs, covariance_block,
    target_matrix, estimate_grid, rolling_selection)
from research.optimizer.paper_factors import CORE_FACTORS, EXTRA_FACTORS


class RiskTuningTests(unittest.TestCase):
    def test_ablation_has_all_eight_subsets_at_each_reference_method(self):
        specs=experiment_specs()
        for method in ['lw_identity','cv_diagonal']:
            base=[s for s in specs if s.window==180 and s.shrinkage==method and
                s.factor_half_life==(None if method=='lw_identity' else 30) and
                s.residual_half_life==s.factor_half_life]
            self.assertEqual(len(base),8)
            self.assertEqual(len({s.factors for s in base}),8)

    def test_lw_identity_exact_and_standardized_lw_preserves_variances(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(90,4))*[.01,.03,.1,.5]
        dates=pd.date_range('2024-01-01',periods=len(x))
        cov,_=covariance_block(x,dates,dates[-1],None,'lw_identity')
        np.testing.assert_allclose(cov,ledoit_wolf(x)[0])
        corr,_=covariance_block(x,dates,dates[-1],None,'lw_correlation')
        np.testing.assert_allclose(np.diag(corr),np.var(x,axis=0),rtol=1e-12)

    def test_all_targets_psd_and_diagonal_targets_preserve_asset_variances(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(100,6))*np.arange(1,7)
        s=np.cov(x,rowvar=False,bias=True)
        for target in ['identity','diagonal','constant_correlation']:
            t=target_matrix(s,target)
            self.assertGreater(np.linalg.eigvalsh(t).min(),0)
            if target!='identity':np.testing.assert_allclose(np.diag(s),np.diag(t))

    def test_cv_split_is_chronological_and_classical_lw_rejects_ewma(self):
        x=np.random.default_rng(5).normal(size=(90,5));dates=pd.date_range('2024-01-01',periods=90)
        for target in ['identity','diagonal','constant_correlation']:
            cov,info=covariance_block(x,dates,dates[-1],14,'cv_'+target)
            self.assertLess(pd.Timestamp(info['cv_train_last']),pd.Timestamp(info['cv_validation_first']))
            self.assertEqual(pd.Timestamp(info['cv_validation_last']),dates[-1])
            self.assertGreater(np.linalg.eigvalsh(cov).min(),0)
        with self.assertRaises(ValueError):covariance_block(x,dates,dates[-1],30,'lw_identity')

    def test_future_perturbations_cannot_change_grid_and_separate_decay_changes_risk(self):
        rng=np.random.default_rng(9);dates=pd.date_range('2023-01-01',periods=240)
        f=pd.DataFrame(rng.normal(0,.02,(240,6)),index=dates,columns=CORE_FACTORS+EXTRA_FACTORS)
        y=pd.DataFrame(f.values[:,:2]+rng.normal(0,.01,(240,2)),index=dates,columns=['A','B'])
        y.iloc[185:201]*=4
        specs=[s for s in experiment_specs() if s.factors==tuple(CORE_FACTORS) and
            s.factor_half_life==30 and s.residual_half_life in [30,60] and s.window==180]
        cutoff=dates[200]
        a=estimate_grid(y,f,list(y),cutoff,specs)
        y.loc[y.index>cutoff]=1000;f.loc[f.index>cutoff]=-1000
        b=estimate_grid(y,f,list(y),cutoff,specs)
        for name in a:np.testing.assert_array_equal(a[name]['covariance'],b[name]['covariance'])
        self.assertGreater(np.max(np.abs(a[specs[0].name]['covariance']-a[specs[-1].name]['covariance'])),1e-8)

    def test_selection_excludes_not_yet_observed_outcomes_and_future_winners(self):
        rows=[]
        for date in pd.date_range('2024-01-01',periods=100):
            for model in ['a','b']:
                rows.append({'asof':date,'outcome_available_at':date+pd.Timedelta(days=1),
                    'model':model,'portfolio_qlike':0 if model=='a' else 1})
        scores=pd.DataFrame(rows);boundary=pd.Timestamp('2024-03-10')
        a=rolling_selection(scores,[boundary],['a','b'],'b')
        scores.loc[scores.outcome_available_at>boundary,'portfolio_qlike']=-1e10
        b=rolling_selection(scores,[boundary],['a','b'],'b')
        self.assertEqual(a,b);self.assertEqual(a[0]['model'],'a')
        self.assertLessEqual(pd.Timestamp(a[0]['latest_outcome_available_at']),boundary)

    def test_subsets_share_the_same_missing_factor_history(self):
        rng=np.random.default_rng(12);dates=pd.date_range('2023-01-01',periods=100)
        factors=pd.DataFrame(rng.normal(0,.02,(100,6)),index=dates,columns=CORE_FACTORS+EXTRA_FACTORS)
        factors.loc[dates[::7],'CSKEW7']=np.nan
        history=pd.DataFrame(rng.normal(0,.03,(100,2)),index=dates,columns=['A','B'])
        specs=[s for s in experiment_specs() if s.shrinkage=='lw_identity']
        results=estimate_grid(history,factors,list(history),dates[-1],specs)
        observations={r['info']['factor_observations'] for r in results.values()}
        self.assertEqual(observations,{85})


if __name__=='__main__':unittest.main()
