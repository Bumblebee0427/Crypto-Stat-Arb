"""Behavioral checks for second-round financial and temporal invariants."""
import tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd

from research.optimizer.contracts import ResearchConfig
from research.optimizer.validation import sortino_ratio, paired_bootstrap, summary
from research.optimizer.solve import solve_weights
from research.optimizer.replay import resize_targets, replay
from research.optimizer.policy import make_policy
from research.optimizer.refinements import build_refined_alpha, select_refinement_schedule, SELECTION_CANDIDATES


class MetricTests(unittest.TestCase):
    def test_sortino_denominator_uses_all_days(self):
        r = np.array([.02, -.01, .03, -.02])
        expected = r.mean()/np.sqrt((.01**2+.02**2)/4)*np.sqrt(365)
        self.assertAlmostEqual(sortino_ratio(r), expected)
        negative_days_only = r.mean()/np.sqrt((.01**2+.02**2)/2)*np.sqrt(365)
        self.assertNotAlmostEqual(sortino_ratio(r), negative_days_only)

    def test_no_downside_is_undefined_not_zero(self):
        self.assertIsNone(sortino_ratio([.01, .02, 0.]))

    def test_block_bootstrap_contains_sortino_difference(self):
        d = pd.date_range('2025-01-01', periods=28)
        a = pd.DataFrame({'net_return': np.tile([.02, -.01, .03, -.02], 7)}, index=d)
        result = paired_bootstrap(a, a, draws=50)
        np.testing.assert_allclose(result['sortino_difference_95ci'], [0., 0.])

    def test_drawdown_includes_loss_from_initial_capital(self):
        d = pd.DataFrame({'net_return': [-.1, .01], 'fee_return': [0., 0.],
            'funding_return': [0., 0.], 'turnover_ratio': [0., 0.],
            'rebalance_fee': [0., 0.], 'before': [100., 90.]},
            index=pd.date_range('2025-01-01', periods=2))
        self.assertAlmostEqual(summary(d)['max_drawdown'], -.1)


class RefinedObjectiveTests(unittest.TestCase):
    def test_payment_only_does_not_reward_receiving_funding(self):
        b = np.array([.5, .5]); cfg = ResearchConfig(gamma=0., eta=.001)
        args = (b, b, np.zeros(2), np.eye(2)*.001, np.array([-.02, -.01]), cfg)
        pay = solve_weights(*args, use_alpha=False, use_cost=False, funding_mode='pay_only')
        signed = solve_weights(*args, use_alpha=False, use_cost=False)
        np.testing.assert_allclose(pay.weights, b, atol=1e-7)
        self.assertGreater(signed.weights[0], signed.weights[1])

    def test_payment_only_short_funding_sign(self):
        b = np.array([-.5, -.5]); cfg = ResearchConfig(gamma=0., eta=.001)
        result = solve_weights(b, b, np.zeros(2), np.eye(2)*.001, np.array([-.02, -.001]), cfg,
                               use_alpha=False, use_cost=False, funding_mode='pay_only')
        self.assertLess(abs(result.weights[0]), abs(result.weights[1]))

    def test_tracking_risk_protects_reference_instead_of_minimizing_total_variance(self):
        b = np.array([.5, .5]); cfg = ResearchConfig(gamma=0., eta=0.)
        r = solve_weights(b, b, np.zeros(2), np.diag([.1, .001]), np.zeros(2), cfg,
                          use_alpha=False, use_cost=False, use_funding=False, tracking_gamma=10.)
        np.testing.assert_allclose(r.weights, b, atol=1e-7)

    def test_blending_preserves_budgets_signs_and_fraction_of_tilt(self):
        asof = pd.Timestamp('2025-01-01'); b = np.array([.5, .5, -.5, -.5])
        f = SimpleNamespace(rows=pd.DataFrame({'column': range(4)}), alpha=np.array([.01, -.01, .01, -.01]),
            alpha_bins=np.zeros(4), covariance=np.eye(4)*.001, funding=np.zeros(4),
            betas=np.zeros((4, 2)), risk_info={'samples': [180]*4}, alpha_supported=np.ones(4, bool),
            funding_info=[{'status': 'observed'}]*4, fitted_at=asof, label_cutoff=asof-pd.Timedelta(days=1))
        data = SimpleNamespace(panel=pd.DataFrame({'asof': [asof-pd.Timedelta(days=500)]}),
            prices={'close': np.full((1, 4), 100.)}, steps=np.ones(4), fee_rate=.0008, symbols=list('ABCD'))
        state = SimpleNamespace(lots=np.array([50, 50, -50, -50]), equity=10000., last_prices=np.full(4, 100.))
        full, _ = make_policy(data, {asof: f}, ResearchConfig(), 'B5_full')(0, asof, b, state)
        mixed, _ = make_policy(data, {asof: f}, ResearchConfig(), 'B5_full', blend=.25)(0, asof, b, state)
        np.testing.assert_allclose(mixed-b, .25*(full-b), atol=1e-12)
        self.assertAlmostEqual(mixed[:2].sum(), 1.); self.assertAlmostEqual(mixed[2:].sum(), -1.)


class ResizeBandTests(unittest.TestCase):
    def test_small_resize_suppressed(self):
        target, count = resize_targets(np.array([50, 50]), np.array([49, 51]), np.array([100., 100.]),
            np.ones(2), 10000., np.array([.5, .5]), .1)
        np.testing.assert_array_equal(target, [49, 51]); self.assertEqual(count, 2)

    def test_daily_rebalance_and_exit_always_pass(self):
        args = (np.array([50, 0]), np.array([49, 1]), np.array([100., 100.]), np.ones(2), 10000., np.array([.5, 0.]), .1)
        target, _ = resize_targets(*args)
        self.assertEqual(target[1], 0)
        target, count = resize_targets(*args, force=True)
        np.testing.assert_array_equal(target, [50, 0]); self.assertEqual(count, 0)

    def test_side_budget_guard_refreshes_even_inside_wide_band(self):
        target, count = resize_targets(np.array([50, 50]), np.array([54, 54]), np.array([100., 100.]),
            np.ones(2), 10000., np.array([.5, .5]), .25)
        np.testing.assert_array_equal(target, [50, 50]); self.assertEqual(count, 0)

    def test_sign_flip_cannot_be_suppressed(self):
        target, _ = resize_targets(np.array([-1, 51]), np.array([1, 49]), np.array([100., 100.]),
            np.ones(2), 10000., np.array([-.01, .51]), .25)
        self.assertEqual(target[0], -1)


class RefinedAlphaTests(unittest.TestCase):
    def test_future_labels_cannot_change_current_month_prediction(self):
        rng = np.random.default_rng(73); rows = []
        for t in pd.date_range('2024-01-01', periods=100):
            for side in [-1, 1]:
                for k in range(4):
                    rows.append({'asof': t, 'symbol': f'{side}_{k}', 'side': side, 'main_rank': k/4,
                        'small_rank': k/4, 'main': k/100, 'small': k/200,
                        'predicted_vol': .02+k*.001, 'expected_funding': k*.0001,
                        'future_return': .01*k+rng.normal(0, .01),
                        'label_available_at': t+pd.Timedelta(hours=25)})
        frame = pd.DataFrame(rows); cfg = ResearchConfig(train_days=30)
        cutoff = pd.Timestamp('2024-03-01')
        with tempfile.TemporaryDirectory() as folder:
            a = build_refined_alpha(frame, cfg, Path(folder))[cutoff]
            frame.loc[frame.label_available_at > cutoff, 'future_return'] = 999.
            b = build_refined_alpha(frame, cfg, Path(folder))[cutoff]
            np.testing.assert_array_equal(a, b)
            records = pd.read_parquet(Path(folder)/'refined_alpha.parquet')
            supported = records.loc[records.supported]
            self.assertTrue((supported.training_label_cutoff <= supported.fitted_at).all())


class WalkForwardTests(unittest.TestCase):
    def test_later_returns_cannot_change_prior_sortino_choice(self):
        times = pd.date_range('2024-01-01', periods=180*24, freq='h')
        curves = {}
        for k, name in enumerate(SELECTION_CANDIDATES):
            daily = np.tile([.01+.0001*k, -.005-.0002*k, .002], 60)
            hourly = np.repeat((1+daily)**(1/24)-1, 24)
            equity = 10000*np.cumprod(1+hourly)
            curves[name] = pd.DataFrame({'time': times, 'equity': equity,
                'equity_before': np.r_[10000., equity[:-1]], 'fee': 0., 'funding_fee': 0.,
                'turnover': 0., 'rebalance_fee': 0., 'long_price_pnl': 0., 'short_price_pnl': 0.})
        cfg = ResearchConfig(first_test='2024-04-01')
        boundary = pd.Timestamp(cfg.first_test)
        a, _ = select_refinement_schedule(curves, cfg, times[-1], 'sortino')
        for curve in curves.values():
            curve.loc[curve.time >= boundary, 'equity'] *= 2.
        b, _ = select_refinement_schedule(curves, cfg, times[-1], 'sortino')
        self.assertEqual(a[boundary], b[boundary])


if __name__ == '__main__':
    unittest.main()
