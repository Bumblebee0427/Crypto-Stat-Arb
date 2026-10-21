"""Economic invariants and temporal isolation, using synthetic markets."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd

from research.optimizer.contracts import ResearchConfig
from research.optimizer.costs import transaction_cost
from research.optimizer.solve import solve_weights
from research.optimizer.risk import estimate_risk
from research.optimizer.funding import forecast_funding, load_settlements
from research.optimizer.alpha import fit_alpha, predict_alpha
from research.optimizer.replay import replay, target_lots
from research.optimizer.validation import select_schedule
from research.optimizer.dataset import Dataset


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig()

    def test_funding_sign_and_equal_side_budgets(self):
        b = np.array([.5, .5, -.225, -.225])
        result = solve_weights(b, b, np.zeros(4), np.eye(4)*.0001,
            np.array([.02, 0., .02, 0.]), replace(self.config, gamma=0, eta=0),
            use_alpha=False, use_cost=False)
        self.assertEqual(result.status, 'optimal')
        self.assertLess(result.weights[0], result.weights[1])
        self.assertLess(result.weights[2], result.weights[3])
        self.assertAlmostEqual(result.weights[:2].sum(), 1.)
        self.assertAlmostEqual(result.weights[2:].sum(), -.45)

    def test_same_leg_alpha_constant_has_no_effect(self):
        b = np.array([.3, .7, -.4, -.6]); cov = np.eye(4)*.01
        a = solve_weights(b, b, np.zeros(4), cov, np.zeros(4), self.config)
        z = solve_weights(b, b, np.array([.05, .05, -.1, -.1]), cov, np.zeros(4), self.config)
        np.testing.assert_allclose(a.weights, z.weights, atol=1e-7)

    def test_exits_cost_and_flip_notional(self):
        self.assertAlmostEqual(transaction_cost([-.05], [.05], .0008), .00008)
        b = np.array([.5, .5, 0.]); before = np.array([0., 0., -.4])
        result = solve_weights(b, before, np.zeros(3), np.eye(3)*.01, np.zeros(3), self.config)
        self.assertAlmostEqual(result.weights[2], 0.)
        self.assertAlmostEqual(result.terms['expected_transaction_cost'], .0008*1.4)

    def test_no_hidden_leverage_and_fixed_short_history(self):
        b = np.array([1., -.45])
        r = solve_weights(b, b, np.array([100., -100.]), np.eye(2), np.zeros(2),
                          self.config, fixed=np.array([True, True]))
        np.testing.assert_allclose(r.weights, b, atol=1e-8)

    def test_invalid_covariance_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PSD'):
            solve_weights(np.array([.5, .5]), np.zeros(2), np.zeros(2),
                          np.array([[1., 2.], [2., 1.]]), np.zeros(2), self.config)

    def test_solver_failure_returns_feasible_baseline(self):
        from unittest.mock import patch
        import cvxpy as cp
        b = np.array([.5, .5])
        with patch('cvxpy.Problem.solve', side_effect=cp.error.SolverError('timeout')):
            r = solve_weights(b, b, np.zeros(2), np.eye(2), np.zeros(2), self.config)
        np.testing.assert_array_equal(r.weights, b)
        self.assertEqual(r.status, 'baseline_fallback')

    def test_optimal_economic_objective_is_no_worse_than_feasible_baseline(self):
        b = np.array([.4, .6, -.25, -.25, 0.])
        current = np.array([.5, .5, 0., -.2, -.3])
        alpha = np.array([.01, .005, -.003, .002, 0.])
        funding = np.array([.002, -.001, .001, -.002, 0.])
        cov = np.eye(5)*.004
        r = solve_weights(b, current, alpha, cov, funding, self.config)
        def value(w):
            return float(alpha@w-self.config.gamma/2*(w@cov@w)-
                transaction_cost(w, current, .0008)-funding@w-
                self.config.eta/2*np.sum(((w-b)/np.maximum(np.abs(b), 1e-8))**2))
        self.assertEqual(r.status, 'optimal')
        self.assertGreaterEqual(value(r.weights), value(b)-1e-9)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig()

    def events(self):
        times = pd.date_range('2025-01-01', periods=9, freq='8h')
        return pd.DataFrame({'time': times, 'event_hour': times, 'rate': .001,
                             'interval_hours': 8., 'source': 'test'})

    def test_funding_uses_only_available_settlements_and_includes_next_boundary(self):
        events = self.events(); asof = pd.Timestamp('2025-01-03')
        value, info = forecast_funding(events, asof, self.config)
        self.assertAlmostEqual(value, .003)
        self.assertEqual(len(info['events']), 3)
        self.assertEqual(info['events'][-1], '2025-01-04T00:00:00')
        events.loc[events.time >= asof, 'rate'] = -999.
        self.assertEqual(forecast_funding(events, asof, self.config)[0], value)
        self.assertLess(pd.Timestamp(info['available_at']), asof)

    def test_hourly_interval_is_not_multiplied_by_three(self):
        events = self.events().iloc[:2].copy()
        events['interval_hours'] = 1.
        value, info = forecast_funding(events, pd.Timestamp('2025-01-02'), self.config)
        self.assertEqual(len(info['events']), 24)
        self.assertAlmostEqual(value, .024)

    def test_short_history_risk_is_positive_and_psd(self):
        rng = np.random.default_rng(4); n = 180
        f = rng.normal(0, .02, (n, 2))
        frame = pd.DataFrame({'BTC-USDT': f[:, 0], 'ETH-USDT': f[:, 1],
            'A': f[:, 0]+rng.normal(0, .02, n), 'NEW': np.nan})
        frame.loc[n-4:, 'NEW'] = .01
        cov, info, _ = estimate_risk(frame, ['A', 'NEW'], self.config)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)
        self.assertGreater(cov[1, 1], .005)
        self.assertEqual(info['samples'][1], 4)

    def panel(self):
        rng = np.random.default_rng(2); rows = []
        for t in pd.date_range('2022-01-01', periods=450):
            for side in [-1, 1]:
                for k in range(4):
                    x = k/2-1
                    rows.append({'asof': t, 'side': side, 'main_rank': x,
                        'small_rank': x*.5, 'future_return': x*.01+rng.normal(0, .005),
                        'label_available_at': t+pd.Timedelta(days=1, hours=1)})
        return pd.DataFrame(rows)

    def test_immature_labels_cannot_change_alpha_model(self):
        panel = self.panel(); cutoff = pd.Timestamp('2023-01-01')
        model = fit_alpha(panel, cutoff, self.config)
        current = panel.loc[panel['asof'] == cutoff]
        a, _ = predict_alpha(current, model)
        panel.loc[panel.label_available_at > cutoff, 'future_return'] = 999.
        b, _ = predict_alpha(current, fit_alpha(panel, cutoff, self.config))
        np.testing.assert_array_equal(a, b)
        self.assertLessEqual(model.label_cutoff, cutoff)

    def test_model_cannot_predict_its_past(self):
        panel = self.panel(); model = fit_alpha(panel, '2023-01-01', self.config)
        with self.assertRaisesRegex(ValueError, 'future-trained'):
            predict_alpha(panel.iloc[:4], model)

    def test_future_prices_cannot_change_risk_history(self):
        rng = np.random.default_rng(3)
        times = pd.date_range('2024-01-01', periods=220)
        returns = pd.DataFrame(rng.normal(0, .02, (220, 3)), index=times,
                               columns=['BTC-USDT', 'ETH-USDT', 'A'])
        data = SimpleNamespace(daily_returns=returns)
        cutoff = times[180]
        h = Dataset.daily_history(data, cutoff, 180)
        first = estimate_risk(h, ['A'], self.config)[0]
        returns.loc[returns.index > cutoff, :] = 100.
        second = estimate_risk(Dataset.daily_history(data, cutoff, 180), ['A'], self.config)[0]
        np.testing.assert_array_equal(first, second)

    def test_missing_funding_is_explicit_not_observed_zero(self):
        value, info = forecast_funding(self.events().iloc[:0], '2025-01-03', self.config)
        self.assertEqual(value, 0.)
        self.assertEqual(info['source'], 'missing')
        self.assertTrue(info['status'].startswith('missing_history'))

    def test_settlement_cache_cannot_survive_changed_market_source(self):
        import tempfile
        from pathlib import Path
        times = pd.date_range('2024-01-01', periods=2, freq='h')
        data = SimpleNamespace(symbols=['SYNTHETIC-UNLISTED-USDT'], times=times,
            prices={'funding_rate': np.array([[.001], [.002]])}, manifest={'version': 1})
        with tempfile.TemporaryDirectory() as folder:
            a = load_settlements(data, Path(folder))
            self.assertAlmostEqual(a.rate.iloc[-1], .002)
            data.prices['funding_rate'] *= 2
            data.manifest = {'version': 2}
            b = load_settlements(data, Path(folder))
            self.assertAlmostEqual(b.rate.iloc[-1], .004)

    def test_insufficient_alpha_history_is_not_a_strong_prediction(self):
        panel = self.panel().iloc[:80]
        model = fit_alpha(panel, '2023-01-01', self.config)
        current = self.panel().loc[lambda d: d['asof'] == '2023-01-01']
        prediction, supported = predict_alpha(current, model)
        np.testing.assert_array_equal(prediction, np.zeros(len(current)))
        self.assertFalse(supported.any())


class ReplayTests(unittest.TestCase):
    def dataset(self):
        times = pd.date_range('2025-01-01 23:00', periods=27, freq='h')
        price = np.full((len(times), 2), 100.)
        funding = np.zeros_like(price); funding[1] = [.01, -.01]; funding[25] = [.01, -.01]
        return SimpleNamespace(times=times, symbols=['A', 'B'], steps=np.array([1., 1.]),
            prices={'open': price.copy(), 'close': price.copy(), 'vwap1m': price.copy(),
                    'funding_rate': funding}, weights=np.tile([.5, -.5], (len(times), 1)),
            initial_equity=10000., fee_rate=.0008, min_order=5., margin_rate=.05)

    def test_legacy_simulator_parity_with_own_equity(self):
        data = self.dataset(); curve, audit = replay(data, verify_simulator=True)
        self.assertEqual(audit['simulator_lot_mismatches'], 0)
        self.assertEqual(curve.funding_fee.iloc[1], 0.)  # old flat account, before entry
        self.assertGreater(curve.funding_fee.iloc[25], 0.)
        self.assertAlmostEqual(curve.equity.iloc[-1], 10000.-curve.fee.sum()-curve.funding_fee.sum())

    def test_future_execution_does_not_change_decision(self):
        data = self.dataset(); seen = []
        def policy(i, asof, b, state):
            seen.append((asof, state.equity, state.lots.tolist())); return b, []
        replay(data, policy); first = seen[0]; seen.clear()
        data.prices['vwap1m'][1:] *= 2
        replay(data, policy)
        self.assertEqual(first, seen[0])

    def test_missing_realized_funding_does_not_become_free(self):
        data = self.dataset(); data.prices['funding_rate'][2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing realized funding'):
            replay(data)

    def test_lot_rounding_and_nonfinite_price(self):
        target = target_lots(1000., np.array([100., np.nan, 100.]),
                             np.array([.251, 1., -.251]), np.ones(3))
        np.testing.assert_array_equal(target, [2, 0, -2])

    def test_ledger_funding_and_fee_reconcile_to_account(self):
        import tempfile
        from pathlib import Path
        data = self.dataset()
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            curve, _ = replay(data, out=out, ledger=True)
            events = pd.read_parquet(out/'funding_events.parquet')
            trades = pd.read_parquet(out/'trades.parquet')
            self.assertAlmostEqual(events.amount.sum(), curve.funding_fee.sum())
            self.assertAlmostEqual(trades.fee.sum(), curve.fee.sum())
            pnl = pd.read_parquet(out/'pnl_attribution.parquet')
            self.assertAlmostEqual(pnl.net_pnl.sum(), curve.equity.iloc[-1]-data.initial_equity)


if __name__ == '__main__':
    unittest.main()
