import unittest
import numpy as np
import pandas as pd

from research.optimizer.v3_history import daily_fast, economic_events, rolling_features, MAIN, SMALL


class HistoricalV3Test(unittest.TestCase):
    def test_vectorized_formula_matches_live_with_boundaries_ties_and_sparse_minutes(self):
        from data_center.lagged_quantile_v3 import daily_from_events, quantiles
        rng = np.random.default_rng(81)
        base = pd.Timestamp('2026-01-01').value//1000
        n = 100000
        stamp = rng.integers(-3_600_000, 86_400_000, n)*1000+base
        stamp[:100] = base+np.arange(100)*60_000_000
        raw = pd.DataFrame({'id': np.arange(n), 'timestamp': stamp,
            'side': np.where(rng.random(n)>.5, 'buy', 'sell'),
            'price': rng.integers(1, 10, n)*.0001, 'amount': rng.integers(1, 10, n)})
        events, _ = economic_events(raw)
        expected_events = events.copy()
        expected_events['time'] = pd.to_datetime(events.timestamp, unit='us', utc=True)
        prior = expected_events.loc[expected_events.timestamp < base]
        day = expected_events.loc[expected_events.timestamp >= base]
        expected = daily_from_events(day, '2026-01-01', quantiles(prior))
        actual = daily_fast(events, '2026-01-01')
        for key in [MAIN, SMALL]: self.assertAlmostEqual(actual[key], expected[key], places=12)
        self.assertEqual(actual['minute_count'], expected['minute_count'])

    def test_conflicting_duplicates_fail(self):
        raw = pd.DataFrame({'id': [1, 1], 'timestamp': [1000, 1000],
            'side': ['buy', 'buy'], 'price': [1., 2.], 'amount': [1., 1.]})
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            economic_events(raw)

    def test_rolling_features_match_live_missingness_and_young_symbol_rules(self):
        frame = pd.DataFrame({MAIN:list(range(1,13)),SMALL:[np.nan]*4+list(range(8))})
        result = rolling_features(frame)
        self.assertEqual(result['small_long'],3.5)
        self.assertEqual(result['small_short'],5.5)
        frame.loc[0,MAIN] = np.nan
        self.assertIsNone(rolling_features(frame))
        young = pd.DataFrame({MAIN:[1.,np.nan,3.],SMALL:[np.nan,2.,3.]})
        self.assertEqual(rolling_features(young)['main'],7/3)


if __name__ == '__main__': unittest.main()


class HistoricalScopeTest(unittest.TestCase):
    def test_contract_scope_retains_delisted_crypto(self):
        from research.optimizer.v3_history import historical_contract_allowed as allowed
        metadata = {'BABAUSDT':{'contractType':'TRADIFI_PERPETUAL','quoteAsset':'USDT','baseAsset':'BABA'},
            'OLDUSDT':{'contractType':'PERPETUAL','quoteAsset':'USDT','baseAsset':'OLD','status':'SETTLING'}}
        self.assertFalse(allowed('BABAUSDT',metadata))
        self.assertFalse(allowed('USDCUSDT',metadata))
        self.assertFalse(allowed('ABCUPUSDT',metadata))
        self.assertTrue(allowed('JUPUSDT',metadata))
        self.assertTrue(allowed('OLDUSDT',metadata))
        self.assertTrue(allowed('LEGACYUSDT',metadata))
