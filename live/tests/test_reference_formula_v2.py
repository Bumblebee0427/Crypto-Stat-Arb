import unittest
import numpy as np
import pandas as pd

from data_center.reference_formula_v2 import aggregate_raw_trades, minute_features_from_events, nearest_half_up, daily_factors, MAIN, SMALL


class ReferenceFormulaV2Test(unittest.TestCase):
    def test_nearest_half_up_not_bankers_rounding(self):
        self.assertEqual(nearest_half_up([1, 2, 3, 4, 5, 6], .5), 4)
        self.assertEqual(nearest_half_up([1, 2, 3], .5), 2)
        self.assertEqual(nearest_half_up([1, 2, 3], 1), 3)

    def test_closed_both_boundary_and_no_zero_fill(self):
        events = pd.DataFrame({'time': pd.to_datetime(['2026-03-01T00:00:00Z', '2026-03-01T00:01:00Z', '2026-03-01T00:03:01Z']),
                               'side': ['buy', 'sell', 'buy'], 'qty': [1., 2., 3.], 'notional': [1., 2., 100.]})
        m = minute_features_from_events(events)
        self.assertEqual(m.index[0], pd.Timestamp('2026-03-01T00:00:00Z'))
        self.assertEqual(m.small_qty.tolist(), [3., 2., 0.])
        self.assertNotIn(pd.Timestamp('2026-03-01T00:02:00Z'), m.index)
        late = events.iloc[:1].copy()
        late['time'] = pd.Timestamp('2026-02-28T23:59:59.999Z')
        with_late = minute_features_from_events(pd.concat([late, events]))
        self.assertEqual(with_late.index[0], pd.Timestamp('2026-02-28T23:59:00Z'))

    def test_raw_event_merge_and_gap_rejection(self):
        d = pd.DataFrame({'id': [1, 2, 3], 'timestamp_ms': [1000, 1000, 1001],
                          'price': [2., 3., 4.], 'qty': [1., 2., 1.], 'side': ['BUY', 'BUY', 'SELL']})
        e = aggregate_raw_trades(d)
        self.assertEqual(e.qty.tolist(), [3., 1.])
        self.assertEqual(e.notional.tolist(), [8., 4.])
        pd.testing.assert_frame_equal(e, aggregate_raw_trades(pd.concat([d, d.iloc[:1]])))
        with self.assertRaises(ValueError): aggregate_raw_trades(d.iloc[[0, 2]])
        bad = d.copy(); bad.loc[1, 'id'] = 1
        with self.assertRaises(ValueError): aggregate_raw_trades(bad)

    def test_roll_before_day_filter(self):
        rng = np.random.default_rng(9)
        m = pd.DataFrame({'big': rng.uniform(-1, 1, 1441), 'ultra': rng.uniform(-1, 1, 1441),
                          'small_qty': rng.integers(0, 100, 1441)},
                         index=pd.date_range('2026-02-28T23:59:00Z', periods=1441, freq='min'))
        result = daily_factors(m, '2026-03-01')
        clipped = daily_factors(m.iloc[1:], '2026-03-01')
        self.assertEqual(result['boundary_minute_count'], 1)
        self.assertEqual(result['minute_count'], 1440)
        self.assertNotAlmostEqual(result[MAIN], clipped[MAIN], places=10)
        self.assertNotAlmostEqual(result[SMALL], clipped[SMALL], places=10)
        self.assertFalse(result['source_certified'])
        for bad in (m.iloc[::-1], pd.concat([m, m.iloc[-1:]])):
            with self.assertRaises(ValueError): daily_factors(bad, '2026-03-01')


if __name__ == '__main__':
    unittest.main()
