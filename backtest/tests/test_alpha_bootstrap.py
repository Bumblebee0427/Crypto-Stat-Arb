import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research.optimizer.alpha import fit_alpha, predict_alpha
from research.optimizer.alpha_bootstrap import alpha_policy, validated_history
from research.optimizer.contracts import ResearchConfig, file_hash, digest, write_json
from research.optimizer.v3_history import V3


def panel():
    rows = []
    for t in pd.date_range('2026-01-01', periods=15):
        for side in [-1, 1]:
            for k in range(3):
                rows.append({'asof': t, 'symbol': str(side)+'_'+str(k), 'side': side,
                    'formula_version': V3, 'main_rank': (k-1)/2, 'small_rank': (1-k)/2,
                    'future_return': .001*(k-1), 'entry_at': t+pd.Timedelta(hours=1),
                    'feature_available_at': t+pd.Timedelta(minutes=8),
                    'exit_at': t+pd.Timedelta(days=1, hours=1),
                    'label_available_at': t+pd.Timedelta(days=1, hours=2)})
    return pd.DataFrame(rows)


class AlphaBootstrapTest(unittest.TestCase):
    def release(self, root, accepted=True):
        folder = root/'alpha_bootstrap'; folder.mkdir()
        panel().to_parquet(folder/'historical_panel.parquet', index=False)
        report = {'accepted': accepted, 'checks': {'temporal': accepted},
                  'alpha_policy':alpha_policy(),
                  'panel_sha256':file_hash(folder/'historical_panel.parquet'),
                  'validation_label_available_at': '2026-01-17', 'test_end_inclusive': '2026-01-15'}
        write_json(folder/'validation.json', report)
        body = {'policy': alpha_policy(), 'panel_sha256': file_hash(folder/'historical_panel.parquet'),
                'validation_sha256': file_hash(folder/'validation.json')}
        write_json(folder/'release.json', {**body, 'release_id': digest(body)})
        return folder

    def test_history_requires_passed_immutable_past_validation(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); self.assertEqual(validated_history(root, '2026-02-01'), (None, None))
            folder = self.release(root)
            loaded, info = validated_history(root, '2026-02-01')
            self.assertEqual(len(loaded), 90); self.assertEqual(info['historical_days'], 15)
            with self.assertRaisesRegex(ValueError, 'future validation'):
                validated_history(root, '2026-01-16')
            with (folder/'historical_panel.parquet').open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed after approval'):
                validated_history(root, '2026-02-01')

    def test_failed_validation_cannot_load(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); self.release(root, accepted=False)
            with self.assertRaisesRegex(ValueError, 'did not pass'):
                validated_history(root, '2026-02-01')

    def test_future_observations_and_immature_labels_do_not_change_fit(self):
        original = panel(); cutoff = pd.Timestamp('2026-01-16')
        cfg = ResearchConfig(train_days=15)
        expected = fit_alpha(original, cutoff, cfg)
        # Even falsely dated maturity cannot introduce observations after fitting.
        future = original.copy(); future['asof'] += pd.Timedelta(days=30)
        future['label_available_at'] = cutoff-pd.Timedelta(days=1)
        future['future_return'] = 1000.
        changed = pd.concat([original, future], ignore_index=True)
        changed.loc[changed.label_available_at > cutoff, 'future_return'] = 99999.
        actual = fit_alpha(changed, cutoff, cfg)
        for side in [-1, 1]:
            np.testing.assert_allclose(actual.models[side].coefficients, expected.models[side].coefficients)
        self.assertLessEqual(actual.label_cutoff, cutoff)


if __name__ == '__main__': unittest.main()
