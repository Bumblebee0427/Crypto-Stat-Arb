from datetime import datetime, timezone, timedelta
from decimal import Decimal
import tempfile
import unittest

from data_center.consumer import live_budget, LIVE_GROSS_CAP_USDT
from data_center.contract import seal
from data_center.optimizer_overlay import apply, load_and_apply, positions_digest, should_wait, awaiting_artifact, MODEL, SCHEMA

NOW = datetime(2026, 10, 22, 0, 4, tzinfo=timezone.utc)


def inputs():
    boundary = NOW.replace(minute=0)
    snapshot = seal({'snapshot_id': 'unused', 'window_end': boundary.isoformat(),
        'generated_at': NOW.isoformat(), 'allocations': {'AUSDT': .25, 'BUSDT': .25, 'CUSDT': -.5}, 'gross_allocation': 1.})
    artifact = seal({'schema_version': SCHEMA, 'model': MODEL, 'blend': .25,
        'status': 'ready', 'diagnostic_only': False, 'base_snapshot_id': snapshot['snapshot_id'],
        'asof': boundary.isoformat(), 'generated_at': NOW.isoformat(),
        'valid_until': (boundary+timedelta(minutes=25)).isoformat(),
        'max_observation_at': boundary.isoformat(), 'positions_digest': positions_digest({'AUSDT': '1'}),
        'budget_usdt': 1500., 'allocations': {'AUSDT': .275, 'BUSDT': .225, 'CUSDT': -.5}})
    return snapshot, artifact


class LiveOptimizerTest(unittest.TestCase):
    def test_added_capital_keeps_original_equity_fraction(self):
        self.assertEqual(live_budget(3082), 1500)
        self.assertEqual(live_budget(2400), 1200)
        self.assertEqual(live_budget(1000), 500)
        self.assertEqual(LIVE_GROSS_CAP_USDT, 3000)

    def test_valid_overlay_preserves_pool_and_side_budgets(self):
        s, a = inputs(); out = apply(s, a, {'AUSDT': Decimal('1')}, 1500, NOW)
        self.assertEqual(out['allocations'], a['allocations'])
        self.assertEqual(out['base_snapshot_id'], s['snapshot_id'])
        self.assertEqual(out['gross_allocation'], 1.)

    def test_untrusted_stale_future_and_changed_positions_are_rejected(self):
        s, a = inputs()
        for patch in [{'base_snapshot_id': 'wrong'}, {'diagnostic_only': True},
                      {'generated_at': (NOW+timedelta(seconds=1)).isoformat()},
                      {'max_observation_at': (NOW+timedelta(days=1)).isoformat()},
                      {'positions_digest': 'wrong'}, {'budget_usdt': 500},
                      {'allocations': {'AUSDT': .3, 'BUSDT': .2, 'CUSDT': -.5}},
                      {'allocations': {'AUSDT': .275, 'BUSDT': .225, 'CUSDT': -.49}}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                apply(s, seal({**a, **patch}), {'AUSDT': '1'}, 1500, NOW)
        with self.assertRaises(ValueError):
            apply(s, a, {'AUSDT': '1'}, 1500, NOW.replace(minute=25))

    def test_missing_artifact_keeps_base_signal(self):
        s, _ = inputs()
        with tempfile.TemporaryDirectory() as root:
            out, info = load_and_apply(s, root, {}, 1500, NOW)
        self.assertEqual(out, s)
        self.assertEqual(info['status'], 'baseline_fallback')

    def test_wait_is_explicit_and_bounded_before_execution_cutoff(self):
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(should_wait({}, root, NOW))
            path=Path(root)/'optimizer/enabled.json';path.parent.mkdir()
            path.write_text(json.dumps({'enabled':True,'model':MODEL}))
            s, _ = inputs()
            self.assertTrue(should_wait({}, root, NOW, s))
            self.assertFalse(should_wait({'status':'baseline_fallback'}, root, NOW, s))
            self.assertFalse(should_wait({'status':'applied'}, root, NOW))
            self.assertFalse(should_wait({}, root, NOW.replace(minute=6), s))
            self.assertFalse(should_wait({}, root, NOW.replace(minute=25), s))
            path.write_text('[]')
            self.assertFalse(should_wait({}, root, NOW, s))

    def test_failed_current_artifact_falls_back_without_waiting(self):
        import json
        from pathlib import Path
        s, a = inputs()
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/'optimizer';folder.mkdir()
            (folder/'enabled.json').write_text(json.dumps({'enabled':True,'model':MODEL}))
            self.assertTrue(awaiting_artifact(s,root,NOW))
            file=folder/('target_'+s['window_end'][:10]+'.json')
            for a in [seal({**a,'status':'blocked','allocations':{}}), {'broken':True}, []]:
                file.write_text(json.dumps(a))
                self.assertFalse(awaiting_artifact(s,root,NOW))
                out, info=load_and_apply(s,root,{},1500,NOW)
                self.assertEqual(out,s)
                self.assertEqual(info['status'],'baseline_fallback')

    def test_late_signal_gets_two_minutes_but_never_past_decision(self):
        import json
        from pathlib import Path
        s,_=inputs();s=seal({**s,'generated_at':NOW.replace(minute=24).isoformat()})
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/'optimizer';folder.mkdir()
            (folder/'enabled.json').write_text(json.dumps({'enabled':True,'model':MODEL}))
            self.assertTrue(awaiting_artifact(s,root,NOW.replace(minute=24,second=44)))
            self.assertFalse(awaiting_artifact(s,root,NOW.replace(minute=24,second=45)))
            self.assertFalse(awaiting_artifact(s,root,NOW.replace(minute=25)))


if __name__ == '__main__':
    unittest.main()
