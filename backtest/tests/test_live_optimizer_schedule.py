"""Verify producer deadlines and explicit failed-artifact delivery without network."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from tools import build_live_optimizer_artifact as builder
from tools import run_live_optimizer as producer

NOW=datetime(2026,10,24,0,20,tzinfo=timezone.utc)


class Clock(datetime):
    @classmethod
    def now(cls,tz=None):return NOW


class LiveScheduleTest(unittest.TestCase):
    def test_builder_accepts_late_fresh_inputs_with_shared_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);boundary=NOW.replace(minute=0)
            signal=builder.seal({'status':'complete','window_end':boundary.isoformat()})
            obs={'checked_at':NOW.isoformat(),'base_snapshot_id':signal['snapshot_id'],
                 'positions':{},'equity':3000,'marks':{}}
            for name,value in [('signal',signal),('obs',obs),('inputs',{
                'returns':'r','factors':'f','settlements':'s','provenance':'provenance.json'})]:
                (root/(name+'.json')).write_text(json.dumps(value))
            (root/'provenance.json').write_text('{}')
            args=['builder','--inputs',str(root/'inputs.json'),'--signal',str(root/'signal.json'),
                  '--observation',str(root/'obs.json'),'--out',str(root/'out.json')]
            with patch('sys.argv',args),patch.object(builder,'datetime',Clock), \
                 patch.object(builder.pd,'read_parquet',return_value=pd.DataFrame()), \
                 patch.object(builder,'allocation',return_value={'allocations':{},
                     'maximum_available_at':boundary.isoformat()}):
                self.assertEqual(builder.main(),0)
            out=json.loads((root/'out.json').read_text())
            self.assertEqual(datetime.fromisoformat(out['valid_until']),boundary+timedelta(minutes=25))
            self.assertEqual(out['status'],'ready')

    def test_blocked_computation_is_delivered_to_allow_immediate_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);boundary=NOW.replace(minute=0)
            (root/'returns.parquet').touch();(root/'inputs.json').write_text('{}')
            config={'data_root':str(root),'trade_root':'/test/trade','trade_host':'test'}
            (root/'config.json').write_text(json.dumps(config))
            signal={'status':'complete','window_end':boundary.isoformat(),
                    'snapshot_id':'test-base','allocations':{'A':.5,'B':-.5}}
            def built(command,**kwargs):
                target=Path(command[command.index('--out')+1])
                target.write_text(json.dumps({'status':'blocked','model':'test-model',
                    'snapshot_id':'test-artifact','allocations':{}}))
                return subprocess.CompletedProcess(command,2)
            with patch('sys.argv',['producer','--config',str(root/'config.json'),'--skip-refresh']), \
                 patch.object(producer,'datetime',Clock), \
                 patch.object(producer.pd,'read_parquet',return_value=pd.DataFrame({'A':[1]})), \
                 patch.object(pd.DataFrame,'to_parquet'), \
                 patch.object(producer,'signal',side_effect=[RuntimeError('temporary SSH error'),signal]), \
                 patch.object(producer.time,'sleep'), \
                 patch.object(producer,'settlements',return_value=[]), \
                 patch.object(producer,'archive_v3_alpha'), \
                 patch.object(producer,'observation',return_value={}), \
                 patch.object(producer.subprocess,'run',side_effect=built), \
                 patch.object(producer,'remote',return_value={'status':'delivered'}) as deliver:
                producer.main()
            deliver.assert_called_once()
            summary=json.loads((root/'last_run.json').read_text())
            self.assertEqual(summary['artifact_status'],'blocked')
            self.assertFalse(summary['orders_submitted'])
            self.assertIn("'status': 'blocked'",deliver.call_args.args[2])


if __name__=='__main__':unittest.main()
