"""Regression coverage for repeated timeouts leaking historical archive ZIPs."""
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from data_center import lagged_factor_worker as worker


class BackfillResourceTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'archive_work').mkdir()
        self.universe=self.root/'universe.json';self.universe.write_text('{}')

    def test_timed_out_child_does_not_leave_download(self):
        def run(command,**kwargs):
            work=Path(command[command.index('--work-dir')+1]);(work/'raw.zip').write_bytes(b'partial')
            raise subprocess.TimeoutExpired(command,kwargs['timeout'])
        with patch.object(worker.subprocess,'run',side_effect=run):
            with self.assertRaises(worker.WorkerFailure) as caught:
                worker.isolated_compute(self.root,'AUSDT','2026-10-25',1)
        self.assertEqual(caught.exception.report['status'],'job_timeout')
        self.assertGreater(caught.exception.report['retry_after_epoch'],time.time()+5*3600)
        self.assertEqual(list((self.root/'archive_work').iterdir()),[])

    def test_real_subprocess_timeout_is_killed_and_workspace_removed(self):
        actual=subprocess.run
        def spawn(command,**kwargs):
            work=command[command.index('--work-dir')+1]
            code='from pathlib import Path;import sys,time;Path(sys.argv[1],"raw.zip").write_bytes(b"partial");time.sleep(10)'
            return actual([sys.executable,'-c',code,work],**kwargs)
        with patch.object(worker.subprocess,'run',side_effect=spawn):
            with self.assertRaises(worker.WorkerFailure):
                worker.isolated_compute(self.root,'AUSDT','2026-10-25',.15)
        self.assertEqual(list((self.root/'archive_work').iterdir()),[])

    def test_killed_child_does_not_abort_parent_or_leak_temp(self):
        with patch.object(worker.subprocess,'run',return_value=SimpleNamespace(returncode=-9,stdout='')):
            with self.assertRaises(worker.WorkerFailure) as caught:
                worker.isolated_compute(self.root,'AUSDT','2026-10-25')
        self.assertEqual(caught.exception.report['status'],'worker_failed')
        self.assertEqual(list((self.root/'archive_work').iterdir()),[])

    def test_successful_child_preserves_result_and_cleans_up(self):
        row={'archive_bytes':10,'candidate_only':False}
        with patch.object(worker.subprocess,'run',return_value=SimpleNamespace(returncode=0,
                stdout=json.dumps({'status':'ok','row':row}))):
            self.assertEqual(worker.isolated_compute(self.root,'AUSDT','2026-10-25'),row)
        self.assertEqual(list((self.root/'archive_work').iterdir()),[])

    def test_large_archive_rejected_before_body_download(self):
        checksum=io.BytesIO(b'a'*64)
        class Huge(io.BytesIO):
            headers={'Content-Length':str(300004692)}
            def read(self,*args):raise AssertionError('must reject before download')
        with patch.object(worker,'urlopen',side_effect=[checksum,Huge()]):
            with self.assertRaisesRegex(ValueError,'size cap'):
                worker.download('IRYSUSDT','2026-10-23',self.root)

    def test_backfill_gate_refuses_to_compete_below_35_gib(self):
        from deploy import seoul_lagged_backfill_gate as gate
        with patch.object(gate.shutil,'disk_usage',return_value=SimpleNamespace(free=34*1024**3)), \
             patch.object(gate,'base_gate') as inherited:
            self.assertEqual(gate.main(),1)
            inherited.assert_not_called()

    def test_low_disk_reserves_headroom_for_live_producers(self):
        with patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=34*1024**3)), \
             patch.object(worker,'download') as download:
            with self.assertRaisesRegex(RuntimeError,'35GiB'):
                worker.compute(self.root,'AUSDT','2026-10-25')
            download.assert_not_called()

    def orphan(self, name='v3-archive-test'):
        folder=self.root/'archive_work'/name;folder.mkdir()
        (folder/'job_lease.json').write_text(json.dumps({'schema':'backfill_temporary_v1',
            'parent_pid':999999999,'symbol':'AUSDT','day':'2026-10-25'}))
        (folder/'AUSDT-trades-2026-10-25.zip').write_bytes(b'partial')
        return folder

    def test_orphaned_leased_job_is_reclaimed_after_hard_kill(self):
        folder=self.orphan()
        self.assertEqual(worker.cleanup_orphans(self.root),[folder.name])
        self.assertFalse(folder.exists())

    def test_active_and_unrecognized_folders_are_preserved(self):
        import os
        folder=self.orphan()
        p=folder/'job_lease.json';lease=json.loads(p.read_text());lease['parent_pid']=os.getpid()
        p.write_text(json.dumps(lease))
        other=self.orphan('v3-archive-unrecognized');(other/'important.csv').write_text('untouched')
        self.assertEqual(worker.cleanup_orphans(self.root),[])
        self.assertTrue(folder.exists());self.assertTrue(other.exists())

    def test_cleanup_refuses_symlinks(self):
        folder=self.orphan();(folder/'linked.zip').symlink_to(self.universe)
        self.assertEqual(worker.cleanup_orphans(self.root),[])
        self.assertTrue(self.universe.exists())

    def arguments(self):
        return ['worker','--root',str(self.root),'--universe',str(self.universe),
                '--day','2026-10-25','--max-jobs','2']

    def test_checkpoint_before_child_and_failure_does_not_block_next_coin(self):
        jobs=[('BADUSDT','2026-10-25'),('GOODUSDT','2026-10-25')]
        def compute(root,symbol,day,timeout):
            state=json.loads((root/'backfill_status.json').read_text())
            self.assertEqual(state['items'][day+'/'+symbol]['status'],'running')
            if symbol=='BADUSDT':
                raise worker.WorkerFailure({'status':'job_timeout','retry_after_epoch':time.time()+6*3600})
            return {'candidate_only':False,'job_seconds':1,'missing_trade_ids':0,'archive_bytes':10}
        with patch('sys.argv',self.arguments()),patch.object(worker,'jobs',return_value=jobs), \
             patch.object(worker,'isolated_compute',side_effect=compute) as run:
            worker.main();self.assertEqual(run.call_count,2)
        state=json.loads((self.root/'backfill_status.json').read_text())
        self.assertEqual(state['items']['2026-10-25/BADUSDT']['status'],'job_timeout')
        self.assertEqual(state['items']['2026-10-25/GOODUSDT']['status'],'computed_final')
        with patch('sys.argv',self.arguments()),patch.object(worker,'jobs',return_value=jobs[:1]), \
             patch.object(worker,'isolated_compute') as run:
            worker.main();run.assert_not_called()

    def test_sigterm_persists_inflight_backoff(self):
        def stop(*args):signal.raise_signal(signal.SIGTERM)
        with patch('sys.argv',self.arguments()), \
             patch.object(worker,'jobs',return_value=[('BADUSDT','2026-10-25')]), \
             patch.object(worker,'isolated_compute',side_effect=stop):
            with self.assertRaises(SystemExit):worker.main()
        row=json.loads((self.root/'backfill_status.json').read_text())['items']['2026-10-25/BADUSDT']
        self.assertEqual(row['status'],'running')
        self.assertGreater(row['retry_after_epoch'],time.time()+5*3600)

    def test_missed_publication_is_reported_as_failure(self):
        from datetime import datetime,timezone
        from deploy import top50_boundary_publish as publisher
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return datetime(2026,10,26,0,26,tzinfo=timezone.utc)
        (self.root/'inputs').mkdir();(self.root/'inputs/btc_hourly.json').write_text('{}')
        self.universe.write_text(json.dumps({'tradable_symbols':[]}))
        with patch.object(publisher,'ROOT',self.root),patch.object(publisher,'UNIVERSE',self.universe), \
             patch.object(publisher,'datetime',Clock),patch.object(publisher,'readiness',return_value=(set(),{})):
            with self.assertRaises(SystemExit) as caught:publisher.main()
        self.assertEqual(caught.exception.code,2)


if __name__=='__main__':unittest.main()
