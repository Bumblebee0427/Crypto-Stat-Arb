from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deploy import seoul_lagged_bootstrap_minutes as repair
from deploy import seoul_lagged_gap_watch as watch


class GapRepairTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'v3';self.stage=self.base/'stage'
        self.universe=self.base/'universe.json';self.universe.write_text(json.dumps({'tradable_symbols':['ADAUSDT']}))
        for name,value in [('ROOT',self.root),('STAGED',self.stage),('UNIVERSE',self.universe)]:
            p=patch.object(repair,name,value);p.start();self.addCleanup(p.stop)
        self.day='2026-10-01';self.minutes=self.root/'minutes'/self.day;self.minutes.mkdir(parents=True)

    def warm(self,missing):
        for n in range(60):
            if n not in missing:(self.minutes/f'00{n:02d}.json').write_text('{"status":"provisional"}')

    def hour(self,hour,lines):
        p=self.stage/self.day/'hours'/f'{hour:02d}'/'ADAUSDT.csv';p.parent.mkdir(parents=True);p.write_text(lines)

    def stamp(self,h,m,s=0):
        return int(datetime(2026,10,1,h,m,s,tzinfo=timezone.utc).timestamp()*1000)

    def test_minute59_includes_next_hour_exact_boundary(self):
        self.warm({59})
        before=(self.minutes/'0000.json').read_bytes()
        self.hour(0,f'1,{self.stamp(0,59,30)},10,2,False\n')
        self.hour(1,f'2,{self.stamp(1,0)},10,3,False\n')
        with patch.object(repair,'threshold',return_value=((1,5,15),'source-sha')):
            result=repair.fill(self.day,0)
        row=json.loads((self.minutes/'0059.json').read_text())
        self.assertEqual(result['minutes_written'],1)
        self.assertEqual(row['components']['ADAUSDT']['ultra_buy'],50)
        self.assertEqual(row['trade_counts']['ADAUSDT'],1)
        self.assertEqual(set(row['threshold_source_sha256']),{'2026-10-01/00/ADAUSDT','2026-10-01/01/ADAUSDT'})
        self.assertEqual((self.minutes/'0000.json').read_bytes(),before)

    def test_complete_hour_requires_no_source_reads(self):
        self.warm(set())
        with patch.object(repair.pd,'read_csv',side_effect=AssertionError('must not reread')):
            self.assertEqual(repair.fill(self.day,0)['minutes_written'],0)

    def test_absent_boundary_stage_defers_without_false_marker(self):
        self.warm({59});self.hour(0,f'1,{self.stamp(0,59,30)},10,2,False\n')
        with self.assertRaisesRegex(ValueError,'not ready'):repair.fill(self.day,0)
        self.assertFalse((self.minutes/'0059.json').exists())

    def test_waiting_marker_is_repaired(self):
        self.warm({10});(self.minutes/'0010.json').write_text('{"status":"waiting"}')
        self.hour(0,f'1,{self.stamp(0,10,30)},10,2,False\n')
        with patch.object(repair,'threshold',return_value=((1,5,15),'source-sha')):
            repair.fill(self.day,0)
        self.assertEqual(json.loads((self.minutes/'0010.json').read_text())['status'],'provisional')

    def test_partial_hour_repairs_only_minutes_before_stable_cutoff(self):
        self.warm({10,11})
        self.hour(0,f'1,{self.stamp(0,10,30)},10,2,False\n2,{self.stamp(0,11,30)},10,99,False\n')
        with patch.object(repair,'threshold',return_value=((1,5,15),'source-sha')):
            result=repair.fill(self.day,0,through_minute=10)
        self.assertEqual(result['minutes_written'],1)
        self.assertFalse((self.minutes/'0011.json').exists())
        self.assertEqual(json.loads((self.minutes/'0010.json').read_text())['components']['ADAUSDT']['ultra_buy'],20)

    def test_automatic_watch_never_rewrites_a_sealed_day(self):
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return datetime(2026,10,2,1,tzinfo=timezone.utc)
        marker=self.root/'inputs/live_daily'/self.day/'complete.json';marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({'day':self.day,'status':'complete','formula_version':watch.FORMULA_VERSION}))
        with patch.object(watch,'ROOT',self.root),patch.object(watch,'datetime',Clock), \
             patch.object(watch.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**5)), \
             patch('sys.argv',['watch','--day',self.day]),patch.object(watch,'fill') as fill:
            watch.main();fill.assert_not_called()
        self.assertEqual(json.loads((self.root/'gap_repair_status.json').read_text())['results'][0]['status'],'sealed_skipped')

    def waiting(self,n,hour):
        instant=datetime(2026,10,1,0,n,tzinfo=timezone.utc)
        row={'status':'waiting','minute':instant.isoformat(),'formula_version':repair.FORMULA_VERSION,
             'minute_code_sha256':repair.CODE_SHA256,'components':{'ADAUSDT':{'big_buy':1}},
             'trade_counts':{'ADAUSDT':1},'threshold_source_sha256':{'2026-10-01/00/ADAUSDT':'sha'},
             'invalid_records_in_lookback':0,'missing_thresholds':[{'symbol':'BADUSDT','hour':hour,'reason':'ValueError'}]}
        (self.minutes/f'00{n:02d}.json').write_text(json.dumps(row));return row

    def test_error_from_unrelated_hour_does_not_exclude_valid_components(self):
        self.warm({10});before=self.waiting(10,'2026-10-01T01:00:00+00:00')
        result=repair.fill(self.day,0,through_minute=10)
        after=json.loads((self.minutes/'0010.json').read_text())
        self.assertEqual(result['normalized_minutes'],1)
        self.assertEqual(after['status'],'provisional')
        self.assertEqual(after['components'],before['components'])

    def test_next_hour_error_still_blocks_its_minute59_boundary(self):
        self.warm({59});self.waiting(59,'2026-10-01T01:00:00+00:00')
        with self.assertRaisesRegex(ValueError,'not ready'):repair.fill(self.day,0)
        self.assertEqual(json.loads((self.minutes/'0059.json').read_text())['status'],'waiting')

    def test_oversized_coin_is_isolated_without_losing_good_components(self):
        self.universe.write_text(json.dumps({'tradable_symbols':['ADAUSDT','BADUSDT']}))
        self.warm({10})
        self.hour(0,f'1,{self.stamp(0,10,30)},10,2,False\n')
        bad=self.stage/self.day/'hours/00/BADUSDT.csv'
        with bad.open('wb') as f:f.truncate(repair.MAX_RECOVERY_HOUR_BYTES+1)
        with patch.object(repair,'threshold',return_value=((1,5,15),'source-sha')):
            repair.fill(self.day,0)
        row=json.loads((self.minutes/'0010.json').read_text())
        self.assertEqual(row['status'],'waiting')
        self.assertEqual(row['components']['ADAUSDT']['ultra_buy'],20)
        self.assertEqual(row['missing_thresholds'][0]['symbol'],'BADUSDT')
        self.assertIn('memory_budget',row['missing_thresholds'][0]['reason'])


if __name__=='__main__':unittest.main()
