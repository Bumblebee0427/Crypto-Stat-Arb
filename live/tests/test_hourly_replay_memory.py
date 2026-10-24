"""Parity and failure isolation for memory-bounded live symbol replay."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from deploy import seoul_lagged_daily_shadow as daily
from data_center.lagged_quantile_v3 import daily_from_events, events_from_trades, quantiles


class HourlyReplayTest(unittest.TestCase):
    def test_hour_at_a_time_matches_whole_day_with_boundaries_and_duplicates(self):
        day='2026-10-23';start=pd.Timestamp(day,tz='UTC');rng=np.random.default_rng(240)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=[];ident=1;prior=None
            for h in range(-1,24):
                rows=[]
                for minute in range(60):
                    t=(start+pd.Timedelta(hours=h,minutes=minute)).value//1000000
                    for offset,maker in [(0,False),(0,False),(20000,True)]:
                        rows.append((ident,t+offset,2.,float(rng.integers(1,500)),maker));ident+=1
                rows.append((ident,(start+pd.Timedelta(hours=h,minutes=59,seconds=59)).value//1000000,0.,0.,False));ident+=1
                rows.append(rows[3])
                frame=pd.DataFrame(rows,columns=daily.COLS)
                stamp=start+pd.Timedelta(hours=h)
                path=root/str(stamp.date())/'hours'/f'{stamp.hour:02d}'/'AUSDT.csv';path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,header=False)
                if h==-1:prior=quantiles(events_from_trades(frame))
                else:raw.append(frame)
            all_rows=pd.concat(raw,ignore_index=True)
            expected=daily_from_events(events_from_trades(all_rows),day,prior)
            count=len(all_rows.drop_duplicates('id').loc[lambda x:~(x.price.eq(0)&x.qty.eq(0))])
            with patch.object(daily,'STAGED',root):actual,evidence=daily.replay_symbol(day,'AUSDT',count)
            for key in [daily.MAIN,daily.SMALL]:self.assertAlmostEqual(actual[key],expected[key],places=14)
            self.assertEqual(actual['minute_count'],expected['minute_count'])
            self.assertEqual(evidence['trade_rows'],count)
            self.assertEqual(len(evidence['hourly_source_sha256']),24)
            self.assertEqual(evidence['processing'],'one_utc_hour_at_a_time')
            # An incorrect baseline cannot be silently accepted.
            with patch.object(daily,'STAGED',root):
                with self.assertRaisesRegex(ValueError,'trade count differs'):daily.replay_symbol(day,'AUSDT',count+1)

    def test_large_hour_is_rejected_before_pandas_allocates(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'large.csv';path.write_text('1,1,1,1,False\n')
            with patch.object(daily,'MAX_REPLAY_HOUR_BYTES',1),patch.object(daily.pd,'read_csv') as read:
                with self.assertRaisesRegex(ValueError,'memory budget'):daily.read_hour(path)
                with self.assertRaisesRegex(ValueError,'memory budget'):daily.replay_hour(path,pd.Timestamp('2026-10-23',tz='UTC'))
                read.assert_not_called()

    def test_wrong_hour_and_conflicting_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'hour.csv';start=pd.Timestamp('2026-10-23',tz='UTC');ms=start.value//1000000
            path.write_text(f'1,{ms},1,1,False\n1,{ms},2,1,False\n')
            with self.assertRaisesRegex(ValueError,'conflicting duplicate'):daily.replay_hour(path,start)
            path.write_text(f'1,{ms-1},1,1,False\n')
            with self.assertRaisesRegex(ValueError,'outside replay UTC hour'):daily.replay_hour(path,start)
