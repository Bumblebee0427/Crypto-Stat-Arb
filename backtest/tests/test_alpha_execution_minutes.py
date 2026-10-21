import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.prepare_alpha_execution_minutes import extract


class OfficialMinuteTest(unittest.TestCase):
    def test_checksums_and_fixed_window_vwap(self):
        import pandas as pd
        stamp = int(pd.Timestamp('2026-01-01', tz='UTC').timestamp()*1000)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'AUSDT-1m-2026-01.zip'
            csv = 'open_time,open,high,low,close,volume,close_time,quote_volume,count,buy_qty,buy_quote,ignore\n'
            for minute, qty, notional in [(9,2,10),(10,0,0),(11,5,100),(59,3,44),(60,2,999)]:
                t = stamp+minute*60000
                csv += f'{t},1,1,1,1,{qty},{t+59999},{notional},1,1,1,0\n'
            with zipfile.ZipFile(path,'w') as z: z.writestr('AUSDT.csv',csv)
            checksum = path.with_name(path.name+'.CHECKSUM')
            checksum.write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name)
            symbol, rows, proof = extract(('AUSDT',path))
            self.assertEqual(symbol,'AUSDT'); self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['price'],18.)
            checksum.write_text('0'*64+'  '+path.name)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'): extract(('AUSDT',path))


if __name__ == '__main__': unittest.main()

class OfficialHourlyRepairTest(unittest.TestCase):
    def test_incomplete_hours_are_not_repaired(self):
        import pandas as pd
        from tools.repair_alpha_market_history import hourly
        stamp=int(pd.Timestamp('2026-01-01',tz='UTC').timestamp()*1000)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'AUSDT-1m-2026-01.zip'
            rows=[]
            for minute in range(119):
                t=stamp+minute*60000
                rows.append(f'{t},1,1,1,2,3,{t+59999},6,1,1,1,0')
            with zipfile.ZipFile(path,'w') as z:z.writestr('AUSDT.csv','\n'.join(rows))
            path.with_name(path.name+'.CHECKSUM').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+' '+path.name)
            symbol,frame,proof=hourly(('AUSDT',path))
            self.assertEqual(symbol,'AUSDT');self.assertEqual(len(frame),1)
            self.assertEqual(frame.iloc[0].quote_volume,360.)
            self.assertEqual(frame.iloc[0].close,2.)
