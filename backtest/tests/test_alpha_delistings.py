import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile
import pandas as pd
from research.optimizer.delistings import index_settlement_bounds


class IndexBoundsTest(unittest.TestCase):
    def test_exact_window_counts_and_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'index.zip';times=pd.date_range('2026-07-02 08:30',periods=30,freq='min')
            def save(count):
                rows=[f'{t.value//1000000},10,12,8,11,0,0,0,{count},0,0,0' for t in times]
                with zipfile.ZipFile(path,'w') as z:z.writestr('index.csv','\n'.join(rows))
                path.with_name(path.name+'.CHECKSUM').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+' index.zip')
            save(60)
            event=index_settlement_bounds(path,'AUSDT','2026-07-02 09:00','synthetic-announcement','synthetic-rule')
            self.assertEqual(event['lower'],8.);self.assertEqual(event['upper'],12.)
            self.assertFalse(event['exact_settlement_price_known'])
            save(59)
            with self.assertRaisesRegex(ValueError,'incomplete 1800'):index_settlement_bounds(path,'AUSDT','2026-07-02 09:00','','')
