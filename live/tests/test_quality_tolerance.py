from datetime import datetime,timezone,timedelta
import tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import unittest
from data_center.base_strategy import FORMULA_VERSION,SOURCE_VERSION
from data_center.quality_tolerance import assess
from data_center.day_boundaries import evidence
from data_center.contract import atomic_json
from data_center.publisher import build
from data_center.base_strategy import MAIN,SMALL

NOW=datetime(2026,9,17,4,tzinfo=timezone.utc)


def row(n=200,gaps=0):
    return {'symbol':'ADAUSDT','day':'2026-09-16','status':'final','formula_version':FORMULA_VERSION,
            'source_version':SOURCE_VERSION,'trade_rows':n,'missing_trade_ids':gaps,
            'first_trade_id':1,'last_trade_id':n+gaps,'available_at':NOW.isoformat(),
            'archive_sha256':'a'*64,'partition_basis':'exchange_time_utc_day',
            'archive_url':'https://data.binance.vision/data/futures/um/daily/trades/ADAUSDT/ADAUSDT-trades-2026-09-16.zip'}


class QualityToleranceTest(unittest.TestCase):
    def test_adjacent_day_anchor_includes_unobserved_head(self):
        r=row();r.pop('archive_url');r.pop('archive_sha256')
        r.update(first_trade_id=1001,last_trade_id=1200,time_monotonic_by_id=True)
        def anchor(day,ident):
            return {'kind':'observed_neighbor_trade','symbol':'ADAUSDT','day':day,'id':ident,
                    'timestamp_ms':int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()*1000),
                    'raw_segment_sha256':'a'*64}
        r['day_boundary_evidence']={'previous':anchor('2026-09-15',1000),'next':anchor('2026-09-17',1201)}
        self.assertTrue(assess(r,'ADAUSDT','2026-09-16',NOW)['passes_tolerance'])
        r['day_boundary_evidence']['previous']['id']=0
        a=assess(r,'ADAUSDT','2026-09-16',NOW)
        self.assertEqual(a['internal_id_fraction_upper_bound'],0.)
        self.assertAlmostEqual(a['missing_fraction_upper_bound'],1000/1200)
        self.assertFalse(a['passes_tolerance'])
        r['day_boundary_evidence']['previous']['symbol']='XRPUSDT'
        self.assertFalse(assess(r,'ADAUSDT','2026-09-16',NOW)['bounded_day_reference'])

    def test_edges_from_real_parquet_statistics_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for d,ids in [('2026-09-15',[1,2]),('2026-09-17',[8,9])]:
                folder=root/'staged'/d/'ADAUSDT';folder.mkdir(parents=True)
                ms=int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp()*1000)
                pq.write_table(pa.table({'id':ids,'timestamp_ms':[ms,ms+1]}),folder/('a'*64+'-0.parquet'))
            e=evidence(root,'ADAUSDT','2026-09-16')
            self.assertEqual(e['previous']['id'],2);self.assertEqual(e['next']['id'],8)

    def test_top50_tolerates_small_bounded_gap_but_base_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);universe=root/'universe.json'
            atomic_json(universe,{'tradable_symbols':['ADAUSDT'],'fetched_at':NOW.isoformat()})
            end=NOW.replace(hour=0);near=end-timedelta(hours=2);far=near-timedelta(days=30)
            atomic_json(root/'inputs/btc_hourly.json',{'closes':{near.isoformat():110,far.isoformat():100},
                         'first_candle_by_symbol':{'ADAUSDT':'2020-01-01T00:00:00+00:00'}})
            for n in range(12):
                day=(end-timedelta(days=n+1)).date().isoformat()
                r=row(200,1);r.update(day=day,source_complete=False,minute_count=1440)
                r[MAIN]=.1;r[SMALL]=.2
                r['archive_url']=f'https://data.binance.vision/data/futures/um/daily/trades/ADAUSDT/ADAUSDT-trades-{day}.zip'
                atomic_json(root/'inputs/daily'/day/'ADAUSDT.json',r)
            atomic_json(root/'inputs/volume25/2026-09-16/ADAUSDT.json',{'symbol':'ADAUSDT','day':'2026-09-16',
                'complete':True,'hours':599,'window_end':(end-timedelta(hours=1)).isoformat(),'target_window_end':end.isoformat(),'volume_lag_hours':1,'available_at':NOW.isoformat(),'quote_volume_mean':100.})
            top=build(root,universe,'2026-09-16',NOW,profile='top50')
            self.assertEqual(top['coverage']['observed'],1)
            self.assertEqual(top['status'],'blocked')  # no source/formula receipt
            self.assertEqual(build(root,universe,'2026-09-16',NOW)['coverage']['observed'],0)

    def test_strict_threshold_and_reference_scope(self):
        self.assertFalse(assess(row(8,1),'ADAUSDT','2026-09-16',NOW)['passes_tolerance'])
        self.assertFalse(assess(row(1,1),'ADAUSDT','2026-09-16',NOW)['passes_tolerance'])
        r=assess(row(9,1),'ADAUSDT','2026-09-16',NOW)
        self.assertTrue(r['passes_tolerance']);self.assertFalse(r['live_ready'])

    def test_relaxed_threshold_still_requires_full_day_evidence(self):
        r=row(985,15)
        self.assertTrue(assess(r,'ADAUSDT','2026-09-16',NOW)['passes_tolerance'])
        r.pop('archive_url')
        self.assertFalse(assess(r,'ADAUSDT','2026-09-16',NOW)['passes_tolerance'])

    def test_no_internal_holes_cannot_prove_partial_day_complete(self):
        r=row();r.pop('archive_url')
        result=assess(r,'ADAUSDT','2026-09-16',NOW)
        self.assertEqual(result['internal_id_fraction_upper_bound'],0.)
        self.assertIsNone(result['missing_fraction_upper_bound'])
        self.assertFalse(result['passes_tolerance'])

    def test_invalid_counters_reference_and_future_fail_closed(self):
        for patch in ({'trade_rows':True},{'missing_trade_ids':-1},{'last_trade_id':201},
                      {'archive_sha256':'fake'},{'available_at':'2026-09-18T00:00:00+00:00'},
                      {'archive_url':'https://untrusted.example/archive.zip'}):
            self.assertFalse(assess({**row(),**patch},'ADAUSDT','2026-09-16',NOW)['passes_tolerance'])

    def test_observed_placeholders_do_not_count_as_lost_economic_trades(self):
        r={**row(200,1),'sequence_rows':210,'zero_value_sequence_rows':10,'last_trade_id':211}
        a=assess(r,'ADAUSDT','2026-09-16',NOW)
        self.assertAlmostEqual(a['missing_fraction_upper_bound'],1/201)


if __name__=='__main__':unittest.main()
