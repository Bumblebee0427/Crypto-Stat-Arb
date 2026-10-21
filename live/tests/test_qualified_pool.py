from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_center.base_strategy import MAIN, SMALL, SOURCE_VERSION
from data_center.contract import validate, digest
from data_center.publisher import build
from data_center.top50_strategy import LAGGED_FORMULA_VERSION, LAGGED_STRATEGY_VERSION, select
from data_center.lagged_quantile_v3 import FORMULA_CODE_SHA256
from deploy import seoul_lagged_daily_prefix as prefix
from deploy import seoul_lagged_daily_shadow as daily
from deploy import seoul_lagged_fast_metadata as metadata
from deploy import top50_boundary_publish as boundary

DAY = '2026-10-02'
NOW = datetime(2026, 10, 3, 0, 40, tzinfo=timezone.utc)
END = NOW.replace(minute=0)


class QualifiedPoolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.symbols = [f'C{i}USDT' for i in range(100)]
        self.universe = self.root/'universe.json'
        self.write(self.universe, {'tradable_symbols': self.symbols,
                   'symbol_onboard_dates': {s: '2020-01-01' for s in self.symbols},
                   'fetched_at': NOW.isoformat()})
        near = END-timedelta(hours=2)
        self.write(self.root/'inputs/btc_hourly.json', {'closes': {
            near.isoformat(): 110, (near-timedelta(days=30)).isoformat(): 100},
            'first_candle_by_symbol': {s: '2020-01-01T00:00:00+00:00' for s in self.symbols}})
        for i, symbol in enumerate(self.symbols):
            for offset in range(12):
                day = (END-timedelta(days=offset+1)).date().isoformat()
                self.write(self.path(symbol, day), {
                    'symbol': symbol, 'day': day, 'status': 'final',
                    'source_version': SOURCE_VERSION, 'formula_version': LAGGED_FORMULA_VERSION,
                    'formula_code_sha256': FORMULA_CODE_SHA256, 'candidate_only': False,
                    'trade_rows': 100, 'sequence_rows': 100, 'zero_value_sequence_rows': 0,
                    'missing_trade_ids': 0, 'first_trade_id': 1, 'last_trade_id': 100,
                    'archive_url': f'https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{day}.zip',
                    'archive_sha256': 'a'*64, 'partition_basis': 'exchange_time_utc_day',
                    'available_at': (NOW-timedelta(minutes=30)).isoformat(),
                    'minute_count': 1440, MAIN: float(i), SMALL: float(i % 13),
                })
            self.write(self.root/'inputs/volume25'/DAY/(symbol+'.json'), {
                'symbol': symbol, 'day': DAY, 'complete': True, 'hours': 599,
                'window_end': (END-timedelta(hours=1)).isoformat(),
                'target_window_end': END.isoformat(), 'volume_lag_hours': 1,
                'quote_volume_mean': float(100-i), 'available_at': (NOW-timedelta(minutes=30)).isoformat(),
            })

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def path(self, symbol, day=DAY):
        return self.root/'inputs/daily'/day/(symbol+'.json')

    def candidate(self):
        return build(self.root, self.universe, DAY, NOW, profile='top50',
                     formula_version=LAGGED_FORMULA_VERSION,
                     strategy_version=LAGGED_STRATEGY_VERSION,
                     formula_code_sha256=FORMULA_CODE_SHA256)

    def assert_complete(self, candidate, excluded):
        self.assertEqual(candidate['status'], 'complete', candidate['blockers'])
        self.assertEqual(set(candidate['coverage']['excluded_symbols']), set(excluded))
        self.assertEqual(candidate['coverage']['observed'], 100-len(excluded))
        self.assertEqual(candidate['coverage']['expected'], 100-len(excluded))
        self.assertEqual(candidate['coverage']['requested'], 100)
        self.assertEqual(candidate['coverage']['missing'], {})
        self.assertEqual(set(candidate['factors']), set(self.symbols)-set(excluded))
        self.assertEqual(candidate['allocations'], select(candidate['factors'], .1)['allocations'])
        self.assertFalse(set(excluded)&set(candidate['allocations']))
        validate(candidate, NOW, actionable=False, expected_formula=LAGGED_FORMULA_VERSION,
                 expected_strategy=LAGGED_STRATEGY_VERSION)

    def test_bad_mature_symbol_does_not_block_good_pool(self):
        p = self.path('C99USDT'); row = json.loads(p.read_text())
        row.update(missing_trade_ids=20, last_trade_id=120)
        self.write(p, row)
        self.assert_complete(self.candidate(), {'C99USDT'})

    def test_missing_history_and_malformed_volume_are_isolated(self):
        self.path('C99USDT').unlink()
        (self.root/'inputs/volume25'/DAY/'C98USDT.json').write_text('{broken')
        self.path('C97USDT').write_text('[]')
        # Empty list is a missing input; it must still exclude only this coin.
        self.assert_complete(self.candidate(), {'C97USDT', 'C98USDT', 'C99USDT'})

    def test_wrong_formula_excluded_and_exact_tolerance_included(self):
        p = self.path('C99USDT'); row = json.loads(p.read_text())
        row['formula_version'] = 'old_formula'; self.write(p, row)
        p = self.path('C98USDT'); row = json.loads(p.read_text())
        row.update(trade_rows=90, sequence_rows=90, missing_trade_ids=10, last_trade_id=100)
        self.write(p, row)
        self.assert_complete(self.candidate(), {'C99USDT'})

    def test_current_marker_excludes_stale_live_row_but_official_can_restore(self):
        p = self.path('C99USDT'); row = json.loads(p.read_text()); p.unlink()
        self.write(self.root/'inputs/live_daily'/DAY/'C99USDT.json', row)
        self.write(self.root/'inputs/live_daily'/DAY/'complete.json', {
            'excluded_symbols': {'C99USDT': ['source_replay_failed']}})
        self.assert_complete(self.candidate(), {'C99USDT'})
        self.write(p, row)
        self.assert_complete(self.candidate(), set())

    def test_shared_btc_failure_still_blocks(self):
        p = self.root/'inputs/btc_hourly.json'; row = json.loads(p.read_text())
        row['closes'] = {}; self.write(p, row)
        self.assertIn('missing_btc_exact_hourly_closes', self.candidate()['blockers'])

    def test_insufficient_qualified_pool_still_blocks(self):
        for symbol in self.symbols[19:]: self.path(symbol).unlink()
        row = self.candidate()
        self.assertIn('eligible_universe_below_20', row['blockers'])
        self.assertEqual(row['coverage']['expected'], 19)


class SourcePoolTest(unittest.TestCase):
    def test_waiting_minute_excludes_bad_mature_symbol_but_keeps_other_components(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(prefix, 'ROOT', Path(temp)):
            row = {'minute': DAY+'T00:00:00+00:00', 'formula_version': LAGGED_FORMULA_VERSION,
                   'minute_code_sha256': prefix.MINUTE_CODE_SHA256, 'status': 'waiting',
                   'missing_thresholds': [{'symbol': 'BEATUSDT'}], 'invalid_records_in_lookback': 0,
                   'components': {'ADAUSDT': {'big_buy': 1}},
                   'threshold_source_sha256': {DAY+'/00/ADAUSDT': 'a'*64}}
            path = prefix.minute_file(DAY, 0); path.parent.mkdir(parents=True)
            path.write_text(json.dumps(row))
            actual, _, _ = prefix.read_minute(DAY, 0)
            self.assertIn('BEATUSDT', actual['excluded_symbols'])
            self.assertIn('ADAUSDT', actual['components'])
            row['invalid_records_in_lookback'] = 1; path.write_text(json.dumps(row))
            with self.assertRaises(ValueError): prefix.read_minute(DAY, 0)

    def test_boundary_publishes_qualified_pool_without_all_symbol_volume_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); root = base/'signal'; v3 = base/'v3'; universe = base/'universe.json'
            universe.write_text(json.dumps({'tradable_symbols': ['ADAUSDT', 'BEATUSDT']}))
            marker = v3/'inputs/live_daily'/DAY/'complete.json'; marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({'status': 'complete', 'formula_version': LAGGED_FORMULA_VERSION,
                                         'symbols': ['ADAUSDT'],
                                         'excluded_symbols': {'BEATUSDT': ['source_replay_failed']}}))
            near = END-timedelta(hours=2); btc = root/'inputs/btc_hourly.json'; btc.parent.mkdir(parents=True)
            btc.write_text(json.dumps({'fetched_at': (END-timedelta(minutes=5)).isoformat(),
                'closes': {near.isoformat(): 110, (near-timedelta(days=30)).isoformat(): 100},
                'first_candle_by_symbol': {'ADAUSDT': '2020-01-01', 'BEATUSDT': '2020-01-01'}}))
            class Clock(datetime):
                calls = 0
                @classmethod
                def now(cls, tz=None):
                    cls.calls += 1
                    return END+timedelta(minutes=cls.calls)
            candidate = {'status': 'complete', 'blockers': [], 'snapshot_id': 'a'*64,
                         'generated_at': (END+timedelta(minutes=1)).isoformat(),
                         'coverage': {'eligible': 1, 'excluded_symbols': {'BEATUSDT': ['bad_quality']}}}
            with patch.object(boundary, 'BASE', base), patch.object(boundary, 'ROOT', root), \
                    patch.object(boundary, 'V3', v3), patch.object(boundary, 'UNIVERSE', universe), \
                    patch.object(boundary, 'datetime', Clock), patch.object(boundary.time, 'sleep'), \
                    patch.object(boundary, 'wait_metadata', return_value=({}, None)), \
                    patch.object(boundary, 'readiness', return_value=({'ADAUSDT', 'BEATUSDT'}, {})), \
                    patch.object(boundary, 'build', return_value=candidate), \
                    patch.object(boundary, 'publish') as publish:
                boundary.main()
            publish.assert_called_once_with(root/'top50', candidate)

    def test_metadata_isolates_missing_symbol_hours(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); work = base/'work'/DAY; work.mkdir(parents=True)
            state = {'day': DAY, 'layout': 'hourly', 'invalid_target_records': 0,
                     'unsealed_paths': [], 'capture_status': 'running',
                     'capture_started_at': '2026-09-23T00:00:00+00:00',
                     'capture_checkpoint_at': '2026-10-03T00:00:01+00:00', 'processed': {},
                     'previous': {'ADAUSDT': {'id': 999, 'timestamp_ms': 1790899199999,
                                             'raw_segment_sha256': 'a'*64}}, 'following': {}}
            (work/'stage_state.json').write_text(json.dumps(state))
            capture = base/'data/individual_trades_v2'; capture.mkdir(parents=True)
            (capture/'state.json').write_text(json.dumps({'status': 'running', 'symbols': {}}))
            def hour(work, day, hour, symbol):
                return {'first_trade_id': 1000+hour, 'last_trade_id': 1000+hour,
                        'first_id_time_ms': hour, 'last_id_time_ms': hour,
                        'first_exchange_time_ms': hour, 'last_exchange_time_ms': hour,
                        'sequence_rows': 1, 'trade_rows': 1, 'zero_value_sequence_rows': 0,
                        'missing_trade_ids': 0, 'time_monotonic_by_id': True}
            with patch.object(metadata, 'BASE', base), patch.object(metadata, 'STAGED', base/'work'), \
                    patch.object(metadata, 'readonly_hour', side_effect=hour):
                result = metadata.metadata(DAY, {'ADAUSDT', 'BEATUSDT'}, exclude_invalid=True)
                self.assertEqual(set(result), {'ADAUSDT'})
                self.assertIn('BEATUSDT', result.excluded_symbols)
                with self.assertRaises(ValueError): metadata.metadata(DAY, {'ADAUSDT', 'BEATUSDT'})

    def test_daily_seal_isolates_replay_failure_and_lists_valid_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); work = root/'work'/DAY; work.mkdir(parents=True)
            (work/'stage_state.json').write_text(json.dumps({'processed': {}}))
            fast = {s: {'input_manifest_hash': digest({}), 'trade_rows': 1} for s in ('ADAUSDT', 'BEATUSDT')}
            prepared = {'last_minute': 1439, 'minute_hashes': {'2359.json': 'a'*64},
                        'trade_counts': {}, 'threshold_hashes': {}, 'symbols': {}}
            def replay(day, symbol, count):
                if symbol == 'BEATUSDT': raise ValueError('source changed during replay')
                return {MAIN: .1, SMALL: .2, 'minute_count': 1440, 'boundary_minute_count': 0}, {}
            with patch.object(daily, 'ROOT', root/'v3'), patch.object(daily, 'STAGED', root/'work'), \
                    patch.object(daily, 'load_prefix', return_value=prepared), \
                    patch.object(daily, 'replay_symbol', side_effect=replay):
                result = daily.emit(DAY, fast)
            marker = json.loads((root/'v3/inputs/live_daily'/DAY/'complete.json').read_text())
            self.assertEqual(marker['symbols'], ['ADAUSDT'])
            self.assertEqual(set(marker['files_sha256']), {'ADAUSDT'})
            self.assertIn('BEATUSDT', result['excluded_symbols'])


if __name__ == '__main__': unittest.main()
