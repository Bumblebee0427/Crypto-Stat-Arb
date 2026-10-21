"""Audited v2 / 600-hour historical baseline; data paths are machine-local.

The live previous-hour v3 signal is a distinct factor definition. This profile
must not be interpreted as a v3 historical replay.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw_data_path = Path(os.environ.get('SELECT_COIN_MARKET_ROOT', ROOT/'data/market'))
spot_path = raw_data_path/'spot_dict.pkl'
swap_path = raw_data_path/'swap_dict.pkl'
backtest_path = ROOT/'data/回测结果'
backtest_iter_path = ROOT/'data/遍历结果'
backtest_name = 'historical_top50_v2_qv600'
start_date = os.environ.get('SELECT_COIN_START', '2021-01-01')
end_date = os.environ.get('SELECT_COIN_END', '2026-10-08')  # exclusive
account_type = '普通账户'
rebalance_mode = {'mode': 'RebAlways', 'params': {}}
initial_usdt = 10000
leverage = 2
margin_rate = 0.05
swap_c_rate = 0.0008
spot_c_rate = 0.0008
swap_min_order_limit = 5
spot_min_order_limit = 10
avg_price_col = 'avg_price_1m'
black_list = ['BTC-USDT', 'ETH-USDT']
white_list = []
min_kline_num = 0
stable_symbol = ['BKRW', 'USDC', 'USDP', 'TUSD', 'BUSD', 'FDUSD', 'DAI',
                 'EUR', 'GBP', 'USBP', 'SUSD', 'PAXG', 'AEUR', 'EURI']
job_num = int(os.environ.get('SELECT_COIN_JOBS', min(8, max((os.cpu_count() or 2)-1, 1))))
factor_col_limit = 64
strategy_list = json.loads((ROOT/'strategy.json').read_text())
for strategy in strategy_list:
    for name, values in strategy.items():
        if 'factor_list' in name or 'filter_list' in name:
            strategy[name] = [tuple(item) for item in values]
re_timing = None
data_source_dict = {
    'big_ultra_harmonic_rm240_mean_1d': (
        'load_partitioned_factor_parquet',
        os.environ.get('SELECT_COIN_MAIN_FACTOR_ROOT', str(ROOT/'data/factors/main'))),
    'small_qty_misalignment_rankcorr_lag5_1d': (
        'load_partitioned_factor_parquet',
        os.environ.get('SELECT_COIN_SMALL_FACTOR_ROOT', str(ROOT/'data/factors/small'))),
}
