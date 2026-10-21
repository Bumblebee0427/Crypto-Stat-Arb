# 本地数据与运行

所有时间使用 UTC。行情、因子和冻结结果由本地准备；仓库不携带数据下载凭证。只加载可信来源的 Python pickle 文件。

## 实盘计算

从 `live/` 运行模块。`SELECT_COIN_LIVE_ROOT` 指向包含 `data/` 的本地计算根目录；默认是包目录。所有模块只在该根目录读写计算结果。

输入包括：`data/individual_trades_v2/` 的已封逐笔分片和 `state.json`，`data/live_factor_storage/_state/universe.json` 的币池及上市信息，以及 BTC 小时收盘价和各币种小时成交额。逐笔字段为 `id, timestamp_ms, price, qty, maker`，需要逐笔 ID 去重、缺失范围及跨日边界证据。逐笔采集网络服务和生产调度不包含在此精简导出中。

主要计算顺序：

1. `deploy.seoul_v3_source_producer`：逐笔按小时整理、封存源证据。
2. `deploy.seoul_lagged_minute_shadow`：以前一完整小时阈值计算分钟分量。
3. `deploy.seoul_lagged_daily_prefix`：积累日内前缀；`deploy.seoul_lagged_daily_shadow` 封板并核验需要重放的币种。
4. `deploy.top50_boundary_publish`：封板、市场输入检查、合格池选币及原子发布。保留原代码中的截止时间、重试与磁盘保护逻辑，调用前准备好相应目录。

例如在已准备输入的计算副本上执行 `python -m deploy.seoul_lagged_minute_shadow`。各模块入口参数见文件底部的 `main()`；这些生产计算脚本不都提供 `--help`，不要把它当成通用试运行参数。

核心公式可直接从 `data_center.lagged_quantile_v3` 导入；不需要启动常驻服务。`lagged_factor_worker` 提供从官方公开逐笔归档回填 v3 的命令行入口，`factor_worker` 保留 v2 回填及发布校验依赖。回填时保留计算版本，勿把 v2 和 v3 日因子混存。

## 原始历史基线

从 `backtest/` 执行 `python backtest.py`。配置变量：

| 环境变量 | 含义 / 默认值 |
| --- | --- |
| `SELECT_COIN_MARKET_ROOT` | `data/market`；包含 `swap_dict.pkl`, `market_pivot_swap.pkl` 和 `market_pivot_spot.pkl` |
| `SELECT_COIN_MAIN_FACTOR_ROOT` | `data/factors/main`；原始主日因子分区 |
| `SELECT_COIN_SMALL_FACTOR_ROOT` | `data/factors/small`；原始小单日因子分区 |
| `SELECT_COIN_START`, `SELECT_COIN_END` | 默认 `2021-01-01`, `2026-10-08`（后者不含） |
| `SELECT_COIN_JOBS` | 并行进程数；默认至多 8 |
| `SELECT_COIN_BTC_MARKET_STATE_PATH` | 可选 BTC close cache；未设置则读行情 pivot |

`swap_dict.pkl` 为 `symbol -> 小时 DataFrame`。必需列包含 `candle_begin_time`, `symbol`, `open`, `close`, `avg_price_1m`, `quote_volume`, `volume`, `symbol_spot`, `symbol_swap`, `is_spot`, `is_trading`，其余预处理元数据按 `core/select_coin.py` 的访问准备。纯合约仍会读取 spot pivot 文件；可提供同一时间轴的合法空 spot pivot。

Pivot 是 `字段 -> 时间×币种 DataFrame`，包括 `open`, `close`, `vwap1m`, `funding_rate`；框架读取执行平均价及资金费。还需要 `backtest/data/min_qty/最小下单量_swap.csv`、`最小下单量_spot.csv`，GBK 编码，列为 `币种,最小下单量`。`update_min_qty.py` 可通过公开交易规则接口更新当前规则（需另外安装 `ccxt`）；重现实验必须保留当时冻结的规则，不能用当前规则冒充历史规则。

外部日因子目录布局为 `YYYY/YYYY_MM_DD/<factor_name>.parquet`，包含币种、`candle_begin_time` 和对应原始日因子列。准确列名与对齐方式见 `core/data_bridge.load_partitioned_factor_parquet`；主因子和小单因子分别使用 `data_source_dict` 的两个键。线性衰减、8/4 日均值在 `factors/` 中完成。结果和临时缓存写入 `backtest/data/`，被 Git 忽略。

## Optimizer 冻结数据适配器

`--manifest` 或 `SELECT_COIN_BACKTEST_MANIFEST` 指向本地 JSON，最小格式：

```json
{
  "factor_formula": "original same-hour quantiles v2",
  "volume_filter_hours": 600,
  "result_folder": "/path/to/frozen/run/data/回测结果/baseline",
  "market_source": "/path/to/frozen/market"
}
```

`result_folder` 包含 `config.pkl`, `选币结果.pkl`, `资金曲线.csv`。适配器从它向上三层得到 `run`，读取 `run/data/min_qty/最小下单量_swap.csv` 和 `run/data/cache/` 中的原始 K 线、主因子与 8/4 日小单因子 Series。Series 必须保留原始行索引。行情目录包含 `market_pivot_swap.pkl`。

`SELECT_COIN_FUNDING_ROOT` 指向本地 `fundingRate` 归档根目录，各币种子目录放 `*-fundingRate-*.zip`。归档 CSV 列为 `calc_time, last_funding_rate, funding_interval_hours`。默认位置是 `backtest/data/fundingRate`。

先执行 README 的 `--parity-only` 命令确认原框架回放一致，再去掉该参数开始研究。回放输出路径必须在冻结源目录之外。数据适配器会拒绝 v3 或非 600 小时设置；新增 v3 历史回测需要先单独构建和核验对应适配器。
