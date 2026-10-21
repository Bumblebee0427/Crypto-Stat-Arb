# 三因子 EWMA14 / 90 天实盘接入

固定模型名为 `core_w90_f14_r14_cv_diagonal`。计算器使用与研究相同的 `estimate_grid`、`RiskSpec` 和 `solve_weights`；当前调用链与 B0–B5 的历史通用入口独立。

| 设置 | 实盘值 |
| --- | --- |
| 因子 | CMKT、CSMB、CMOM |
| 风险窗口 | 最近 90 个日历日 |
| 因子 / 残差 EWMA 半衰期 | 均为 14 天 |
| 收缩 | 向对角矩阵收缩；前 80% 拟合、后 20% QLIKE 验证 |
| 收缩强度候选 | 0、0.1、0.25、0.5、0.75、1；因子和残差分别选择 |
| 原始优化解单币边界 | 原始权重的 0.5–1.5 倍，禁止翻方向 |
| 实际应用 | 75% 原始权重 + 25% 优化解，故相对原权重为 0.875–1.125 倍 |
| 方向与预算 | 选币池、多头预算、空头预算均保持原策略设置 |
| 目标项 | Alpha − gamma/2 × 方差 − 成本 − 预期资金费 − 稳定性惩罚；gamma=2，eta=1e-5，单边成本 8 bp |
| 资金基准 | 使用策略现金预算作为优化权益基准，保持研究中的 2 倍名义权重尺度 |
| 频率 | 每日北京时间 08:00 换仓；不运行研究中的每小时 10% 再平衡阈值 |

当前名义目标为 `2 × min(1500, 0.5 × 账户权益) × gross_allocation`。正常多空时 `gross_allocation=1`；原策略的 BTC 过滤减少空头预算时不重新放大。步长取整、低于最小名义金额的同向调整保留，以及实际持仓漂移仍按原执行规则处理，实际持仓金额可能偏离目标，开仓继续检查交易所限额与可用保证金。

## 数据更新

`tools/update_live_optimizer_data.py` 读取币安官网的公开接口：

- `https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/complianceSymbolList`：当前市值信息及 CMC 唯一 ID。
- `https://www.binance.com/bapi/composite/v1/public/promo/cmc/cryptocurrency/quotes/historical`：CoinMarketCap 来源的历史 USD 市值、价格和流通量。
- `https://fapi.binance.com/fapi/v1/klines`：官方已结束日线的价格与成交额。
- `https://fapi.binance.com/fapi/v1/fundingRate`：已结算资金费。

使用历史报价，不使用“当前流通量 × 历史价格”回填。UTC 零点 D 的报价记录为 D−1 的收盘观测，继续按研究的 source day +2 天规则处理可用时间，并限制市值过期填充。它是修订后的历史序列，不能宣称拥有真实历史发布版本。

原始响应、哈希与观测时间留在本地 `data/`，新增数据不覆盖原冻结回测。市值缺失的因子组成币不会被伪造市值；因子池继续使用已知且有效的组成币。实盘选出的新币仍可利用其价格历史估计暴露；样本不足时固定在原权重。最新因子必须完整且覆盖最近决策日，90 天窗口至少有 60 条完整观测。

初始化示例（本地目录自行准备）：

```bash
cd backtest
../.venv/bin/python tools/update_live_optimizer_data.py \
  --root data/live_optimizer \
  --legacy-cap-root /path/to/legacy/coin-cap \
  --seed-closes /path/to/frozen/daily_closes.parquet \
  --seed-quotes /path/to/frozen/daily_quote_volume.parquet
```

## 每日生产与交易机校验

`tools/run_live_optimizer.py --config /path/to/private/live_config.json` 在 UTC 00:00 开始更新数据，等待首尔完整 v3 选币快照，读取交易机只读持仓与权益、计算资金费预测，再生成当天权重文件。重用已有行情和市值时下载近几天的重叠区间。

私有配置包含 `data_root, legacy_cap_root, seed_closes, seed_quotes, ssh_key, trade_host, trade_root, seoul_host, seoul_root, env_file`。地址和密钥路径由部署环境配置；交易所密钥只在交易机原有环境文件内读取，不传回计算机。`--preview --skip-refresh` 可执行只读预演，预演文件带有不可交易标记。

交易机每日执行器在生成订单前调用 `optimizer_overlay.load_and_apply`。它检查完整哈希、原始快照 ID、当日边界、生成时间、数据截止时间、当前持仓哈希、预算变化、币池、方向、权重边界与两边预算。生成的派生信号保留 `base_snapshot_id`，账本记录模型、优化文件 ID 和回退状态。

启动配置 `optimizer/enabled.json` 指明固定模型。收到完整原始信号后最多等待至北京时间 08:08；等待文件时不反复读取账户。届时没有有效优化文件，就继续使用原策略权重。原始信号自身不合格时仍按原有执行规则处理。可通过关闭该启动配置回到原始分配，资金预算独立保留。

生产定时器每天 UTC 00:00:05 / 北京时间 08:00:05 运行；交易机原每日执行定时器负责下单。算法计算不直接调用订单接口，候选文件通过 SSH 原子投递，文件归属于交易服务用户且权限为 0600。

## Alpha 冷启动与样本区别

不把 v2 历史标签当作 v3 标签。线上 v3 尚未积累足够训练日时，预期 Alpha 为零；风险、交易成本和资金费仍参与优化。资金费缺失或过期、且不适合主动调权的币保留原权重。

每日保存 v3 选币特征；标签的起点使用信号发布后下一完整分钟的 VWAP，终点为次日对应基准价。标签保守地在次日 UTC 01:00 后才可用于月度模型训练。未取到价格的标签保持缺失，不以零收益填补。

这次实盘保留每日换仓节奏；历史研究的每小时再平衡、在线 v3 数据可用时间和策略预算与历史 v2 实验存在区别，历史收益不能直接作为此次实盘收益预期。
