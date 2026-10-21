# Optimizer 核心结构

优化对象为原始策略已经选出的多空币权重，不重新决定选币池。默认保持原始各腿名义预算；BTC 状态过滤造成的少配空头预算也随原策略保留，不强制变回市场中性。

目标为 `alpha @ w − gamma/2 * w.T @ Sigma @ w − 交易成本 − 预期资金费 − 稳定性惩罚`；各项具体尺度、约束、求解诊断和回退规则见 `solve.py`。默认单币权重范围为原始权重的 0.5–1.5 倍，不能翻转方向；研究变体可以进一步收缩至原始权重。默认数值配置见 `contracts.ResearchConfig`。

| 模块 | 用途 |
| --- | --- |
| `dataset.py` | 冻结数据、因子行索引对齐、标签成熟时间 |
| `alpha.py`, `forecasts.py` | 仅用已成熟收益标签估计预期 Alpha |
| `risk.py` | 基础协方差估计与收缩 |
| `paper_factors.py` | CMKT/CSMB/CMOM 与 CILLIQ10/CVOL30/CSKEW7；周度分组、日收益 |
| `factor_covariance.py` | 因子收益、暴露与残差风险组合为协方差 |
| `factor_risk_tuning.py` | 风险因子消融、半衰期、收缩目标与滚动选择 |
| `funding.py`, `costs.py` | 可用时间约束下的资金费预测、费用预测 |
| `solve.py`, `policy.py` | 约束优化、诊断和回退；基线与目标项消融 |
| `refinements.py` | 与等权混合、再平衡改进和滚动选择 |
| `replay.py` | 小时级真实持仓回放；费用、资金费、最小下单量与清算检查 |
| `validation.py` | 完整 UTC 日指标、Sortino、分块 Bootstrap、走步验证 |

`tools/run_optimizer_research.py` 是基础实验入口：原框架回放对账、B0–B5 消融、gamma 走步选择和成本/资金规模压力测试。它不会自动跑全部因子风险精调组合；精调参数集合由 `factor_risk_tuning.experiment_specs()` 给出，额外实验由对应研究模块组合调用。仓库保留核心实现和测试，省略绑定私有数据目录的日期版编排脚本与海量中间结果。

历史模型使用事后修订行情和因子，不等于在线时间版本。市场资本化使用保守发布延迟与失效规则，不能以当前市值回填历史。资金费事件保留结算时间及可用延迟；缺少归档时只剩旧 pivot 中的非零资金费事件，无法完整识别零费率结算，因此完整研究应提供资金费归档。

完整样本、成熟 Alpha 样本与参数选择后的样本应分开评价；不能将全区间挑选出来的最优参数当作独立样本外证据。历史实验保留原来的入口与口径。2026-10-21 起另用 `research/optimizer/live.py` 接入固定三因子 EWMA14 / 90 天日频实盘方案，详见 [接入说明](live_optimizer.md)。
