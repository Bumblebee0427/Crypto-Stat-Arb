import hashlib
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.utils.factor_hub import FactorHub
from core.utils.log_kit import logger
from core.utils.strategy_hub import DummyStrategy


def filter_series_by_range(series, range_str):
    # 提取运算符和数值
    operator = range_str[:2] if range_str[:2] in ['>=', '<=', '==', '!='] else range_str[0]
    value = float(range_str[len(operator):])

    match operator:
        case '>=':
            return series >= value
        case '<=':
            return series <= value
        case '==':
            return series == value
        case '!=':
            return series != value
        case '>':
            return series > value
        case '<':
            return series < value
        case _:
            raise ValueError(f"Unsupported operator: {operator}")


@dataclass(frozen=True)
class FactorConfig:
    name: str = 'Bias'  # 选币因子名称
    is_sort_asc: bool = True  # 是否正排序
    param: int = 3  # 选币因子参数
    weight: float = 1  # 选币因子权重
    is_cross: bool = False  # 是否截面因子

    @classmethod
    def parse_config_list(cls, config_list: List[tuple]):
        all_long_factor_weight = sum([factor[3] for factor in config_list])
        factor_list = []
        for factor_name, is_sort_asc, parameter_list, weight in config_list:
            new_weight = weight / all_long_factor_weight if all_long_factor_weight != 0 else 0
            factor = FactorHub.get_by_name(factor_name)
            factor_list.append(cls(name=factor_name, is_sort_asc=is_sort_asc, param=parameter_list, weight=new_weight,
                                   is_cross=factor.is_cross))
        return factor_list

    @cached_property
    def col_name(self):
        return f'{self.name}_{str(self.param)}'

    def __repr__(self):
        return f'{self.col_name}{"↑" if self.is_sort_asc else "↓"}权重:{self.weight}'

    def to_tuple(self):
        return self.name, self.is_sort_asc, self.param, self.weight


@dataclass(frozen=True)
class FilterMethod:
    how: str = ''  # 过滤方式
    range: str = ''  # 过滤值

    def __repr__(self):
        match self.how:
            case 'rank':
                name = '排名'
            case 'pct':
                name = '百分比'
            case 'val':
                name = '数值'
            case _:
                raise ValueError(f'不支持的过滤方式：`{self.how}`')

        return f'{name}:{self.range}'

    def to_val(self):
        return f'{self.how}:{self.range}'


@dataclass(frozen=True)
class FilterFactorConfig:
    name: str = 'Bias'  # 选币因子名称
    param: int = 3  # 选币因子参数
    method: FilterMethod = None  # 过滤方式
    is_sort_asc: bool = True  # 是否正排序
    is_cross: bool = False  # 是否截面因子

    def __repr__(self):
        _repr = self.col_name
        if self.method:
            _repr += f'{"↑" if self.is_sort_asc else "↓"}{self.method}'
        return _repr

    @cached_property
    def col_name(self):
        return f'{self.name}_{str(self.param)}'

    @classmethod
    def init(cls, filter_factor: tuple):
        # 仔细看，结合class的默认值，这个和默认策略中使用的过滤是一模一样的
        config = dict(name=filter_factor[0], param=filter_factor[1])
        if len(filter_factor) > 2:
            # 可以自定义过滤方式
            _how, _range = re.sub(r'\s+', '', filter_factor[2]).split(':')
            cls.check_value(_range)
            config['method'] = FilterMethod(how=_how, range=_range)
        if len(filter_factor) > 3:
            # 可以自定义排序
            config['is_sort_asc'] = filter_factor[3]
        factor = FactorHub.get_by_name(filter_factor[0])
        config['is_cross'] = factor.is_cross
        return cls(**config)

    def to_tuple(self, full_mode=False):
        if full_mode:
            return self.name, self.param, self.method.to_val(), self.is_sort_asc
        else:
            return self.name, self.param

    @staticmethod
    def check_value(range_str):
        _operator = range_str[:2] if range_str[:2] in ['>=', '<=', '==', '!='] else range_str[0]
        try:
            _ = float(range_str[len(_operator):])
        except ValueError as e:
            raise ValueError(f'过滤配置暂不支持表达式：`{range_str}`')


def calc_factor_common(df, factor_list: List[FactorConfig]):
    factor_val = np.zeros(df.shape[0], dtype=np.float64)
    for factor_config in factor_list:
        # 权重为 0，跳过计算
        if factor_config.weight == 0:
            continue
        col_name = f'{factor_config.name}_{str(factor_config.param)}'
        # 计算单个因子的排名
        _rank = df.groupby('candle_begin_time')[col_name].rank(ascending=factor_config.is_sort_asc, method='min')
        # 将因子按照权重累加
        factor_val += _rank * factor_config.weight
    return factor_val


def calc_factor_with_override(
    df: pd.DataFrame,
    factor_list: List[FactorConfig],
    asc_override: Dict[str, bool] | None = None,
    weight_override: Dict[str, float] | None = None,
):
    factor_val = np.zeros(df.shape[0], dtype=np.float64)
    for factor_config in factor_list:
        col_name = f'{factor_config.name}_{str(factor_config.param)}'
        is_sort_asc = asc_override.get(col_name, factor_config.is_sort_asc) if asc_override else factor_config.is_sort_asc
        weight = weight_override.get(col_name, factor_config.weight) if weight_override else factor_config.weight
        if weight == 0:
            continue
        _rank = df.groupby('candle_begin_time')[col_name].rank(ascending=is_sort_asc, method='min')
        factor_val += _rank * float(weight)
    return factor_val


def calc_factor_with_timevarying_override(
    df: pd.DataFrame,
    factor_list: List[FactorConfig],
    asc_override_by_time: pd.DataFrame | None = None,
    weight_override_by_time: pd.DataFrame | None = None,
):
    """
    使用“随时间变化”的方向和权重计算复合因子。
    asc/weight 使用 candle_begin_time 对齐，缺失时回退到因子原始配置。
    """
    factor_val = np.zeros(df.shape[0], dtype=np.float64)
    ts = df['candle_begin_time']

    for factor_config in factor_list:
        col_name = factor_config.col_name

        rank_low_better = df.groupby('candle_begin_time')[col_name].rank(ascending=True, method='min').to_numpy(
            dtype=np.float64
        )
        rank_high_better = df.groupby('candle_begin_time')[col_name].rank(ascending=False, method='min').to_numpy(
            dtype=np.float64
        )

        if asc_override_by_time is not None and col_name in asc_override_by_time.columns:
            asc_series = ts.map(asc_override_by_time[col_name])
            asc_arr = asc_series.fillna(factor_config.is_sort_asc).astype(bool).to_numpy()
        else:
            asc_arr = np.full(df.shape[0], factor_config.is_sort_asc, dtype=bool)

        if weight_override_by_time is not None and col_name in weight_override_by_time.columns:
            weight_series = pd.to_numeric(ts.map(weight_override_by_time[col_name]), errors='coerce')
            weight_arr = weight_series.fillna(float(factor_config.weight)).to_numpy(dtype=np.float64)
        else:
            weight_arr = np.full(df.shape[0], float(factor_config.weight), dtype=np.float64)

        rank_arr = np.where(asc_arr, rank_low_better, rank_high_better)
        factor_val += rank_arr * weight_arr

    return factor_val


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _infer_min_periods_from_window_days(window_days: float, hold_period: str) -> int:
    """
    将“天数窗口”转换为当前持仓周期下的最小样本点数。
    例：
      1H -> 60D => 1440
      4H -> 60D => 360
      1D -> 60D => 60
      2D -> 60D => 30
    """
    hp = str(hold_period).upper()
    m = re.match(r'^(\d+)([HD])$', hp)
    if not m:
        return max(int(round(window_days)), 1)

    period_num = max(int(m.group(1)), 1)
    period_type = m.group(2)
    if period_type == 'H':
        bars_per_day = 24.0 / period_num
    else:
        bars_per_day = 1.0 / period_num
    return max(int(round(float(window_days) * bars_per_day)), 1)


def estimate_fama_macbeth_override(
    df: pd.DataFrame,
    factor_list: List[FactorConfig],
    *,
    method_name: str,
    train_ratio: float,
    train_end: str,
    weight_by: str,
    min_cs_samples: int,
    ridge: float,
    use_sign: bool,
    min_abs_score: float,
    strategy_name: str,
    side_name: str,
) -> tuple[Dict[str, bool], Dict[str, float]]:
    """
    基于逐时点截面 OLS（Fama-MacBeth）估计因子方向与非等权权重（code1 风格）：
    1) 每个时点做截面 OLS（含截距）
    2) 对各时点 beta 取时间均值（weight_by='mean'）或时间 t 值（weight_by='tstat'）
    返回：
      - asc_override: col_name -> bool
      - weight_override: col_name -> normalized weight
    """
    factor_cols = [f.col_name for f in factor_list]
    required_cols = {'candle_begin_time', 'close', 'next_close', *factor_cols}
    if any(c not in df.columns for c in required_cols):
        logger.warning(f'[{strategy_name}] Fama-MacBeth({side_name})缺少必要列，回退等权。')
        return {}, {}

    work = df[['candle_begin_time', 'close', 'next_close', *factor_cols]].copy()
    work['ret_fwd'] = work['next_close'] / work['close'] - 1
    work.replace([np.inf, -np.inf], np.nan, inplace=True)
    work.dropna(subset=['ret_fwd', *factor_cols], inplace=True)
    if work.empty:
        logger.warning(f'[{strategy_name}] Fama-MacBeth({side_name})有效样本为空，回退等权。')
        return {}, {}

    all_times = np.sort(work['candle_begin_time'].unique())
    if len(all_times) == 0:
        logger.warning(f'[{strategy_name}] Fama-MacBeth({side_name})无可用时间截面，回退等权。')
        return {}, {}

    if train_end:
        cutoff = pd.to_datetime(train_end)
        train_times = all_times[all_times < cutoff]
        if len(train_times) == 0:
            logger.warning(f'[{strategy_name}] Fama-MacBeth({side_name}) train_end={train_end} 无样本，回退等权。')
            return {}, {}
    else:
        train_n = max(int(len(all_times) * train_ratio), 1)
        train_times = all_times[:train_n]

    train_set = set(train_times.tolist())
    beta_list = []
    intercept_list = []
    n_factor = len(factor_cols)
    min_cs_samples = max(int(min_cs_samples), 3)
    ridge = max(float(ridge), 0.0)

    for t, g in work.groupby('candle_begin_time', sort=True):
        if t not in train_set:
            continue
        if len(g) < min_cs_samples:
            continue

        x = g[factor_cols].to_numpy(dtype=np.float64)
        y = g['ret_fwd'].to_numpy(dtype=np.float64)

        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(y) < min_cs_samples:
            continue

        # code1 风格：截面 OLS 含截距项
        X = np.column_stack([np.ones(len(y), dtype=np.float64), x])
        xtx = X.T @ X
        xty = X.T @ y
        if ridge > 0:
            # 不惩罚截距，仅对 slope 做岭正则
            penalty = np.zeros((n_factor + 1, n_factor + 1), dtype=np.float64)
            penalty[1:, 1:] = ridge * np.eye(n_factor, dtype=np.float64)
        else:
            penalty = 0.0
        try:
            params = np.linalg.solve(xtx + penalty, xty)
        except np.linalg.LinAlgError:
            params, *_ = np.linalg.lstsq(xtx + penalty, xty, rcond=None)
        intercept_list.append(params[0])
        beta_list.append(params[1:])

    if len(beta_list) == 0:
        logger.warning(f'[{strategy_name}] Fama-MacBeth({side_name})有效截面不足，回退等权。')
        return {}, {}

    beta = np.vstack(beta_list)
    beta_mean = np.nanmean(beta, axis=0)
    beta_std = np.nanstd(beta, axis=0, ddof=1)
    beta_t = np.divide(
        beta_mean,
        beta_std / np.sqrt(len(beta)),
        out=np.zeros_like(beta_mean),
        where=beta_std > 1e-12,
    )

    if weight_by == 'mean':
        raw_score = beta_mean
    else:
        raw_score = beta_t

    raw_score = np.nan_to_num(raw_score, nan=0.0, posinf=0.0, neginf=0.0)

    asc_override: Dict[str, bool] = {}
    weight_override: Dict[str, float] = {}
    min_abs_score = max(float(min_abs_score), 0.0)
    for fc, score in zip(factor_list, raw_score):
        if abs(score) < min_abs_score:
            score = 0.0
        col = fc.col_name
        asc_override[col] = bool(score < 0) if use_sign else fc.is_sort_asc
        weight_override[col] = abs(float(score))

    weight_sum = sum(weight_override.values())
    if weight_sum <= 1e-12:
        logger.warning(f'[{strategy_name}] Fama-MacBeth({side_name})权重退化为0，回退等权。')
        return {}, {}

    for col in list(weight_override.keys()):
        weight_override[col] = weight_override[col] / weight_sum

    intercept_mean = float(np.nanmean(intercept_list)) if len(intercept_list) > 0 else 0.0
    coeff_desc = []
    for fc, m, t_val in zip(factor_list, beta_mean, beta_t):
        coeff_desc.append(
            f'{fc.col_name}:mean={m:+.3e},t={t_val:+.2f},'
            f'{"↑(low_better)" if asc_override[fc.col_name] else "↓(high_better)"},'
            f'w={weight_override[fc.col_name]:.3f}'
        )
    logger.info(
        f'[{strategy_name}] Fama-MacBeth({side_name}) '
        f'train_t={len(train_times)}, used_t={len(beta_list)}, intercept_mean={intercept_mean:+.3e} | '
        + ' | '.join(coeff_desc)
    )

    return asc_override, weight_override


def estimate_fama_macbeth_rolling_override(
    df: pd.DataFrame,
    factor_list: List[FactorConfig],
    *,
    weight_by: str,
    min_cs_samples: int,
    ridge: float,
    use_sign: bool,
    min_abs_score: float,
    rolling_window: int,
    rolling_window_days: float,
    rolling_min_periods: int,
    strategy_name: str,
    side_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fama-MacBeth rolling 版本：
    1) 每个时点做截面 OLS 得到 beta_t
    2) 对 beta_t 做 rolling(window) 聚合，得到动态方向与动态权重
       - weight_by='mean': 用 rolling mean(beta)
       - weight_by='tstat': rolling t-stat(beta)
    3) shift(1) 后用于当期打分，避免未来函数

    返回：
      - asc_override_by_time: index=time, columns=factor_col, bool/NaN
      - weight_override_by_time: index=time, columns=factor_col, float/NaN
    """
    factor_cols = [f.col_name for f in factor_list]
    required_cols = {'candle_begin_time', 'close', 'next_close', *factor_cols}
    if any(c not in df.columns for c in required_cols):
        logger.warning(f'[{strategy_name}] Fama-MacBeth-Rolling({side_name})缺少必要列，回退等权。')
        return pd.DataFrame(), pd.DataFrame()

    work = df[['candle_begin_time', 'close', 'next_close', *factor_cols]].copy()
    work['ret_fwd'] = work['next_close'] / work['close'] - 1
    work.replace([np.inf, -np.inf], np.nan, inplace=True)
    work.dropna(subset=['ret_fwd', *factor_cols], inplace=True)
    if work.empty:
        logger.warning(f'[{strategy_name}] Fama-MacBeth-Rolling({side_name})有效样本为空，回退等权。')
        return pd.DataFrame(), pd.DataFrame()

    min_cs_samples = max(int(min_cs_samples), 3)
    ridge = max(float(ridge), 0.0)
    rolling_window = max(int(rolling_window), 1)
    rolling_window_days = max(float(rolling_window_days or 0.0), 0.0)
    rolling_min_periods = max(1, int(rolling_min_periods))

    n_factor = len(factor_cols)
    beta_by_time = {}
    intercept_by_time = {}

    for t, g in work.groupby('candle_begin_time', sort=True):
        if len(g) < min_cs_samples:
            continue

        x = g[factor_cols].to_numpy(dtype=np.float64)
        y = g['ret_fwd'].to_numpy(dtype=np.float64)

        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(y) < min_cs_samples:
            continue

        # code1 风格：每时点截面 OLS，含截距
        X = np.column_stack([np.ones(len(y), dtype=np.float64), x])
        xtx = X.T @ X
        xty = X.T @ y
        if ridge > 0:
            # 只惩罚 slope，不惩罚截距
            penalty = np.zeros((n_factor + 1, n_factor + 1), dtype=np.float64)
            penalty[1:, 1:] = ridge * np.eye(n_factor, dtype=np.float64)
        else:
            penalty = 0.0
        try:
            params = np.linalg.solve(xtx + penalty, xty)
        except np.linalg.LinAlgError:
            params, *_ = np.linalg.lstsq(xtx + penalty, xty, rcond=None)

        t_key = pd.to_datetime(t)
        intercept_by_time[t_key] = float(params[0])
        beta_by_time[t_key] = params[1:]

    if len(beta_by_time) == 0:
        logger.warning(f'[{strategy_name}] Fama-MacBeth-Rolling({side_name})有效截面不足，回退等权。')
        return pd.DataFrame(), pd.DataFrame()

    beta_df = pd.DataFrame.from_dict(beta_by_time, orient='index', columns=factor_cols).sort_index()
    if rolling_window_days > 0:
        roll_obj = beta_df.rolling(window=f'{rolling_window_days}D', min_periods=rolling_min_periods)
    else:
        roll_obj = beta_df.rolling(window=rolling_window, min_periods=rolling_min_periods)
    roll_mean = roll_obj.mean()
    if weight_by == 'mean':
        raw_score_df = roll_mean
    else:
        roll_std = roll_obj.std(ddof=1)
        roll_n = roll_obj.count()
        raw_score_df = roll_mean / (roll_std / np.sqrt(roll_n))
        raw_score_df = raw_score_df.where((roll_std > 1e-12) & (roll_n > 1))

    raw_score_df = raw_score_df.shift(1)  # t时点打分只用到 <= t-1 的信息
    raw_score_df = raw_score_df.replace([np.inf, -np.inf], np.nan)

    min_abs_score = max(float(min_abs_score), 0.0)
    if min_abs_score > 0:
        raw_score_df = raw_score_df.where(raw_score_df.abs() >= min_abs_score, 0.0)

    abs_df = raw_score_df.abs()
    row_sum = abs_df.sum(axis=1)
    weight_df = abs_df.div(row_sum.replace(0, np.nan), axis=0)

    if use_sign:
        asc_df = raw_score_df.lt(0).where(raw_score_df.notna())
    else:
        asc_df = pd.DataFrame(
            {fc.col_name: fc.is_sort_asc for fc in factor_list},
            index=raw_score_df.index,
        )

    valid_rows = int(weight_df.notna().any(axis=1).sum())
    intercept_mean = float(np.nanmean(list(intercept_by_time.values()))) if len(intercept_by_time) > 0 else 0.0
    logger.info(
        f'[{strategy_name}] Fama-MacBeth-Rolling({side_name}) '
        f'beta_t={len(beta_df)}, valid_score_t={valid_rows}, '
        f'window_days={rolling_window_days}, window={rolling_window}, min_periods={rolling_min_periods}, '
        f'intercept_mean={intercept_mean:+.3e}, use_sign={use_sign}'
    )

    return asc_df, weight_df


def filter_common(df, filter_list):
    condition = pd.Series(True, index=df.index)

    for filter_config in filter_list:
        col_name = f'{filter_config.name}_{str(filter_config.param)}'
        # todo 这里可以优化，针对 method 进行聚合计算，减少grouby的次数
        match filter_config.method.how:
            case 'rank':
                rank = df.groupby('candle_begin_time')[col_name].rank(ascending=filter_config.is_sort_asc, pct=False)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case 'pct':
                rank = df.groupby('candle_begin_time')[col_name].rank(ascending=filter_config.is_sort_asc, pct=True)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case 'val':
                condition = condition & filter_series_by_range(df[col_name], filter_config.method.range)
            case _:
                raise ValueError(f'不支持的过滤方式：{filter_config.method.how}')

    return condition


@dataclass
class StrategyConfig:
    name: str = 'Strategy'
    strategy: str = 'Strategy'

    # 持仓周期。目前回测支持日线级别、小时级别。例：1H，6H，3D，7D......
    # 当持仓周期为D时，选币指标也是按照每天一根K线进行计算。
    # 当持仓周期为H时，选币指标也是按照每小时一根K线进行计算。
    hold_period: str = '1D'.replace('h', 'H').replace('d', 'D')

    # 配置offset
    offset: int = 0  # 策略配置的特定的offset
    offset_list: List[int] = (0,)

    # 是否使用现货
    is_use_spot: bool = False  # True：使用现货。False：不使用现货，只使用合约。

    # 选币市场范围 & 交易配置
    #   配置解释： 选币范围 + '_' + 优先交易币种类型
    #
    #   spot_spot: 在 '现货' 市场中进行选币。如果现货币种含有'合约'，优先交易 '现货'。
    #   swap_swap: 在 '合约' 市场中进行选币。如果现货币种含有'现货'，优先交易 '合约'。
    #   spot_swap: 在 '现货' 市场中进行选币。如果现货币种含有'合约',优先交易'合约'。
    #   mix_spot:  在 '现货与合约' 的市场中进行选币。如果两边市场都存在的币种，会保留'现货'，并优先下单'现货'。
    #   mix_swap:  在 '现货与合约' 的市场中进行选币。如果两边市场都存在的币种，会保留'合约'，并优先下单'合约'。
    market: str = 'swap_swap'

    # 多头选币数量。1 表示做多一个币; 0.1 表示做多10%的币
    long_select_coin_num: int | float | tuple = 0.1
    # 空头选币数量。1 表示做空一个币; 0.1 表示做空10%的币，'long_nums'表示和多头一样多的数量
    short_select_coin_num: int | float | tuple | str = 'long_nums'  # 注意：多头为0的时候，不能配置'long_nums'
    # 选币范围控制，
    # 默认为'both'，表示 <= 和 >=；'left' 表示 >= 和 <；'right' 表示 > 和 <=。
    # 也支持多空分离，比如 ['both', 'left']，表示多头是 both 模式，空头是 left 模式
    select_inclusive: str | tuple = 'right'

    # 多头的选币因子列名。
    long_factor: str = '因子'  # 因子：表示使用复合因子，默认是 factor_list 里面的因子组合。需要修改 calc_factor 函数配合使用
    # 空头的选币因子列名。多头和空头可以使用不同的选币因子
    short_factor: str = '因子'

    # 策略空头和多头的权重，默认是 1:1
    long_cap_weight: float = field(default=1)
    short_cap_weight: float = field(default=1)

    # 选币因子信息列表，用于`2_选币_单offset.py`，`3_计算多offset资金曲线.py`共用计算资金曲线
    factor_list: List[tuple] = ()  # 因子名（和factors文件中相同），排序方式，参数，权重。

    long_factor_list: List[FactorConfig] = ()  # 多头选币因子
    short_factor_list: List[FactorConfig] = ()  # 空头选币因子

    # 确认过滤因子及其参数，用于`2_选币_单offset.py`进行过滤
    filter_list: List[tuple] = ()  # 因子名（和factors文件中相同），参数

    long_filter_list: List[FilterFactorConfig] = ()  # 多头过滤因子
    short_filter_list: List[FilterFactorConfig] = ()  # 空头过滤因子

    # 后置过滤因子及其参数，用于`2_选币_单offset.py`进行过滤
    filter_list_post: List[tuple] = ()  # 因子名（和factors文件中相同），参数

    long_filter_list_post: List[FilterFactorConfig] = ()  # 多头后置过滤因子
    short_filter_list_post: List[FilterFactorConfig] = ()  # 空头后置过滤因子

    use_custom_func: bool = True  # 是否使用自定义函数

    # 因子合成方式：
    # - equal: 固定权重（默认）
    # - fama_macbeth: 截面OLS估计方向和权重（研究用）
    factor_weight_method: str = 'equal'
    # 简洁入口：
    # - rolling 窗口（样本点数）
    fama_macbeth_window: int = 24 * 7
    # - rolling 窗口（天数，>0 时优先于 fama_macbeth_window）
    fama_macbeth_window_days: float = 60.0
    # 高级参数入口：仅在 factor_weight_method='fama_macbeth' 时生效
    fama_macbeth_config: dict = field(default_factory=dict)
    # fama_macbeth_mode:
    # - rolling: 每时点截面 OLS -> 过去窗口内 beta 均值（推荐，默认）
    # - fixed: 每时点截面 OLS -> 固定训练区间 beta 时间均值
    fama_macbeth_mode: str = 'rolling'
    fama_macbeth_train_ratio: float = 0.7
    fama_macbeth_train_end: str = ''
    fama_macbeth_weight_by: str = 'mean'  # tstat / mean
    fama_macbeth_min_cs_samples: int = 30
    fama_macbeth_ridge: float = 1e-6
    fama_macbeth_use_sign: bool = True
    fama_macbeth_min_abs_score: float = 0.0
    fama_macbeth_rolling_window: int = 24 * 7
    fama_macbeth_rolling_min_periods: int = 24 * 7

    cap_weight: float = 1  # 策略权重

    md5_hash: str = ''  # 策略的md5值

    @cached_property
    def select_scope(self):
        return self.market.split('_')[0]

    @cached_property
    def order_first(self):
        return self.market.split('_')[1]

    @cached_property
    def is_day_period(self):
        return self.hold_period.endswith('D')

    @cached_property
    def is_hour_period(self):
        return self.hold_period.endswith('H')

    @cached_property
    def period_num(self) -> int:
        return int(self.hold_period.upper().replace('H', '').replace('D', ''))

    @cached_property
    def period_type(self) -> str:
        return self.hold_period[-1]

    @cached_property
    def factor_columns(self) -> List[str]:
        factor_columns = set()  # 去重

        # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
        for factor_config in set(self.long_factor_list + self.short_factor_list):
            # 策略因子最终在df中的列名
            factor_columns.add(factor_config.col_name)  # 添加到当前策略缓存信息中

        # 针对当前策略的过滤因子信息，整理之后的列名信息，并且缓存到全局
        for filter_factor in set(self.long_filter_list + self.short_filter_list):
            # 策略过滤因子最终在df中的列名
            factor_columns.add(filter_factor.col_name)  # 添加到当前策略缓存信息中

        # 针对当前策略的过滤因子信息，整理之后的列名信息，并且缓存到全局
        for filter_factor in set(self.long_filter_list_post + self.short_filter_list_post):
            # 策略过滤因子最终在df中的列名
            factor_columns.add(filter_factor.col_name)  # 添加到当前策略缓存信息中

        return list(factor_columns)

    @cached_property
    def all_factors(self) -> set:
        return (set(self.long_factor_list + self.short_factor_list) |
                set(self.long_filter_list + self.short_filter_list) |
                set(self.long_filter_list_post + self.short_filter_list_post))

    @classmethod
    def init(cls, index: int, file: DummyStrategy = None, **config):
        # 自动补充因子列表
        long_select_num = config.get('long_select_coin_num', file.long_select_coin_num if file else 0.1)
        short_select_num = config.get('short_select_coin_num', file.short_select_coin_num if file else 'long_nums')

        # 初始化多空分离策略因子
        factor_list = config.get('factor_list', file.factor_list if file else [])
        if 'long_factor_list' in config or 'short_factor_list' in config:
            # 如果设置过的话，默认单边是挂空挡
            factor_list = []
        config['long_factor_list'] = FactorConfig.parse_config_list(config.get('long_factor_list', factor_list))
        config['short_factor_list'] = FactorConfig.parse_config_list(config.get('short_factor_list', factor_list))

        # 因子加权方法
        factor_weight_method = str(config.get('factor_weight_method', 'equal')).lower()
        if factor_weight_method not in {'equal', 'fama_macbeth'}:
            logger.warning(f'未知 factor_weight_method={factor_weight_method}，自动回退为 equal')
            factor_weight_method = 'equal'
        config['factor_weight_method'] = factor_weight_method

        # ================== Fama-MacBeth 配置（简洁入口 + 高级配置） ==================
        fm_defaults = {
            'mode': 'rolling',
            'window': 24 * 7,
            'window_days': 60.0,
            'min_periods': None,          # None -> 跟随 window
            'train_ratio': 0.7,
            'train_end': '',
            'weight_by': 'mean',          # tstat / mean
            'min_cs_samples': 30,
            'ridge': 1e-6,
            'use_sign': True,
            'min_abs_score': 0.0,
        }
        fm_user = config.get('fama_macbeth_config', {})
        if fm_user is None:
            fm_user = {}
        if not isinstance(fm_user, dict):
            logger.warning('fama_macbeth_config 必须是 dict，已回退为默认配置')
            fm_user = {}

        fm_cfg = {**fm_defaults, **fm_user}

        # 简洁入口：策略层只改 window
        if 'fama_macbeth_window' in config:
            fm_cfg['window'] = config.get('fama_macbeth_window')
        if 'fama_macbeth_window_days' in config:
            fm_cfg['window_days'] = config.get('fama_macbeth_window_days')

        # 兼容旧字段写法（旧字段优先级更高，避免历史配置行为变化）
        legacy_to_cfg = {
            'fama_macbeth_mode': 'mode',
            'fama_macbeth_rolling_window': 'window',
            'fama_macbeth_window_days': 'window_days',
            'fama_macbeth_rolling_min_periods': 'min_periods',
            'fama_macbeth_train_ratio': 'train_ratio',
            'fama_macbeth_train_end': 'train_end',
            'fama_macbeth_weight_by': 'weight_by',
            'fama_macbeth_min_cs_samples': 'min_cs_samples',
            'fama_macbeth_ridge': 'ridge',
            'fama_macbeth_use_sign': 'use_sign',
            'fama_macbeth_min_abs_score': 'min_abs_score',
        }
        for legacy_key, cfg_key in legacy_to_cfg.items():
            if legacy_key in config:
                fm_cfg[cfg_key] = config.get(legacy_key)

        fm_mode = str(fm_cfg.get('mode', 'rolling')).lower()
        if fm_mode not in {'rolling', 'fixed'}:
            logger.warning(f'fama_macbeth_mode={fm_mode} 非法，自动回退为 rolling')
            fm_mode = 'rolling'

        fm_window = max(int(fm_cfg.get('window', 24 * 7)), 1)
        fm_window_days_raw = fm_cfg.get('window_days', 0.0)
        fm_window_days = max(float(fm_window_days_raw or 0.0), 0.0)
        fm_min_periods_raw = fm_cfg.get('min_periods', None)
        if fm_min_periods_raw is None:
            if fm_window_days > 0:
                fm_min_periods_raw = _infer_min_periods_from_window_days(
                    fm_window_days,
                    config.get('hold_period', file.hold_period if file else '1D'),
                )
            else:
                fm_min_periods_raw = fm_window
        fm_min_periods = max(int(fm_min_periods_raw), 1)
        if fm_window_days <= 0:
            fm_min_periods = min(fm_min_periods, fm_window)

        fm_train_ratio = float(fm_cfg.get('train_ratio', 0.7))
        if not (0 < fm_train_ratio <= 1):
            logger.warning(f'fama_macbeth_train_ratio={fm_train_ratio} 非法，自动回退为 0.7')
            fm_train_ratio = 0.7
        fm_train_end = str(fm_cfg.get('train_end', '') or '')

        fm_weight_by = str(fm_cfg.get('weight_by', 'mean')).lower()
        if fm_weight_by not in {'tstat', 'mean'}:
            logger.warning(f'fama_macbeth_weight_by={fm_weight_by} 非法，自动回退为 mean')
            fm_weight_by = 'mean'

        fm_min_cs_samples = max(int(fm_cfg.get('min_cs_samples', 30)), 3)
        fm_ridge = max(float(fm_cfg.get('ridge', 1e-6)), 0.0)
        fm_use_sign = _to_bool(fm_cfg.get('use_sign', True))
        fm_min_abs_score = max(float(fm_cfg.get('min_abs_score', 0.0)), 0.0)

        # 规范化后，写回结构化配置 + 兼容字段
        config['fama_macbeth_window'] = fm_window
        config['fama_macbeth_window_days'] = fm_window_days
        config['fama_macbeth_config'] = {
            'mode': fm_mode,
            'window': fm_window,
            'window_days': fm_window_days,
            'min_periods': fm_min_periods,
            'train_ratio': fm_train_ratio,
            'train_end': fm_train_end,
            'weight_by': fm_weight_by,
            'min_cs_samples': fm_min_cs_samples,
            'ridge': fm_ridge,
            'use_sign': fm_use_sign,
            'min_abs_score': fm_min_abs_score,
        }

        # 历史字段兼容（内部逻辑仍使用这些字段）
        config['fama_macbeth_mode'] = fm_mode
        config['fama_macbeth_rolling_window'] = fm_window
        config['fama_macbeth_rolling_min_periods'] = fm_min_periods
        config['fama_macbeth_window_days'] = fm_window_days
        config['fama_macbeth_train_ratio'] = fm_train_ratio
        config['fama_macbeth_train_end'] = fm_train_end
        config['fama_macbeth_weight_by'] = fm_weight_by
        config['fama_macbeth_min_cs_samples'] = fm_min_cs_samples
        config['fama_macbeth_ridge'] = fm_ridge
        config['fama_macbeth_use_sign'] = fm_use_sign
        config['fama_macbeth_min_abs_score'] = fm_min_abs_score

        for f in config['long_factor_list'] + config['short_factor_list']:
            if not f.is_cross:
                continue
            factor = FactorHub.get_by_name(f.name)
            _factor_list = [(cross_sec[0], True, cross_sec[1], 0) for cross_sec in factor.get_factor_list(f.param)]
            if _factor_list:
                config['long_factor_list'].extend(FactorConfig.parse_config_list(_factor_list))
                config['short_factor_list'].extend(FactorConfig.parse_config_list(_factor_list))

        # 初始化多空分离过滤因子
        filter_list = config.get('filter_list', file.filter_list if file else [])
        if 'long_filter_list' in config or 'short_filter_list' in config:
            # 如果设置过的话，则默认单边是挂空挡
            filter_list = []

        # 检查配置是否合法
        if file.is_abstract:
            use_custom_func = False
        else:
            use_custom_func = config.get('use_custom_func', True)
        config['use_custom_func'] = use_custom_func  # 重置config，否则后面实例化配置还是错的，导致因子文件无法生成

        long_filter_list = config.get('long_filter_list', filter_list)
        short_filter_list = config.get('short_filter_list', filter_list)
        new_filter_param = [len(item) > 2 for item in set(long_filter_list) | set(short_filter_list)]
        if any(new_filter_param) and use_custom_func:
            logger.error('过滤因子配置自定义规则，需要配置 use_custom_func 为 False')
            exit()
        old_filter_param = [len(item) <= 2 for item in set(long_filter_list) | set(short_filter_list)]
        if any(old_filter_param) and not use_custom_func:
            logger.error('策略中包含老的因子，但没有配置filter过滤规则，请检查config中策略的 filter_list 参数')
            exit()

        config['long_filter_list'] = [FilterFactorConfig.init(item) for item in long_filter_list]
        config['short_filter_list'] = [FilterFactorConfig.init(item) for item in short_filter_list]

        for f in config['long_filter_list'] + config['short_filter_list']:
            if not f.is_cross:
                continue
            factor = FactorHub.get_by_name(f.name)
            _factor_list = [(cross_sec[0], cross_sec[1], 'pct:<2') for cross_sec in factor.get_factor_list(f.param)]
            if _factor_list:
                config['long_filter_list'].extend([FilterFactorConfig.init(item) for item in _factor_list])
                config['short_filter_list'].extend([FilterFactorConfig.init(item) for item in _factor_list])

        # 初始化后置过滤因子
        filter_list_post = config.get('filter_list_post', [])
        if 'long_filter_list_post' in config or 'short_filter_list_post' in config:
            # 如果设置过的话，则默认单边是挂空挡
            filter_list_post = []
        config['long_filter_list_post'] = [FilterFactorConfig.init(item) for item in
                                           config.get('long_filter_list_post', filter_list_post)]
        config['short_filter_list_post'] = [FilterFactorConfig.init(item) for item in
                                            config.get('short_filter_list_post', filter_list_post)]

        for f in config['long_filter_list_post'] + config['short_filter_list_post']:
            if not f.is_cross:
                continue
            factor = FactorHub.get_by_name(f.name)
            _factor_list = [(cross_sec[0], cross_sec[1], 'pct:<2') for cross_sec in factor.get_factor_list(f.param)]
            if _factor_list:
                config['long_filter_list_post'].extend([FilterFactorConfig.init(item) for item in _factor_list])
                config['short_filter_list_post'].extend([FilterFactorConfig.init(item) for item in _factor_list])

        # 多空分离因子字段
        if config['long_factor_list'] != config['short_factor_list']:
            config['long_factor'] = '多头因子'
            config['short_factor'] = '空头因子'

        # 检查配置是否合法
        if (len(config['long_factor_list']) == 0) and (config.get('long_select_coin_num', 0) != 0):
            raise ValueError('多空分离因子配置有误，多头因子不能为空')
        if (len(config['short_factor_list']) == 0) and (config.get('short_select_coin_num', 0) != 0):
            raise ValueError('多空分离因子配置有误，空头因子不能为空')

        # auto overwrite
        # 如果多头不配置资金，然后空头还要和多头选一样多的情况下，直接把参数overwrite
        if config.get('long_cap_weight', 1) == 0 and short_select_num == 'long_nums':
            config['short_select_coin_num'] = long_select_num

        all_long_short_cap_weight = config.get('long_cap_weight', 1) + config.get('short_cap_weight', 1)
        if all_long_short_cap_weight == 0:
            raise ValueError('多空分离因子配置有误，空头和多头的总权重不能都为0')
        config['long_cap_weight'] = config.get('long_cap_weight', 1) / all_long_short_cap_weight
        config['short_cap_weight'] = config.get('short_cap_weight', 1) / all_long_short_cap_weight

        # 如果没有配置 market，则继续使用 is_use_spot
        if 'is_use_spot' in config and 'market' not in config:
            config['market'] = 'spot_swap' if config['is_use_spot'] else 'swap_swap'

        # 开始初始化策略对象
        if file is None or file.is_abstract:
            config['name'] = f"#{index}.{config.get('strategy', 'strategy')}"
            return cls(**config)

        """
        兼容历史原因下使用的策略配置逻辑
        """
        _config = dict(
            name=f'#{index}.{file.name}',
            hold_period=file.hold_period.upper(),
            long_select_coin_num=file.long_select_coin_num,
            short_select_coin_num=file.short_select_coin_num,
            long_factor=file.long_factor,
            short_factor=file.short_factor,
            factor_list=file.factor_list,
            filter_list=file.filter_list,
            cap_weight=1,
            md5_hash=file.md5_hash  # ** 回测特有 ** 记录文件的md5，避免重复计算
        )
        # 兼容历史原因，回测的相关策略特有逻辑
        if hasattr(file, 'if_use_spot'):
            _config['is_use_spot'] = file.if_use_spot
            _config['offset_list'] = [file.offset]
            _config['offset'] = file.offset

        _config.update(config)
        stg_conf = cls(**_config)

        # ** 整合原生函数的功能 **
        # 重新组合一下after_merge_index
        stg_conf.after_merge_index = file.after_merge_index

        # 重新组合一下filter_list
        stg_conf.factor_list = list(dict.fromkeys(
            [factor_config.to_tuple() for factor_config in stg_conf.long_factor_list + stg_conf.short_factor_list]))
        stg_conf.filter_list = list(dict.fromkeys(
            [filter_factor.to_tuple() for filter_factor in stg_conf.long_filter_list + stg_conf.short_filter_list]))

        # 如果使用自定义函数，重新赋值
        if stg_conf.use_custom_func and not file.is_abstract:
            stg_conf.calc_factor = file.calc_factor
            stg_conf.before_filter = file.before_filter
        return stg_conf

    def get_fullname(self, as_folder_name=False):
        factor_desc_list = [f'{self.long_factor_list}', f'前滤{self.long_filter_list}',
                            f'后滤{self.long_filter_list_post}']
        long_factor_desc = '&'.join(factor_desc_list)

        factor_desc_list = [f'{self.short_factor_list}', f'前滤{self.short_filter_list}',
                            f'后滤{self.short_filter_list_post}']
        short_factor_desc = '&'.join(factor_desc_list)

        # ** 回测特有 ** 因为需要计算hash，因此包含的信息不同
        fullname = f"""{self.name}-{self.hold_period}-{self.market}-offset{self.offset_list}"""
        if self.factor_weight_method != 'equal':
            fullname += f'-weight:{self.factor_weight_method}-{self.fama_macbeth_mode}-{self.fama_macbeth_weight_by}'
            if self.fama_macbeth_mode == 'rolling':
                if self.fama_macbeth_window_days > 0:
                    fullname += f'-rwd{self.fama_macbeth_window_days:g}'
                else:
                    fullname += f'-rw{self.fama_macbeth_rolling_window}'
                fullname += f'-mp{self.fama_macbeth_rolling_min_periods}'
            else:
                fullname += f'-tr{self.fama_macbeth_train_ratio:.2f}'
        if self.long_cap_weight > 0:
            fullname += f"""-多|权重:{self.long_cap_weight:.2f},数量:{self.long_select_coin_num},因子{long_factor_desc}"""
        if self.short_cap_weight > 0:
            fullname += f"""-空|权重:{self.short_cap_weight:.2f},数量:{self.short_select_coin_num},因子{short_factor_desc}"""

        md5_hash = hashlib.md5(f'{fullname}-{self.offset_list}-{self.md5_hash}'.encode('utf-8')).hexdigest()
        return f'{self.name}-{md5_hash[:8]}' if as_folder_name else fullname

    def __repr__(self):
        return f"""
................................
{self.name} 配置信息：
- 持仓周期: {self.hold_period}
- offset: ({len(self.offset_list)}个) {self.offset_list}
- 选币范围: {self.select_scope}
- 优先下单: {self.order_first}
- 多空比例: {self.long_cap_weight}:{self.short_cap_weight}
- 因子加权: {self.factor_weight_method}
- Fama-MacBeth参数: mode={self.fama_macbeth_mode}, train_ratio={self.fama_macbeth_train_ratio}, train_end={self.fama_macbeth_train_end or 'None'}, rolling_window={self.fama_macbeth_rolling_window}, rolling_window_days={self.fama_macbeth_window_days}, rolling_min_periods={self.fama_macbeth_rolling_min_periods}, weight_by={self.fama_macbeth_weight_by}, min_cs={self.fama_macbeth_min_cs_samples}, ridge={self.fama_macbeth_ridge}, use_sign={self.fama_macbeth_use_sign}, min_abs={self.fama_macbeth_min_abs_score}
- 多头选币设置(权重 {self.long_cap_weight * 100:.2f}%):
  * 选币数量: {self.long_select_coin_num}
  * 策略因子: {self.long_factor_list}
  * 前置过滤: {self.long_filter_list}
  * 后置过滤: {self.long_filter_list_post}
- 空头选币设置(权重 {self.short_cap_weight * 100:.2f}%):
  * 选币数量: {self.short_select_coin_num}
  * 策略因子: {self.short_factor_list}
  * 前置过滤: {self.short_filter_list}
  * 后置过滤: {self.short_filter_list_post}"""

    def calc_factor(self, df, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def calc_select_factor(self, df) -> pd.DataFrame:
        # 如果没有通过新的配置启动的话，使用原来的 `strategy` 中定义的函数计算
        if self.use_custom_func:
            # 1.2.1 新增：调用自定义函数的时候，把conf的对象传递给函数获取详细配置
            return self.calc_factor(df, external_list=self.factor_list, conf=self)

        # ========= 以上代码是为了兼容历史代码而写的 ========
        if self.factor_weight_method == 'equal':
            new_cols = {self.long_factor: calc_factor_common(df, self.long_factor_list)}
            if self.short_factor != self.long_factor:
                new_cols[self.short_factor] = calc_factor_common(df, self.short_factor_list)
            return pd.DataFrame(new_cols, index=df.index)

        if self.factor_weight_method == 'fama_macbeth':
            override_cache = {}

            def _calc_fm_score(factor_list: List[FactorConfig], side_name: str):
                key = tuple((f.col_name, f.is_sort_asc, f.weight) for f in factor_list)
                if key not in override_cache:
                    if self.fama_macbeth_mode == 'rolling':
                        override_cache[key] = estimate_fama_macbeth_rolling_override(
                            df,
                            factor_list,
                            weight_by=self.fama_macbeth_weight_by,
                            min_cs_samples=self.fama_macbeth_min_cs_samples,
                            ridge=self.fama_macbeth_ridge,
                            use_sign=self.fama_macbeth_use_sign,
                            min_abs_score=self.fama_macbeth_min_abs_score,
                            rolling_window=self.fama_macbeth_rolling_window,
                            rolling_window_days=self.fama_macbeth_window_days,
                            rolling_min_periods=self.fama_macbeth_rolling_min_periods,
                            strategy_name=self.name,
                            side_name=side_name,
                        )
                    else:
                        override_cache[key] = estimate_fama_macbeth_override(
                            df,
                            factor_list,
                            method_name=self.factor_weight_method,
                            train_ratio=self.fama_macbeth_train_ratio,
                            train_end=self.fama_macbeth_train_end,
                            weight_by=self.fama_macbeth_weight_by,
                            min_cs_samples=self.fama_macbeth_min_cs_samples,
                            ridge=self.fama_macbeth_ridge,
                            use_sign=self.fama_macbeth_use_sign,
                            min_abs_score=self.fama_macbeth_min_abs_score,
                            strategy_name=self.name,
                            side_name=side_name,
                        )

                if self.fama_macbeth_mode == 'rolling':
                    asc_by_time, weight_by_time = override_cache[key]
                    return calc_factor_with_timevarying_override(
                        df,
                        factor_list,
                        asc_override_by_time=asc_by_time,
                        weight_override_by_time=weight_by_time,
                    )

                asc_override, weight_override = override_cache[key]
                return calc_factor_with_override(df, factor_list, asc_override=asc_override, weight_override=weight_override)

            new_cols = {self.long_factor: _calc_fm_score(self.long_factor_list, 'long')}
            if self.short_factor != self.long_factor:
                new_cols[self.short_factor] = _calc_fm_score(self.short_factor_list, 'short')
            return pd.DataFrame(new_cols, index=df.index)

        raise ValueError(f'不支持的 factor_weight_method: {self.factor_weight_method}')

    def before_filter(self, df, **kwargs) -> (pd.DataFrame, pd.DataFrame):
        raise NotImplementedError

    def filter_before_select(self, df):
        if self.use_custom_func:
            # 1.2.1 新增：调用自定义函数的时候，把conf的对象传递给函数获取详细配置
            return self.before_filter(df, ex_filter_list=self.filter_list, conf=self)

        # ========= 以上代码是为了兼容历史代码而写的 ========
        # 过滤多空因子
        long_filter_condition = filter_common(df, self.long_filter_list)

        # 如果单独设置了空头过滤因子
        if self.long_filter_list != self.short_filter_list:
            short_filter_condition = filter_common(df, self.short_filter_list)
        else:
            short_filter_condition = long_filter_condition

        return df[long_filter_condition].copy(), df[short_filter_condition].copy()

    def filter_after_select(self, df):
        long_filter_condition = (df['方向'] == 1) & filter_common(df, self.long_filter_list_post)
        short_filter_condition = (df['方向'] == -1) & filter_common(df, self.short_filter_list_post)

        return df[long_filter_condition | short_filter_condition].copy()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def after_merge_index(self, candle_df, symbol, factor_dict, data_dict) -> Tuple[pd.DataFrame, dict, dict]:
        return candle_df, factor_dict, data_dict

    def select_by_coin_num(self, df, coin_num):
        select_range = coin_num if isinstance(coin_num, (tuple, list)) else (None, coin_num)
        select_inclusive = self.select_inclusive if isinstance(self.select_inclusive, (tuple, list)) else (
            self.select_inclusive, self.select_inclusive)

        def get_select_condition(side_select_num, inclusive, is_left):
            if side_select_num is not None:
                select_num = df['总币数'] * side_select_num if int(side_select_num) == 0 else side_select_num
                if is_left:
                    include_cond = (df['rank'] >= select_num) if inclusive != 'right' else (df['rank'] > select_num)
                else:
                    include_cond = (df['rank'] <= select_num) if inclusive != 'left' else (df['rank'] < select_num)
            else:
                include_cond = pd.Series([True] * len(df), index=df.index)
            return include_cond

        # Calculate conditions for left and right sides
        left_condition = get_select_condition(select_range[0], select_inclusive[0], is_left=True)
        right_condition = get_select_condition(select_range[1], select_inclusive[1], is_left=False)

        return df[left_condition & right_condition].copy(False)
