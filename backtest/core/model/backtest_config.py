import shutil
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Dict, Optional, Set

import pandas as pd

from config import backtest_path, backtest_iter_path, backtest_name
from core.model.account_type import AccountType
from core.model.rebalance_mode import RebalanceMode
from core.model.strategy_config import StrategyConfig, FactorConfig
from core.model.timing_signal import TimingSignal
from core.utils.factor_hub import FactorHub
from core.utils.log_kit import logger as default_logger
from core.utils.path_kit import get_folder_path
from core.utils.strategy_hub import StrategyHub


class BacktestConfig:
    data_file_fingerprint: str = ''  # 记录数据文件的指纹

    def __init__(self, name: str, **conf):
        self.name: str = name  # 账户名称，建议用英文，不要带有特殊符号
        self.logger = conf.get("logger", default_logger)

        self.start_date: str = conf.get("start_date", '2021-01-01')  # 回测开始时间
        self.end_date: str = conf.get("end_date", '2024-03-30')  # 回测结束时间

        # 账户回测交易模拟配置
        self.account_type: AccountType = AccountType.translate(conf.get("account_type", '普通账户'))  # 账户类型
        self.rebalance_mode: RebalanceMode = RebalanceMode.init(conf.get('rebalance_mode', None))
        self.avg_price_col: str = conf.get("avg_price_col", 'avg_price_1m')  # 平均成交价格
        self.initial_usdt: int | float = conf.get("initial_usdt", 10000)  # 初始现金
        self.leverage: int | float = conf.get("leverage", 1)  # 杠杆数。我看哪个赌狗要把这里改成大于1的。高杠杆如梦幻泡影。不要想着一夜暴富，脚踏实地赚自己该赚的钱。
        self.margin_rate = conf.get('margin_rate', 0.05)  # 维持保证金率，净值低于这个比例会爆仓

        self.swap_c_rate: float = conf.get("swap_c_rate", 6e-4)  # 合约买卖手续费
        self.spot_c_rate: float = conf.get("spot_c_rate", 2e-3)  # 现货买卖手续费

        self.swap_min_order_limit: int | float = conf.get("swap_min_order_limit", 5)  # 合约最小下单量
        self.spot_min_order_limit: int | float = conf.get("spot_min_order_limit", 10)  # 现货最小下单量

        # 策略配置
        self.black_list: List[str] = conf.get('black_list',
                                              [])  # 拉黑名单，永远不会交易。不喜欢的币、异常的币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
        self.white_list: List[str] = conf.get('white_list',
                                              [])  # 如果不为空，即只交易这些币，只在这些币当中进行选币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
        self.min_kline_num: int = conf.get('min_kline_num', 168)  # 最少上市多久，不满该K线根数的币剔除，即剔除刚刚上市的新币。168：标识168个小时，即：7*24

        # 再择时配置
        self.timing: Optional[TimingSignal] = None

        self.select_scope_set: Set[str] = set()
        self.order_first_set: Set[str] = set()
        self.is_day_period: bool = False  # 是否是日盘，否则是小时盘
        self.is_hour_period: bool = False  # 是否是小时盘，否则是日盘
        self.factor_params_dict: Dict[str, set] = {}
        self.factor_col_name_list: List[str] = []
        self.max_hold_period: str = '1H'  # 最大的持仓周期，默认值设置为最小
        self.hold_period_list: List[str] = []  # 持仓周期列表
        self.max_offset_len: int = 0

        # 策略列表，包含每个策略的详细配置
        self.strategy_list: List[StrategyConfig] = []
        self.strategy_name_list: List[str] = []
        self.strategy_list_raw: List[dict] = []

        # 策略评价
        self.report: Optional[pd.DataFrame] = None

        # 遍历标记
        self.iter_round: int | str = 0  # 遍历的INDEX，0表示非遍历场景，从1、2、3、4、...开始表示是第几个循环，当然也可以赋值为具体名称

    def __repr__(self):
        return f"""{'+' * 56}
# {self.name} 配置信息如下：
+ 回测时间: {self.start_date} ~ {self.end_date}
+ 手续费: 合约{self.swap_c_rate * 100:.2f}%，现货{self.spot_c_rate * 100:.2f}%
+ 杠杆: {self.leverage:.2f}
+ 最小K线数量: {self.min_kline_num}
+ 维持保证金率: {self.margin_rate * 100:.2f}%
+ 拉黑名单: {self.black_list}，只交易名单: {self.white_list}
+ Rebalance 模式: {self.rebalance_mode}
+ 再择时: {self.timing}
{''.join([str(item) for item in self.strategy_list])}
{'+' * 56}
"""

    @property
    def section_params_dict(self):
        section_params_dict = {}
        for factor_name, param_list in self.factor_params_dict.items():
            if FactorHub.get_by_name(factor_name).is_cross:
                section_params_dict[factor_name] = param_list
        return section_params_dict

    @property
    def section_depend_factor_col_name_list(self):
        depend_factor_param_list = []
        for factor_name, param_list in self.factor_params_dict.items():
            factor = FactorHub.get_by_name(factor_name)
            if factor.is_cross:
                for param in param_list:
                    depend_factor_param_list.extend(factor.get_factor_list(param))
        return list(set(f'{f}_{n}' for f, n in set(depend_factor_param_list)))

    @property
    def has_section_factor(self):
        return any(FactorHub.get_by_name(factor_name).is_cross for factor_name in self.factor_params_dict.keys())

    @property
    def hold_period_type(self):
        return 'D' if self.is_day_period else 'H'

    def info(self):
        # 输出一下配置信息
        self.logger.debug(self)

    def get_fullname(self, as_folder_name=False):
        fullname_list = [self.name]
        for stg in self.strategy_list:
            fullname_list.append(f"{stg.get_fullname(as_folder_name)}")

        if self.timing:
            fullname_list.append(f'再择时:{self.timing}')

        fullname = ' '.join(fullname_list)
        return f'{self.name}' if as_folder_name else fullname

    def load_strategy_config(self, strategy_list: list | tuple, re_timing_config=None):
        self.strategy_list_raw = strategy_list
        # 所有策略中的权重
        all_cap_weight = sum(item["cap_weight"] for item in strategy_list)

        for index, stg_dict in enumerate(strategy_list):
            # 更新策略权重
            strategy_name = stg_dict['strategy']

            stg_cfg = StrategyConfig.init(index, file=StrategyHub.get_by_name(strategy_name), **stg_dict)

            offset_list = list(filter(lambda x: x < stg_cfg.period_num, stg_cfg.offset_list))
            if len(offset_list) != len(stg_cfg.offset_list):
                self.logger.warning(
                    f'策略{stg_cfg.name}的offset_list设置有问题，自动裁剪。原始值：{stg_cfg.offset_list},裁剪后：{offset_list}')
            stg_cfg.offset_list = offset_list
            stg_cfg.cap_weight = stg_cfg.cap_weight / all_cap_weight

            if stg_cfg.is_day_period:
                self.is_day_period = True
            else:
                self.is_hour_period = True

            # 缓存持仓周期的事情
            if stg_cfg.hold_period not in self.hold_period_list:
                self.hold_period_list.append(stg_cfg.hold_period)
                # 更新最大的持仓周期
                if pd.to_timedelta(self.max_hold_period) < pd.to_timedelta(stg_cfg.hold_period):
                    self.max_hold_period = stg_cfg.hold_period

            self.select_scope_set.add(stg_cfg.select_scope)
            self.order_first_set.add(stg_cfg.order_first)
            if not {'spot', 'mix'}.isdisjoint(self.select_scope_set) and self.leverage >= 2:
                self.logger.error(f'现货策略不支持杠杆大于等于2的情况，请重新配置')
                exit(1)

            if stg_cfg.long_select_coin_num == 0 and (stg_cfg.short_select_coin_num == 0 or
                                                      stg_cfg.short_select_coin_num == 'long_nums'):
                self.logger.warning('策略中的选股数量都为0，忽略此策略配置')
                continue

            self.strategy_list.append(stg_cfg)
            self.strategy_name_list.append(stg_cfg.name)
            self.factor_col_name_list += stg_cfg.factor_columns

            # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
            for factor_config in stg_cfg.all_factors:
                # 添加到并行计算的缓存中
                if factor_config.name not in self.factor_params_dict:
                    self.factor_params_dict[factor_config.name] = set()
                self.factor_params_dict[factor_config.name].add(factor_config.param)

            if len(stg_cfg.offset_list) > self.max_offset_len:
                self.max_offset_len = len(stg_cfg.offset_list)

        self.factor_col_name_list = list(set(self.factor_col_name_list))

        if all((self.is_hour_period, self.is_day_period)):
            self.logger.critical(f'策略中同时存在小时线和日线的策略融合，请检查配置')
            exit()

        if re_timing_config:
            self.timing = TimingSignal(**re_timing_config)

    @classmethod
    def init_from_config(cls, load_strategy_list: bool = True) -> "BacktestConfig":
        import config

        backtest_config = cls(
            config.backtest_name,
            account_type=config.account_type,  # 账户类型
            rebalance_mode=getattr(config, 'rebalance_mode', None),  # rebalance类型
            start_date=config.start_date,  # 回测开始时间
            end_date=config.end_date,  # 回测结束时间
            # ** 交易配置 **
            initial_usdt=config.initial_usdt,  # 初始usdt
            leverage=config.leverage,  # 杠杆
            margin_rate=config.margin_rate,  # 维持保证金率
            swap_c_rate=config.swap_c_rate,  # 合约买入手续费
            spot_c_rate=config.spot_c_rate,  # 现货买卖手续费
            spot_min_order_limit=config.spot_min_order_limit,  # 现货最小下单量
            swap_min_order_limit=config.swap_min_order_limit,  # 合约最小下单量
            # ** 数据参数 **
            avg_price_col=config.avg_price_col,  # 平均价格列名
            black_list=config.black_list,  # 拉黑名单
            white_list=config.white_list,  # 只交易名单
            min_kline_num=config.min_kline_num,  # 最小K线数量，k线数量少于这个数字的部分不会计入计算
        )

        # ** 策略配置 **
        # 初始化策略，默认都是需要初始化的
        if load_strategy_list:
            re_timing_config = getattr(config, 're_timing', None)  # 从config中读取选币再择时的策略配置
            backtest_config.load_strategy_config(config.strategy_list, re_timing_config)

        return backtest_config

    def set_report(self, report: pd.DataFrame):
        report['param'] = self.get_fullname()
        self.report = report

    def get_result_folder(self) -> Path:
        if self.iter_round == 0:
            return get_folder_path(backtest_path, self.get_fullname(as_folder_name=True), as_path_type=True)
        else:
            return get_folder_path(
                backtest_iter_path,
                self.name,
                f'参数组合_{self.iter_round}' if isinstance(self.iter_round, int) else self.iter_round,
                as_path_type=True
            )

    def get_strategy_config_sheet(self, with_factors=True, sep_filter=False) -> dict:
        factor_dict = {}
        for stg in self.strategy_list:
            for attr_in in ['hold_period', 'is_use_spot', 'offset_list', 'cap_weight']:
                if attr_in not in factor_dict:
                    factor_dict[attr_in] = []
                factor_dict[attr_in].append(getattr(stg, attr_in))

            for factor_config in stg.all_factors:
                if sep_filter:
                    factor_type = 'FACTOR' if isinstance(factor_config, FactorConfig) else 'FILTER'
                    _name = f'#{factor_type}-{factor_config.name}'
                else:
                    _name = f'#FACTOR-{factor_config.name}'
                _val = factor_config.param
                if _name not in factor_dict:
                    factor_dict[_name] = []
                factor_dict[_name].append(_val)
        ret = {
            '策略': self.name,
            'fullname': self.get_fullname(),
        }
        if with_factors:
            ret.update(**{
                k: "_".join(map(str, v)) for k, v in factor_dict.items()
            })

        if self.timing:
            ret['再择时'] = str(self.timing)
        return ret

    def save(self):
        pd.to_pickle(self, self.get_result_folder() / 'config.pkl')

    def delete_cache(self):
        shutil.rmtree(self.get_result_folder())


class BacktestConfigFactory:
    """
    遍历参数的时候，动态生成配置
    """
    STRATEGY_FACTOR_ATTR = [
        'factor_list',
        'long_factor_list',
        'short_factor_list',
        'filter_list',
        'long_filter_list',
        'short_filter_list',
        'filter_list_post',
        'long_filter_list_post',
        'short_filter_list_post',
    ]

    def __init__(self, **conf):
        # ====================================================================================================
        # ** 参数遍历配置 **
        # 可以指定因子遍历的参数范围
        # ====================================================================================================
        self.factor_param_range_dict: dict = conf.get("factor_param_range_dict", {})
        self.strategy_param_range_dict: dict = conf.get("strategy_param_range_dict", {})
        self.default_param_range = conf.get("default_param_range", [])
        self.backtest_name = conf.get("backtest_name", backtest_name)

        if not self.backtest_name:
            self.backtest_name = f'默认策略-{datetime.now().strftime("%Y%m%dT%H%M%S")}'

        # 缓存全局配置
        self.select_scope_set: Set[str] = set()
        self.black_list = conf.get("black_list", set())

        # 存储生成好的config list和strategy list
        self.config_list: List[BacktestConfig] = []
        self.strategy_list: List[StrategyConfig] = []

        default_logger.info(f'遍历结果输出路径：{self.result_folder}')

    @property
    def result_folder(self) -> Path:
        return get_folder_path(backtest_iter_path, self.backtest_name, as_path_type=True)

    def update_meta_by_config(self, config: BacktestConfig):
        """
        # 缓存是否使用现货等状态
        :param config: 生成的配置信息
        :return: None
        """
        self.select_scope_set = self.select_scope_set | config.select_scope_set
        self.black_list = self.black_list | set(config.black_list)

    def generate_long_and_short_configs(self) -> List[BacktestConfig]:
        """
        纯多/纯空的配置，用于多空曲线的计算
        :return:
        """
        import config

        long_short_strategy_list = []
        pure_long_strategy_list = []
        pure_short_strategy_list = []
        for strategy_dict in config.strategy_list:
            strategy_cfg = strategy_dict.copy()
            long_strategy_cfg = {**strategy_dict, **{'long_cap_weight': 1, 'short_cap_weight': 0}}
            short_strategy_cfg = {**strategy_dict, **{'long_cap_weight': 0, 'short_cap_weight': 1}}

            long_short_strategy_list.append(strategy_cfg)
            pure_long_strategy_list.append(long_strategy_cfg)
            pure_short_strategy_list.append(short_strategy_cfg)

        config_list: List[BacktestConfig] = []
        for stg, suffix in zip([long_short_strategy_list, pure_long_strategy_list, pure_short_strategy_list],
                               ['多空模拟', '纯多模拟', '纯空模拟']):
            backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
            backtest_config.load_strategy_config(stg)
            if len(backtest_config.strategy_list) == 0:
                default_logger.critical(f'【{suffix}场景】没有生成有效的子策略回测回测，可能所有选币都被重置为0，跳过')
                continue
            backtest_config.name = self.backtest_name
            backtest_config.iter_round = suffix

            self.update_meta_by_config(backtest_config)

            config_list.append(backtest_config)

        self.config_list = config_list

        return self.config_list

    def generate_all_factor_config(self):
        backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
        strategy_list = []
        for conf in self.config_list:
            strategy_list.extend(conf.strategy_list_raw)
        backtest_config.load_strategy_config(strategy_list)
        return backtest_config

    def get_name_params_sheet(self) -> pd.DataFrame:
        rows = []
        for config in self.config_list:
            rows.append(config.get_strategy_config_sheet())

        sheet = pd.DataFrame(rows)
        sheet.to_excel(self.config_list[-1].get_result_folder().parent / '策略回测参数总表.xlsx', index=False)
        return sheet

    def generate_configs_by_strategies(self, strategies, re_timing_strategies=None) -> List[BacktestConfig]:
        config_list = []
        iter_round = 0

        if not re_timing_strategies:
            re_timing_strategies = [None]

        for strategy_list, re_timing_config in product(strategies, re_timing_strategies):
            iter_round += 1
            backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
            if self.backtest_name:
                backtest_config.name = self.backtest_name
            backtest_config.load_strategy_config(strategy_list, re_timing_config)
            backtest_config.iter_round = iter_round

            self.update_meta_by_config(backtest_config)

            config_list.append(backtest_config)

        self.config_list = config_list

        return config_list
