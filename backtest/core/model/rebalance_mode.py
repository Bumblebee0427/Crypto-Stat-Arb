import inspect

from core import rebalance


def get_rebalance_mode_names():
    reb_modes = inspect.getmembers(rebalance, inspect.isclass)
    return [mode[0] for mode in reb_modes]


class RebalanceMode:
    ALWAYS = 'always'
    EQUITY_RATIO = 'equity_ratio'
    POSITION_RATIO = 'position_ratio'

    def __init__(self, mode, params):
        self.mode = mode
        self.params = params

        if hasattr(rebalance, mode):
            self.reb_cls = getattr(rebalance, mode)
        else:
            mode_names = get_rebalance_mode_names()
            raise ValueError(f'不支持的 Rebalance 模式 {mode}, 目前只支持 {mode_names}')

    def __repr__(self):
        return f'{self.mode}({self.params})'

    def __str__(self):
        match self.mode:
            case 'RebAlways':
                return f'每个周期rebalance'
            case 'RebByEquityRatio':
                return f'调仓金额大于资产{self.params["min_order_usdt_ratio"] * 100}%时进行rebalance'
            case 'RebByPositionRatio':
                return f'调仓金额大于币种持仓{self.params["min_order_usdt_ratio"] * 100}%时进行rebalance'
            case _:
                return f'{self.mode}({self.params})'

    def create(self, spot_lot_sizes, swap_lot_sizes):
        params = {'spot_lot_sizes': spot_lot_sizes, 'swap_lot_sizes': swap_lot_sizes}
        params.update(self.params)

        required_param_names = list(inspect.signature(self.reb_cls.__init__).parameters.keys())
        params = {k: v for k, v in params.items() if k in required_param_names}
        return self.reb_cls(**params)

    @classmethod
    def init(cls, config):
        if config is None:
            return cls('RebAlways', {})

        return cls(**config)
