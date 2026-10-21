import numpy as np
import numba as nb
from numba.experimental import jitclass

# 纯多模式仅使用 97% 的资金
LONG_ONLY_EQUITY_RATIO = 0.97


@nb.njit
def calc_target_lots_by_ratio(equity, prices, ratios, lot_sizes):
    """
    根据目标持仓比例，计算目标持仓手数
    """
    # 初始化目标持仓手数
    target_lots = np.zeros(len(lot_sizes), dtype=np.int64)

    # 每个币分配目标持仓资金(带方向)
    target_equity = equity * ratios

    # 目标持仓资金大于 0.01U 则认为是有效持仓
    # 同时要求 价格 和 每手币数 都不为 0，如果数据为 0，后面计算出现 除0 操作，造成数据位数溢出
    mask = np.logical_and(np.abs(target_equity) > 0.01, np.logical_and(prices != 0, lot_sizes != 0))

    # 为有效持仓分配目标持仓手数, 手数 = 目标持仓资金 / 币价 / 每手币数
    target_lots[mask] = (target_equity[mask] / prices[mask] / lot_sizes[mask]).astype(np.int64)

    return target_lots


@nb.njit
def calc_delta_lots_amount(target_lots, current_lots, prices, lot_sizes):
    """
    计算调仓手数和金额
    """
    # 调仓手数 = 目标持仓手数 - 当前持仓手数
    delta_lots = target_lots - current_lots

    # 初始化调仓金额为 0
    delta_amount = np.zeros(len(lot_sizes), dtype=np.float64)

    # 对需要调仓的 symbol 计算调仓金额(绝对值)
    # 调仓金额 = abs(调仓手数) * 每手币数 * 币价
    mask = (delta_lots != 0)
    delta_amount[mask] = np.abs(delta_lots[mask]) * lot_sizes[mask] * prices[mask]

    return delta_lots, delta_amount


@nb.njit
def filter_deltas(target_lots, current_lots, delta_lots, delta_amount, min_order_limit):
    # (当前持仓手数 == 0) 且 (目标持仓手数 != 0), 是建仓
    mask_builds = np.logical_and(current_lots == 0, target_lots != 0)

    # (当前持仓手数 != 0) 且 (目标持仓手数 == 0), 是清仓
    mask_liquidations = np.logical_and(current_lots != 0, target_lots == 0)

    # 建仓和清仓
    mask_bld_liq = np.logical_or(mask_builds, mask_liquidations)

    # 对于除建仓和清仓的以外的调仓，若调仓金额小于最小调仓金额，则标记为无效调仓
    mask_invalid = np.logical_and(delta_amount < min_order_limit, np.logical_not(mask_bld_liq))

    # 清除无效调仓
    delta_lots[mask_invalid] = 0

    return delta_lots


@jitclass
class RebAlways:
    """
    默认 Rebalance 模式，根据目标持仓比例调仓，不添加其它限制
    """
    spot_lot_sizes: nb.float64[:]  # 每手币数，表示一手加密货币中包含的币数
    swap_lot_sizes: nb.float64[:]

    def __init__(self, spot_lot_sizes, swap_lot_sizes):
        n_syms_spot = len(spot_lot_sizes)
        n_syms_swap = len(swap_lot_sizes)

        self.spot_lot_sizes = np.zeros(n_syms_spot, dtype=np.float64)
        self.spot_lot_sizes[:] = spot_lot_sizes

        self.swap_lot_sizes = np.zeros(n_syms_swap, dtype=np.float64)
        self.swap_lot_sizes[:] = swap_lot_sizes

    def calc_lots(self, equity, spot_prices, spot_lots, spot_ratios, swap_prices, swap_lots, swap_ratios):
        """
        计算每个币种的目标手数
        :param equity: 总权益
        :param spot_prices: 现货最新价格
        :param spot_lots: 现货当前持仓手数
        :param spot_ratios: 现货币种的资金比例
        :param swap_prices: 合约最新价格
        :param swap_lots: 合约当前持仓手数
        :param swap_ratios: 合约币种的资金比例
        :return: tuple[现货目标手数, 合约目标手数]
        """
        is_spot_only = False

        # 合约总权重小于极小值，认为是纯多(纯现货)模式
        if np.sum(np.abs(swap_ratios)) < 1e-6:
            is_spot_only = True
            equity *= LONG_ONLY_EQUITY_RATIO  # 留一部分的资金作为缓冲

        # 直接计算现货目标持仓手数
        spot_target_lots = calc_target_lots_by_ratio(equity, spot_prices, spot_ratios, self.spot_lot_sizes)

        if is_spot_only:
            swap_target_lots = np.zeros(len(self.swap_lot_sizes), dtype=np.int64)
            return spot_target_lots, swap_target_lots

        # 直接计算合约目标持仓手数
        swap_target_lots = calc_target_lots_by_ratio(equity, swap_prices, swap_ratios, self.swap_lot_sizes)

        return spot_target_lots, swap_target_lots


@jitclass
class RebByEquityRatio:
    """
    预计调仓金额大于总权益百分比才调仓
    """

    spot_lot_sizes: nb.float64[:]  # 每手币数，表示一手加密货币中包含的币数
    swap_lot_sizes: nb.float64[:]

    min_order_usdt_ratio: float

    def __init__(self, spot_lot_sizes, swap_lot_sizes, min_order_usdt_ratio):
        n_syms_spot = len(spot_lot_sizes)
        n_syms_swap = len(swap_lot_sizes)

        self.spot_lot_sizes = np.zeros(n_syms_spot, dtype=np.float64)
        self.spot_lot_sizes[:] = spot_lot_sizes

        self.swap_lot_sizes = np.zeros(n_syms_swap, dtype=np.float64)
        self.swap_lot_sizes[:] = swap_lot_sizes

        self.min_order_usdt_ratio = min_order_usdt_ratio

    def _calc(self, equity, prices, current_lots, ratios, lot_sizes):
        # 1. 计算目标持仓手数
        target_lots = calc_target_lots_by_ratio(equity, prices, ratios, lot_sizes)

        # 2. 最小调仓金额 = 账户总权益 * 最小下单比例
        min_order_limit = equity * self.min_order_usdt_ratio

        # 3. 计算调仓手数(带方向)和调仓金额(绝对值)
        delta_lots, delta_amount = calc_delta_lots_amount(target_lots, current_lots, prices, lot_sizes)

        # 4. 过滤调仓金额小于最小调仓金额的调仓，但建仓和清仓不过滤
        delta_lots = filter_deltas(target_lots, current_lots, delta_lots, delta_amount, min_order_limit)

        # 5. 根据过滤后的调仓手数，计算目标持仓手数
        target_lots = current_lots + delta_lots
        return target_lots

    def calc_lots(self, equity, spot_prices, spot_lots, spot_ratios, swap_prices, swap_lots, swap_ratios):
        """
        计算每个币种的目标手数
        :param equity: 总权益
        :param spot_prices: 现货最新价格
        :param spot_lots: 现货当前持仓手数
        :param spot_ratios: 现货币种的资金比例
        :param swap_prices: 合约最新价格
        :param swap_lots: 合约当前持仓手数
        :param swap_ratios: 合约币种的资金比例
        :return: tuple[现货目标手数, 合约目标手数]
        """
        is_spot_only = False

        # 合约总权重小于极小值，认为是纯多(纯现货)模式
        if np.sum(np.abs(swap_ratios)) < 1e-6:
            is_spot_only = True
            equity *= LONG_ONLY_EQUITY_RATIO  # 留一部分的资金作为缓冲

        # 现货目标持仓手数
        spot_target_lots = self._calc(equity, spot_prices, spot_lots, spot_ratios, self.spot_lot_sizes)

        if is_spot_only:
            swap_target_lots = np.zeros(len(swap_prices), dtype=np.int64)
            return spot_target_lots, swap_target_lots

        # 合约目标持仓手数
        swap_target_lots = self._calc(equity, swap_prices, swap_lots, swap_ratios, self.swap_lot_sizes)
        return spot_target_lots, swap_target_lots


@jitclass
class RebByPositionRatio:
    """
    预计调仓金额大于标的分配资金百分比才调仓
    """

    spot_lot_sizes: nb.float64[:]  # 每手币数，表示一手加密货币中包含的币数
    swap_lot_sizes: nb.float64[:]

    min_order_usdt_ratio: float

    def __init__(self, spot_lot_sizes, swap_lot_sizes, min_order_usdt_ratio):
        n_syms_spot = len(spot_lot_sizes)
        n_syms_swap = len(swap_lot_sizes)

        self.spot_lot_sizes = np.zeros(n_syms_spot, dtype=np.float64)
        self.spot_lot_sizes[:] = spot_lot_sizes

        self.swap_lot_sizes = np.zeros(n_syms_swap, dtype=np.float64)
        self.swap_lot_sizes[:] = swap_lot_sizes

        self.min_order_usdt_ratio = min_order_usdt_ratio

    def _calc(self, equity, prices, current_lots, ratios, lot_sizes):
        # 1. 计算目标持仓手数
        target_lots = calc_target_lots_by_ratio(equity, prices, ratios, lot_sizes)

        # 2. 计算最小调仓金额

        # 当前持仓价值 = abs(当前持仓手数) * 每手币数 * 币价
        current_symbol_value = np.abs(current_lots) * lot_sizes * prices

        # 最小调仓金额 = 当前持仓价值 * 最小下单比例
        min_order_limit = current_symbol_value * self.min_order_usdt_ratio

        # 3. 计算调仓手数(带方向)和调仓金额(绝对值)
        delta_lots, delta_amount = calc_delta_lots_amount(target_lots, current_lots, prices, lot_sizes)

        # 4. 过滤调仓金额小于最小调仓金额的调仓，但建仓和清仓不过滤
        delta_lots = filter_deltas(target_lots, current_lots, delta_lots, delta_amount, min_order_limit)

        # 5. 根据过滤后的调仓手数，计算目标持仓手数
        target_lots = current_lots + delta_lots
        return target_lots

    def calc_lots(self, equity, spot_prices, spot_lots, spot_ratios, swap_prices, swap_lots, swap_ratios):
        """
        计算每个币种的目标手数
        :param equity: 总权益
        :param spot_prices: 现货最新价格
        :param spot_lots: 现货当前持仓手数
        :param spot_ratios: 现货币种的资金比例
        :param swap_prices: 合约最新价格
        :param swap_lots: 合约当前持仓手数
        :param swap_ratios: 合约币种的资金比例
        :return: tuple[现货目标手数, 合约目标手数]
        """
        is_spot_only = False

        # 合约总权重小于极小值，认为是纯多(纯现货)模式
        if np.sum(np.abs(swap_ratios)) < 1e-6:
            is_spot_only = True
            equity *= LONG_ONLY_EQUITY_RATIO  # 留一部分的资金作为缓冲

        # 现货目标持仓手数
        spot_target_lots = self._calc(equity, spot_prices, spot_lots, spot_ratios, self.spot_lot_sizes)

        if is_spot_only:
            swap_target_lots = np.zeros(len(swap_prices), dtype=np.int64)
            return spot_target_lots, swap_target_lots

        # 合约目标持仓手数
        swap_target_lots = self._calc(equity, swap_prices, swap_lots, swap_ratios, self.swap_lot_sizes)
        return spot_target_lots, swap_target_lots
