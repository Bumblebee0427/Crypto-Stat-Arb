"""Frozen rule baseline. Weights are allocation ratios BEFORE account leverage.

Evidence: saved config.pkl and final_select_results.pkl from the May-31 baseline.
24H/offset0 selects the 23:00 candle, executed after it closes at 00:00 UTC.
BTC shift1 is ONE HOURLY candle, not one daily candle.
"""
from __future__ import annotations

import math
import re

STRATEGY_VERSION = "bigultra_smallqty_btc30_base_20260531_v1"
FORMULA_VERSION = "ref_notional_nearest_half_up_closed_both_v2"
SOURCE_VERSION = "binance_individual_trade_ts_side_ms_v2"
BASE_LEVERAGE = 2.0
MAIN = "big_ultra_harmonic_rm240_mean_1d"
SMALL = "small_qty_misalignment_rankcorr_lag5_1d"
RECIPE = {
    "strategy_version": STRATEGY_VERSION,
    "main_lindecay": 12, "small_long_roll": 8, "small_short_roll": 4,
    "small_long_pct_min": 0.40, "small_short_pct_max": 0.35,
    "long_select_pct": 0.20, "short_select_pct": 0.15,
    "long_allocation": 0.50, "short_allocation": 0.50,
    "btc_trend_days": 30, "btc_shift_hours": 1,
    "btc_threshold": 0.20, "protected_short_scale": 0.45,
    "account_leverage": BASE_LEVERAGE, "backtest_total_cost_bps": 8,
    "rebalance_utc": "00:00", "blacklist": ["BTCUSDT", "ETHUSDT"],
}


def symbol_name(symbol):
    result = str(symbol).removesuffix("-PERP").replace("-", "")
    # Research history also contains delisted aliases such as CVX_SW0-USDT.
    if not re.fullmatch(r"\w+USDT", result):
        raise ValueError(f"unsupported USD-M symbol: {symbol}")
    return result


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values):
    """pandas rank(method='average', pct=True), including ties."""
    ordered = sorted(values, key=values.get)
    output = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for symbol in ordered[i:j]:
            output[symbol] = ((i + 1 + j) / 2) / len(ordered)
        i = j
    return output


def select(rows, btc_trend, volume_top_pct=None):
    """Rows contain main/small_long/small_short; missingness is leg-specific.

    For historical parity BTC NaN disables only the extra short component.
    Publication adds a stricter missing-BTC gate before consumers can act.
    """
    normalized = {}
    if volume_top_pct is not None and not 0 < volume_top_pct <= 1:
        raise ValueError('invalid volume cutoff')
    for symbol, row in rows.items():
        symbol = symbol_name(symbol)
        if symbol in normalized:
            raise ValueError("duplicate normalized symbol")
        if symbol not in RECIPE["blacklist"] and row.get("tradable", True):
            normalized[symbol] = row

    def leg(field, cutoff, keep_high, select_pct, main_high, extra=False):
        available = {s: r for s, r in normalized.items()
                     if finite(r.get("main")) and finite(r.get(field))}
        if volume_top_pct is not None:
            available = {s:r for s,r in available.items() if finite(r.get('quote_volume_mean')) and r['quote_volume_mean'] >= 0}
        if extra:
            # The saved framework drops NaN BTC scores before cross-section
            # ranking. Newly listed symbols can therefore change the EXTRA
            # short pool even though the BTC price series is market-wide.
            available = {s: r for s, r in available.items() if finite(r.get("btc", btc_trend))}
        pcts = percentile({s: r[field] for s, r in available.items()})
        pool = [s for s in available if
                (pcts[s] >= cutoff if keep_high else pcts[s] <= cutoff)]
        if volume_top_pct is not None:
            # Both filters rank the same valid leg universe, independently.
            # Ranking small-qty AFTER filtering TOP50 changes the strategy.
            volume_pcts = percentile({s:-r['quote_volume_mean'] for s,r in available.items()})
            pool = [s for s in pool if volume_pcts[s] <= volume_top_pct]
        if extra:
            pool = [s for s in pool if available[s].get("btc", btc_trend) <= .20]
        ordered = sorted(pool, key=lambda s: available[s]["main"], reverse=main_high)
        result = []
        previous = None
        rank = 0
        for i, s in enumerate(ordered, 1):
            if previous is None or available[s]["main"] != previous:
                rank, previous = i, available[s]["main"]
            if rank <= len(pool) * select_pct:
                result.append(s)
        return result, len(pool)

    longs, long_pool = leg("small_long", .40, True, .20, True)
    shorts, short_pool = leg("small_short", .35, False, .15, False)
    extra_shorts, _ = leg("small_short", .35, False, .15, False, extra=True)
    short_scale = 1.0 if finite(btc_trend) and btc_trend <= .20 else .45
    allocations = {}
    for selected, budget in ((longs, .5), (shorts, -.225), (extra_shorts, -.275)):
        for symbol in selected:
            allocations[symbol] = allocations.get(symbol, 0.) + budget / len(selected)
    return {"allocations": dict(sorted(allocations.items())), "longs": longs,
            "shorts": shorts, "extra_shorts": extra_shorts,
            "long_pool": long_pool, "short_pool": short_pool,
            "btc_trend": btc_trend if finite(btc_trend) else None,
            "short_scale": short_scale}
