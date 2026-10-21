"""Credential-free shadow consumer and adapter for AccountConfig.calc_order_amount.

Only GET requests exist in this module. It never imports startup.py/config.py,
creates an exchange account client, or submits/cancels orders.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from data_center.contract import atomic_json, validate
from data_center.base_strategy import STRATEGY_VERSION


# Live authorization is expressed as a cash budget. The strategy keeps its
# leverage multiplier of 2, so a 1,500 USDT budget produces at most 3,000 USDT
# gross target notional on a 3,000 USDT account. Live callers pass this cap
# explicitly; the generic planner remains usable for historical tests.
LIVE_BUDGET_CAP_USDT = 1500.0
LIVE_EQUITY_FRACTION = 0.50
LIVE_GROSS_CAP_USDT = 3000.0


def live_budget(equity: float) -> float:
    """Return the authorized strategy cash budget for a live account."""
    if isinstance(equity, bool) or not math.isfinite(float(equity)) or float(equity) <= 0:
        raise ValueError("invalid equity")
    return min(LIVE_BUDGET_CAP_USDT, float(equity) * LIVE_EQUITY_FRACTION)


def get_json(url, timeout=15):
    with urlopen(url, timeout=timeout) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("response too large")
    return json.loads(raw)


def market_rules(payload):
    result = {}
    for row in payload["symbols"]:
        if row.get("quoteAsset") != "USDT" or row.get("contractType") != "PERPETUAL":
            continue
        filters = {r["filterType"]: r for r in row["filters"]}
        lot = filters["LOT_SIZE"]
        market_lot = filters.get("MARKET_LOT_SIZE", {})
        result[row["symbol"]] = {"step": lot["stepSize"], "min_qty": lot["minQty"],
                                 "max_qty": lot["maxQty"], "market_step": market_lot.get("stepSize"),
                                 "market_min_qty": market_lot.get("minQty"), "market_max_qty": market_lot.get("maxQty"),
                                 "status": row["status"],
                                 "min_notional": filters.get("MIN_NOTIONAL", {}).get("notional")}
    return result


def plan(snapshot, equity, positions, prices, rules, now=None, *, diagnostic=False,
         expected_strategy=STRATEGY_VERSION, expected_formula=None,
         budget_usdt=None, max_gross_notional=None, manual_late=False):
    if manual_late and diagnostic:
        raise ValueError('manual late execution cannot use a preview')
    if diagnostic and snapshot.get('diagnostic_only') is not True:
        raise ValueError('diagnostic simulation requires explicit preview')
    from data_center.base_strategy import FORMULA_VERSION
    validate(snapshot, now=now, actionable=not (diagnostic or manual_late), expected_strategy=expected_strategy,
             expected_formula=expected_formula or FORMULA_VERSION)
    if manual_late and (snapshot.get('status')!='complete' or snapshot.get('blockers')
                        or snapshot.get('diagnostic_only')
                        or snapshot.get('coverage',{}).get('observed') != snapshot.get('coverage',{}).get('expected')
                        or not snapshot.get('coverage',{}).get('expected')):
        raise ValueError('manual late signal is not complete')
    if isinstance(equity, bool) or not math.isfinite(equity) or equity <= 0:
        raise ValueError("invalid equity")
    if any(not math.isfinite(float(v)) for v in positions.values()):
        raise ValueError("invalid current positions")
    allocations = snapshot["allocations"]
    if budget_usdt is not None:
        if isinstance(budget_usdt, bool):
            raise ValueError("invalid strategy budget")
        budget_usdt = float(budget_usdt)
        if not math.isfinite(budget_usdt) or budget_usdt <= 0:
            raise ValueError("invalid strategy budget")
        if budget_usdt > live_budget(equity):
            raise ValueError("strategy budget exceeds authorized cap")
        max_gross_notional = (LIVE_GROSS_CAP_USDT if max_gross_notional is None
                              else float(max_gross_notional))
        if not math.isfinite(max_gross_notional) or max_gross_notional <= 0:
            raise ValueError("invalid gross notional cap")
        max_gross_notional = min(max_gross_notional, LIVE_GROSS_CAP_USDT, float(equity))
    gross_target = Decimal("0")
    output = []
    for symbol in sorted(set(allocations) | set(positions)):
        price = float(prices.get(symbol, 0))
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"missing executable reference price: {symbol}")
        rule = rules.get(symbol)
        if not rule or rule["status"] != "TRADING":
            raise ValueError(f"missing/trading-inactive market rule: {symbol}")
        def number(key, *, zero=False):
            try:
                value = Decimal(str(rule[key]))
            except (KeyError, ArithmeticError, ValueError):
                raise ValueError(f"missing/invalid market rule: {symbol}/{key}") from None
            if not value.is_finite() or value < 0 or (not zero and value == 0):
                raise ValueError(f"invalid market rule: {symbol}/{key}")
            return value

        step = number("step")
        min_qty, max_qty = number("min_qty", zero=True), number("max_qty")
        min_notional = number("min_notional")
        # Capped TOP50 plans require the actual MARKET filter. Missing fields
        # must never silently inherit LOT_SIZE. A zero market step is disabled.
        if budget_usdt is not None or "market_step" in rule:
            market_step = number("market_step", zero=True)
            min_qty = max(min_qty, number("market_min_qty", zero=True))
            max_qty = min(max_qty, number("market_max_qty"))
            if market_step:
                scale = 10 ** max(0, -step.as_tuple().exponent, -market_step.as_tuple().exponent)
                step = Decimal(math.lcm(int(step * scale), int(market_step * scale))) / scale
        if min_qty > max_qty:
            raise ValueError(f"inconsistent market quantity bounds: {symbol}")
        allocation = float(allocations.get(symbol, 0.))
        basis = budget_usdt if budget_usdt is not None else equity
        target_notional = basis * snapshot["leverage"] * allocation
        raw_qty = Decimal(str(target_notional)) / Decimal(str(price))
        target_qty = (raw_qty / step).to_integral_value(rounding=ROUND_DOWN) * step
        current_qty = Decimal(str(positions.get(symbol, 0.)))
        delta = target_qty - current_qty
        if delta % step:
            raise ValueError(f"current position incompatible with market lot step: {symbol}")
        exiting = target_qty == 0 and current_qty != 0
        if delta and (abs(delta) > max_qty or abs(delta) < min_qty
                      or (not exiting and abs(delta) * Decimal(str(price)) < min_notional)):
            raise ValueError(f"unexecutable lot/min-notional: {symbol}; no automatic portfolio rescaling")
        if allocation and target_qty == 0:
            raise ValueError(f"target rounds to zero: {symbol}; no silent leg removal")
        gross_target += abs(target_qty * Decimal(str(price)))
        # Native framework input uses unsigned target quantity with its own
        # direction handling upstream; calc_order_amount itself accepts signed
        # 目标持仓量 and subtracts current signed positions.
        output.append({"symbol": symbol, "symbol_type": "swap", "close": price,
                       "candle_begin_time": snapshot["window_end"],
                       "target_alloc_ratio": allocation, "target_notional": target_notional,
                       "目标持仓量": float(target_qty), "当前持仓量": float(current_qty),
                       "实际下单量": float(delta), "交易模式": "清仓" if exiting else "调仓",
                       "side": "BUY" if delta > 0 else "SELL" if delta < 0 else None,
                       "order_type": "MARKET",
                       "submitted_to_exchange": False})
    if budget_usdt is not None and gross_target > Decimal(str(max_gross_notional)):
        raise ValueError(f"gross target notional exceeds live cap: {gross_target:.8f}")
    return output


def framework_targets(orders):
    """Convert plan rows to native calc_order_amount input; no account initialization."""
    import pandas as pd
    frame = pd.DataFrame(orders, columns=["candle_begin_time", "symbol", "symbol_type", "close", "目标持仓量"])
    frame["candle_begin_time"] = pd.to_datetime(frame["candle_begin_time"], utc=True)
    return frame


def run_once(url, root, equity):
    root = Path(root)
    now = datetime.now(timezone.utc)
    try:
        snapshot = get_json(url.rstrip("/") + "/v1/targets/latest")
        validate(snapshot, now=now)
        state_path = root / "shadow_portfolio.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        period = snapshot["window_end"]
        if state.get("last_window_end", "") >= period:
            status = {"status": "already_consumed", "snapshot_id": snapshot["snapshot_id"]}
        else:
            prices = {r["symbol"]: float(r["price"]) for r in get_json("https://fapi.binance.com/fapi/v1/ticker/price")}
            rules = market_rules(get_json("https://fapi.binance.com/fapi/v1/exchangeInfo"))
            orders = plan(snapshot, equity, state.get("positions", {}), prices, rules)
            atomic_json(root / "plans" / f"{snapshot['snapshot_id']}.json",
                        {"snapshot_id": snapshot["snapshot_id"], "equity_basis": "fixed_shadow_budget",
                         "equity": equity, "orders": orders, "submitted_to_exchange": False})
            atomic_json(state_path, {"last_window_end": period, "last_snapshot_id": snapshot["snapshot_id"],
                                    "positions": {r["symbol"]: r["目标持仓量"] for r in orders if r["目标持仓量"]},
                                    "simulation_only": True})
            status = {"status": "shadow_planned", "snapshot_id": snapshot["snapshot_id"], "rows": len(orders)}
    except HTTPError as exc:
        # 503 carries exact data-readiness blockers. Do not turn it into zeros.
        status = {"status": "blocked", "http_status": exc.code}
        try:
            error = json.loads(exc.read(16 * 1024 * 1024))
            status["blockers"] = error.get("blockers", [error.get("error", "upstream_unavailable")])
        except ValueError:
            status["blockers"] = ["invalid_upstream_response"]
    except (ValueError, KeyError, OSError) as exc:
        status = {"status": "blocked", "reason": str(exc)}
    status.update(checked_at=now.isoformat(), submitted_to_exchange=False, mode="shadow")
    atomic_json(root / "status.json", status)
    print(json.dumps(status, ensure_ascii=False))
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--root", required=True)
    p.add_argument("--equity", type=float, default=1000.)
    a = p.parse_args()
    if not a.url.startswith("http://127.0.0.1:"):
        p.error("use a loopback SSH tunnel")
    run_once(a.url, a.root, a.equity)


if __name__ == "__main__":
    main()
