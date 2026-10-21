"""Sequential legacy accounting with a policy callback at each day boundary."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .contracts import AccountState, write_json


def target_lots(equity, prices, weights, steps):
    from core.rebalance import calc_target_lots_by_ratio
    # Numba and numpy reduction/rounding can differ by a lot at an integer boundary.
    # Delegate to the original quantity projector, preserving exact binary arithmetic.
    clean_weights = np.asarray(weights, dtype=float).copy()
    clean_weights[~np.isfinite(prices)] = 0.
    return calc_target_lots_by_ratio(equity, prices, clean_weights, steps)


def settle(state, price, steps):
    mask = (state.lots != 0) & np.isfinite(price)
    pnl = (price[mask]-state.last_prices[mask])*steps[mask]*state.lots[mask]
    long_pnl = pnl[state.lots[mask] > 0].sum()
    short_pnl = pnl[state.lots[mask] < 0].sum()
    state.equity += pnl.sum()
    state.last_prices[np.isfinite(price)] = price[np.isfinite(price)]
    return float(long_pnl), float(short_pnl)


def resize_targets(desired, current, prices, steps, equity, weights, threshold,
                   *, force=False, max_side_drift=.05):
    """Decide at CLOSE, without next-hour execution prices; mandatory exits always pass."""
    if not 0 <= threshold < 1 or not 0 < max_side_drift < 1:
        raise ValueError('invalid resize threshold or side guard')
    if force or threshold == 0:
        return desired.copy(), 0
    actual = np.nan_to_num(current*steps*prices/equity)
    # Refresh the complete portfolio when a side drifts >5% of its target budget.
    for sign in [-1, 1]:
        wanted = np.abs(weights[weights*sign > 0]).sum()
        held = np.abs(actual[actual*sign > 0]).sum()
        if abs(held-wanted) > max_side_drift*max(wanted, 1e-8):
            return desired.copy(), 0
    within = np.abs(actual-weights) <= threshold*np.abs(weights)
    mandatory = (weights == 0) | (current*weights <= 0) | ~np.isfinite(prices)
    suppress = within & ~mandatory & (desired != current)
    result = desired.copy(); result[suppress] = current[suppress]
    return result, int(suppress.sum())


class Ledger:
    def __init__(self, out, enabled):
        self.out, self.enabled = out, enabled
        self.buffers, self.writers = {}, {}

    def add(self, name, frame):
        if not self.enabled or frame.empty:
            return
        self.buffers.setdefault(name, []).append(frame)
        if len(self.buffers[name]) >= 240:
            self.flush(name)

    def flush(self, name):
        import pyarrow as pa
        import pyarrow.parquet as pq
        frames = self.buffers.get(name, [])
        if not frames:
            return
        table = pa.Table.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False)
        if name not in self.writers:
            self.writers[name] = pq.ParquetWriter(self.out/(name+'.parquet'), table.schema, compression='zstd')
        self.writers[name].write_table(table)
        self.buffers[name] = []

    def close(self):
        for name in self.buffers:
            self.flush(name)
        for writer in self.writers.values():
            writer.close()


def replay(data, policy=None, *, out: Path | None = None, initial_equity=None,
           cost_multiplier=1., ledger=False, verify_simulator=False,
           resize_threshold=0., max_side_drift=.05):
    n = len(data.symbols)
    state = AccountState(float(initial_equity or data.initial_equity),
                         np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n))
    from core.simulator import Simulator
    cash = state.equity
    sim = Simulator(0., data.steps, data.fee_rate*cost_multiplier,
                    np.zeros(n, np.int64), data.min_order)
    if out:
        out.mkdir(parents=True, exist_ok=True)
    sink = Ledger(out, ledger and out is not None)
    gold = None
    lot_mismatches = 0
    if verify_simulator:
        gold = Simulator(0., data.steps, data.fee_rate*cost_multiplier,
                         np.zeros(n, np.int64), data.min_order)
    active = np.zeros(n)
    rows, decisions = [], []
    labels = np.array(data.symbols)
    try:
        for i, t in enumerate(data.times):
            op, ex, cl, funding = (data.prices[k][i] for k in ['open', 'vwap1m', 'close', 'funding_rate'])
            before = state.equity
            old_lots = sim.lots.copy()
            old_prices = sim.last_prices.copy()
            valid = (old_lots != 0) & np.isfinite(op)
            if not np.isfinite(funding[valid]).all():
                raise ValueError('held symbol has missing realized funding at '+str(t))
            funding_amount = old_lots[valid]*data.steps[valid]*op[valid]*funding[valid]
            _, fc, pos_values = sim.on_open(op, funding, op)
            gross = np.abs(pos_values).sum()
            margin = (cash+sim.equity)/gross if gross else 10000.
            if margin < data.margin_rate:
                raise RuntimeError('liquidation at '+str(t))
            before_execution_prices = sim.last_prices.copy()
            delta = sim.target_lots-old_lots
            amount = np.abs(delta)*data.steps*ex
            fills = (delta != 0) & np.isfinite(ex) & (amount >= data.min_order)
            _, turnover, fee = sim.on_execution(ex)
            before_close_prices = sim.last_prices.copy()
            sim.on_close(cl)
            state.equity = cash+sim.equity
            state.lots = sim.lots.copy()
            state.last_prices = sim.last_prices.copy()
            leg_pnl = np.zeros(n)
            for lots, previous_price, new_price in [(old_lots, old_prices, op),
                    (old_lots, before_execution_prices, ex), (state.lots, before_close_prices, cl)]:
                mask = (lots != 0) & np.isfinite(new_price)
                contribution = (new_price[mask]-previous_price[mask])*data.steps[mask]*lots[mask]
                leg_pnl[mask] += contribution
            # A sign flip can contain both long and short carry within a bar.
            long_pnl = short_pnl = 0.
            for lots, previous_price, new_price in [(old_lots, old_prices, op),
                    (old_lots, before_execution_prices, ex), (state.lots, before_close_prices, cl)]:
                mask = (lots != 0) & np.isfinite(new_price)
                contribution = (new_price[mask]-previous_price[mask])*data.steps[mask]*lots[mask]
                long_pnl += float(contribution[lots[mask] > 0].sum())
                short_pnl += float(contribution[lots[mask] < 0].sum())
            values = state.lots*data.steps*cl
            lv = float(np.nansum(values[state.lots > 0])); sv = -float(np.nansum(values[state.lots < 0]))
            if gold is not None:
                gold.on_open(op, funding, op); gold.on_execution(ex); gold.on_close(cl)
                lot_mismatches += int(np.count_nonzero(gold.lots != state.lots))
                if gold.equity+cash != state.equity:
                    raise AssertionError('independent Simulator accounting drift')
            rows.append({'time': t, 'equity': state.equity, 'net_return': state.equity/before-1,
                         'turnover': turnover, 'fee': fee, 'funding_fee': fc,
                         'long_price_pnl': long_pnl, 'short_price_pnl': short_pnl,
                         'long_pos_value': lv, 'short_pos_value': sv, 'marginRatio': margin,
                         'rebalance_fee': fee if t.hour != 0 else 0.,
                         'equity_before': before})
            if sink.enabled:
                held = np.flatnonzero(state.lots)
                sink.add('positions', pd.DataFrame({'time': t, 'symbol': labels[held],
                    'lots': state.lots[held], 'quantity': state.lots[held]*data.steps[held],
                    'mark_price': cl[held], 'notional': values[held]}))
                filled = np.flatnonzero(fills)
                sink.add('trades', pd.DataFrame({'time': t, 'symbol': labels[filled],
                    'quantity': delta[filled]*data.steps[filled], 'price': ex[filled],
                    'notional': amount[filled], 'fee': amount[filled]*data.fee_rate*cost_multiplier}))
                paid = np.flatnonzero(valid & (funding != 0))
                # funding_amount was calculated on OLD lots before execution.
                amounts = np.zeros(n); amounts[valid] = funding_amount
                sink.add('funding_events', pd.DataFrame({'time': t, 'symbol': labels[paid],
                    'rate': funding[paid], 'amount': amounts[paid]}))
                per_coin_fee = np.zeros(n)
                per_coin_fee[fills] = amount[fills]*data.fee_rate*cost_multiplier
                attributed = np.flatnonzero((leg_pnl != 0) | (amounts != 0) | (per_coin_fee != 0))
                sink.add('pnl_attribution', pd.DataFrame({'time': t, 'symbol': labels[attributed],
                    'price_pnl': leg_pnl[attributed], 'fee': per_coin_fee[attributed],
                    'funding_cost': amounts[attributed],
                    'net_pnl': leg_pnl[attributed]-per_coin_fee[attributed]-amounts[attributed]}))
            previous_active = active.copy()
            if t.hour == 23:
                active = data.weights[i].copy()
                if policy is not None:
                    active, records = policy(i, t+pd.Timedelta(hours=1), active, state)
                    decisions.extend(records)
            # Direct-through uses the actual hourly frozen matrix, including final cleanup.
            if policy is None:
                active = data.weights[i].copy()
            desired = target_lots(state.equity, cl, active, data.steps)
            threshold = resize_threshold(t+pd.Timedelta(hours=1)) if callable(resize_threshold) else resize_threshold
            state.targets, suppressed = resize_targets(desired, state.lots, cl, data.steps,
                state.equity, active, threshold, force=t.hour == 23 or not np.array_equal(active, previous_active),
                max_side_drift=max_side_drift)
            rows[-1]['resize_suppressed_symbols'] = suppressed
            rows[-1]['actual_gross_leverage'] = (lv+sv)/state.equity
            sim.set_target_lots(state.targets)
            if gold is not None:
                from core.rebalance import calc_target_lots_by_ratio
                gold.set_target_lots(calc_target_lots_by_ratio(gold.equity+cash, cl, active, data.steps))
                lot_mismatches += int(np.count_nonzero(gold.target_lots != state.targets))
    finally:
        sink.close()
    result = pd.DataFrame(rows)
    if out:
        result.to_parquet(out/'equity_hourly.parquet', index=False)
        if decisions:
            pd.DataFrame(decisions).to_parquet(out/'decisions.parquet', index=False)
    return result, {'simulator_lot_mismatches': lot_mismatches}


def parity(data, out):
    result, info = replay(data, out=out, verify_simulator=True)
    metrics = {}
    ref = data.reference
    for name in ['equity', 'fee', 'funding_fee', 'long_pos_value', 'short_pos_value', 'turnover']:
        error = np.abs(result[name].to_numpy()-ref[name].to_numpy())
        metrics[name+'_max_abs_error'] = float(error.max())
    relative = np.abs(result.equity.to_numpy()/ref.equity.to_numpy()-1)
    metrics.update(info, hours=len(result), max_relative_equity_error=float(relative.max()))
    metrics['passed'] = bool(relative.max() < 1e-10 and
        metrics['fee_max_abs_error'] < 1e-6 and metrics['funding_fee_max_abs_error'] < 1e-6 and
        info['simulator_lot_mismatches'] == 0)
    write_json(out/'baseline_parity.json', metrics)
    if not metrics['passed']:
        raise AssertionError('baseline parity failed: '+str(metrics))
    return metrics
