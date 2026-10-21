#!/usr/bin/env python3
"""Monthly walk-forward evaluation of historical v3 alpha, with fixed settings.

Uses the daily live risk/optimizer functions and actual account budget convention.
No exchange or account API is called. A failed gate cannot issue a live release.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.alpha import fit_alpha, predict_alpha, alpha_fingerprint
from research.optimizer.alpha_bootstrap import alpha_policy
from research.optimizer.contracts import ResearchConfig, file_hash, digest, write_json
from research.optimizer.live import SPEC, MODEL, BLEND
from research.optimizer.factor_risk_tuning import estimate_grid
from research.optimizer.funding import forecast_funding
from research.optimizer.solve import solve_weights
from research.optimizer.validation import sortino_ratio
from research.optimizer.v3_history import temporal_panel_check, V3


def metrics(r):
    r = np.asarray(r, float); nav = np.r_[1., np.cumprod(1+r)]
    return {'days': len(r), 'cumulative_return': float(nav[-1]-1),
        'mean_daily_bps': float(r.mean()*10000),
        'annualized_sharpe': float(r.mean()/r.std(ddof=1)*np.sqrt(365)) if r.std(ddof=1) else None,
        'sortino': sortino_ratio(r), 'max_drawdown': float((nav/np.maximum.accumulate(nav)-1).min()),
        'worst_day': float(r.min())}


def bootstrap_difference(x):
    x = np.asarray(x); rng = np.random.default_rng(20261021); values = []
    for _ in range(2000):
        starts = rng.integers(0, len(x)-7+1, int(np.ceil(len(x)/7)))
        idx = np.concatenate([np.arange(s, s+7) for s in starts])[:len(x)]
        values.append(x[idx].mean()*10000)
    return {'mean_daily_increment_bps': float(x.mean()*10000),
        'block_days': 7, 'draws': 2000, 'mean_95ci_bps': np.quantile(values, [.025, .975]).tolist()}


def execution_inputs(args):
    root = args.history
    receipt = root/'execution_inputs.json'
    sources = {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
               for p in [args.market, args.market_tail] if p is not None}
    repair_path=root/'repaired_execution_close.parquet'
    if repair_path.exists():sources[str(repair_path)]={'size':repair_path.stat().st_size,'mtime_ns':repair_path.stat().st_mtime_ns}
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if (saved['sources'] == sources and saved.get('official_10m_sha256') == file_hash(root/'execution_prices_10m.parquet')
                and all(file_hash(root/k) == v for k, v in saved['outputs'].items())):
            apply_execution_labels(root, sources)
            return
    close, funding, prices = {}, {}, {}
    for source in [args.market, args.market_tail]:
        if source is None: continue
        raw = pd.read_pickle(source)
        for symbol, frame in raw.items():
            name = symbol.replace('-', '')
            x = frame.set_index('candle_begin_time').sort_index().loc['2026-06-30':'2026-10-09']
            if x.empty: continue
            for dest, col in [(close, 'close'), (funding, 'funding_fee')]:
                s = x[col]
                dest[name] = s.combine_first(dest[name]) if name in dest else s
            s = x.loc[x.index.hour == 1, 'avg_price_1m']; s.index = s.index.normalize()
            prices[name] = s.combine_first(prices[name]) if name in prices else s
        del raw
    close_frame=pd.DataFrame(close).sort_index()
    if repair_path.exists():close_frame=pd.read_parquet(repair_path).combine_first(close_frame).sort_index()
    close_frame.to_parquet(root/'execution_close.parquet')
    pd.DataFrame(funding).sort_index().to_parquet(root/'execution_funding.parquet')
    old = pd.read_parquet(root/'entry_prices_01h.parquet').pivot(index='asof', columns='symbol', values='price')
    pd.DataFrame(prices).combine_first(old).sort_index().to_parquet(root/'execution_prices_01h.parquet')
    official = pd.read_parquet(root/'execution_prices_10m.parquet')
    official.to_parquet(root/'execution_prices.parquet')
    apply_execution_labels(root, sources)
    write_json(receipt, {'sources': sources, 'official_10m_sha256':file_hash(root/'execution_prices_10m.parquet'),
        'outputs': {k: file_hash(root/k) for k in [
        'execution_close.parquet', 'execution_funding.parquet', 'execution_prices.parquet']}})


def apply_execution_labels(root, sources):
    # Complete the final evaluation labels from the independently archived market tail.
    panel = pd.read_parquet(root/'alpha_panel.parquet')
    if not (root/'alpha_panel_original_01h.parquet').exists():
        panel.to_parquet(root/'alpha_panel_original_01h.parquet', index=False)
    panel['entry_at'] = panel['asof']+pd.Timedelta(minutes=10)
    panel['exit_at'] = panel['asof']+pd.Timedelta(days=1, minutes=10)
    panel['execution_window_end'] = panel['asof']+pd.Timedelta(hours=1)
    panel['exit_window_end'] = panel['asof']+pd.Timedelta(days=1, hours=1)
    panel['label_available_at'] = panel['asof']+pd.Timedelta(days=1, hours=1, minutes=1)
    panel['sample_source'] = 'historical_v3_official_0010_0100_vwap'
    p = pd.read_parquet(root/'execution_prices.parquet')
    for col, offset in [('entry_price', 0), ('exit_price', 1)]:
        panel[col] = [p.at[t+pd.Timedelta(days=offset), s] if t+pd.Timedelta(days=offset) in p.index and s in p else np.nan
                      for t, s in zip(panel['asof'], panel.symbol)]
    panel['future_return'] = panel.exit_price/panel.entry_price-1
    panel.to_parquet(root/'alpha_panel.parquet', index=False)
    inside = panel.loc[(panel['asof'] >= p.index.min()) &
                       (panel['asof']+pd.Timedelta(days=1) <= p.index.max())]
    missing = inside.loc[inside.future_return.isna(),['asof','symbol','side','entry_price','exit_price']]
    missing.to_csv(root/'missing_label_rows.csv',index=False)
    write_json(root/'label_coverage.json', {'rows_within_benchmark_calendar':len(inside),
        'missing_benchmark_rows':len(missing), 'missing_fraction':len(missing)/len(inside) if len(inside) else None,
        'missing_labels_not_filled':True, 'missing_training_labels_excluded':True,
        'missing_evaluation_execution_prices_block_release':True})
    manifest = json.loads((root/'panel_manifest.json').read_text())
    manifest.update(panel_sha256=file_hash(root/'alpha_panel.parquet'),
        execution_tail_sources=sources, mature_rows=int(panel.future_return.notna().sum()),
        entry_benchmark='00:10–01:00 UTC fixed-window VWAP; official archive SHA256 verified')
    manifest['limitations'] = [s.replace('fixed 01:00 UTC benchmark', 'fixed 00:10 UTC benchmark')
                               for s in manifest['limitations']]
    write_json(root/'panel_manifest.json', manifest)


def performance_checks(primary):
    diff=primary.optimizer_trained_alpha-primary.optimizer_zero_alpha
    a,z=metrics(primary.optimizer_trained_alpha),metrics(primary.optimizer_zero_alpha)
    return {'mean_increment_positive':bool(diff.mean()>0),
        'sortino_not_lower':a['sortino'] is not None and z['sortino'] is not None and a['sortino']>=z['sortino'],
        'drawdown_deterioration_at_most_2pct':a['max_drawdown']>=z['max_drawdown']-.02,
        'exclude_best_increment_mean_nonnegative':bool(diff.drop(diff.idxmax()).mean()>=0)}


def evaluate(args):
    root, out = args.history, args.out; out.mkdir(parents=True, exist_ok=True)
    design = json.loads((root/'validation_design.json').read_text())
    from research.optimizer.delistings import load_delistings
    delistings=load_delistings(root)
    panel = temporal_panel_check(pd.read_parquet(root/'alpha_panel.parquet'))
    factors = pd.read_parquet(root/'risk/factors.parquet')
    returns = pd.read_parquet(root/'risk/returns.parquet')
    prices = pd.read_parquet(root/'execution_prices.parquet')
    closes = pd.read_parquet(root/'execution_close.parquet')
    realized_funding = pd.read_parquet(root/'execution_funding.parquet')
    events = pd.read_parquet(root/'settlements.parquet'); events['event_hour'] = events.time.dt.floor('h')
    if args.settlements:
        better = pd.read_parquet(args.settlements).copy(); better['symbol'] = better.symbol.str.replace('-', '', regex=False)
        events = pd.concat([events, better], ignore_index=True).drop_duplicates(['symbol', 'event_hour'], keep='last')
    funding_jitter=(events.time-events.time.dt.floor('h')).dt.total_seconds()
    if (funding_jitter > 1).any():
        raise ValueError('intrahour funding requires execution-window cashflow replay')
    # Official events can lag the nominal settlement boundary by milliseconds.
    # Under the bounded <=1s tolerance none crosses the 00:10–01:00 window.
    verified_rates=events.pivot_table(index='event_hour',columns='symbol',values='rate',aggfunc='sum')
    realized_funding=verified_rates.combine_first(realized_funding).sort_index()
    events = events.sort_values(['symbol', 'time']); event_map = dict(tuple(events.groupby('symbol')))
    cfg = ResearchConfig(risk_days=90)
    models, refits, predictions, risks, problems, observations = {}, [], [], {}, [], []
    days = pd.date_range(design['test_start'], design['test_end_inclusive'])
    for asof in days:
        rows = panel.loc[panel['asof'] == asof].sort_values('symbol').copy()
        cutoff = asof.to_period('M').start_time
        if cutoff not in models:
            model = fit_alpha(panel, cutoff, cfg); models[cutoff] = model
            train = panel.loc[(panel['asof'] >= cutoff-pd.Timedelta(days=365)) & (panel.label_available_at <= cutoff) & panel.future_return.notna()]
            refits.append({'fitted_at': str(cutoff), 'label_cutoff': str(model.label_cutoff),
                'model_fingerprint':alpha_fingerprint(model),
                'training_side_days': {str(k): int(v) for k, v in train.groupby('side')['asof'].nunique().items()},
                'supported_sides': sorted(model.models), 'models': {str(s): m.to_dict() for s, m in model.models.items()}})
        model = models[cutoff]
        if model.label_cutoff is not None and model.label_cutoff > cutoff: problems.append(str(asof)+':future training label')
        missing=[]
        for row in rows.itertuples():
            if pd.notna(row.future_return):continue
            event=delistings.get((asof,row.symbol))
            if (event is None or pd.isna(row.entry_price)
                    or not asof+pd.Timedelta(hours=1)<pd.Timestamp(event['settled_at'])<asof+pd.Timedelta(days=1,minutes=10)):
                missing.append(row.symbol)
        if rows.empty or missing:problems.append(str(asof)+':missing selection or execution label: '+','.join(missing))
        alpha, supported = predict_alpha(rows, model)
        rows['prediction'] = alpha; rows['supported'] = supported; predictions.append(rows)
        try:
            r = estimate_grid(returns.loc[:asof], factors.loc[:asof], rows.symbol.tolist(), asof, [SPEC])[MODEL]
            if r['info'].get('factor_observations', 0) < 60 or pd.Timestamp(r['info']['max_observation_at']) != asof:
                raise ValueError('insufficient or stale risk history')
            funding, fixed = [], []
            for s, n, ok in zip(rows.symbol, r['info']['samples'], supported):
                e = event_map.get(s, events.iloc[:0]); f, info = forecast_funding(e, asof, cfg)
                funding.append(f)
                fixed.append((n < 20 and not ok) or info['status'].startswith('missing_history') or info.get('stale', False))
            risks[asof] = r['covariance'], np.asarray(funding), np.asarray(fixed), np.asarray(r['info']['samples'])
            observations.append({'asof': str(asof), **r['info']})
        except (ValueError, KeyError) as exc: problems.append(str(asof)+':'+str(exc))
        print(json.dumps({'phase': 'forecasts', 'asof': str(asof.date()), 'coins': len(rows), 'alpha_supported': int(supported.sum())}), flush=True)
    predicted = pd.concat(predictions, ignore_index=True)
    predicted.to_parquet(out/'predictions.parquet', index=False)
    forecast_scores = []
    for (date, side), group in predicted.groupby(['asof', 'side']):
        y = group.future_return-group.future_return.mean()
        ic = group.prediction.corr(group.future_return, method='spearman') if group.prediction.nunique() > 1 else np.nan
        forecast_scores.append({'asof':date,'side':side,'rank_ic':ic,
            'mse_trained':float(((group.prediction-y)**2).mean()), 'mse_zero':float((y**2).mean()),
            'symbols':len(group)})
    pd.DataFrame(forecast_scores).to_csv(out/'forecast_quality.csv',index=False)
    write_json(out/'monthly_models.json', refits); write_json(out/'risk_diagnostics.json', observations)
    if problems:
        write_json(out/'validation.json', {'accepted': False, 'problems': problems, 'phase': 'input_validation'})
        raise ValueError('validation data incomplete; inspect validation.json')
    accounts, decisions = [], []; cases = ['baseline_equal', 'optimizer_zero_alpha', 'optimizer_trained_alpha']
    for scenario in (['adverse','favorable'] if delistings else ['adverse']):
        for multiplier in [1., 2.]:
            for case in cases:
                equity, positions = 3000., {}; records = []
                for asof, rows in predicted.groupby('asof'):
                    symbols = rows.symbol.tolist(); union = sorted(set(symbols) | set(positions))
                    # All exits need a real observed execution benchmark as well.
                    entry = prices.loc[asof, union].to_numpy(float)
                    if not np.isfinite(entry).all() or (entry <= 0).any(): raise ValueError('missing forced-exit execution price')
                    budget = min(1500., equity*.5)
                    b = rows.set_index('symbol').baseline.reindex(union, fill_value=0.).to_numpy()
                    previous = np.array([positions.get(s, 0.) for s in union])*entry/budget
                    loc = pd.Index(union).get_indexer(symbols)
                    c, f, fixed, counts = risks[asof]; covariance = np.zeros((len(union), len(union)))
                    covariance[np.ix_(loc, loc)] = c
                    mu, carry = np.zeros(len(union)), np.zeros(len(union)); mu[loc] = rows.prediction; carry[loc] = f
                    freeze = np.zeros(len(union), bool); freeze[loc] = fixed
                    if case == 'optimizer_zero_alpha': freeze[loc] |= counts < 20
                    if case == 'baseline_equal': w, status = b, 'baseline'
                    else:
                        solution = solve_weights(b, previous, mu, covariance, carry, cfg,
                            use_alpha=case == 'optimizer_trained_alpha', fixed=freeze, cost_rate=.0008*multiplier)
                        if solution.status not in {'optimal','baseline_fallback'}:raise ValueError('unknown validation solver outcome')
                        w, status = b+BLEND*(solution.weights-b), solution.status
                    lots = w*budget/entry
                    for j, s in enumerate(union):
                        decisions.append({'asof':asof,'case':case,'cost_multiplier':multiplier,'settlement_scenario':scenario,
                            'symbol':s,'baseline':float(b[j]),'weight':float(w[j]),'prediction':float(mu[j])})
                    fee = float(np.abs(lots*entry-previous*budget).sum()*.0008*multiplier)
                    exit_day = asof+pd.Timedelta(days=1)
                    exit_prices = prices.loc[exit_day, symbols].to_numpy(float,copy=True)
                    active_lots = lots[loc]
                    retired={};settlement_fee=0.
                    for j,symbol in enumerate(symbols):
                        event=delistings.get((asof,symbol))
                        if event is None:continue
                        adverse_key='lower' if active_lots[j]>=0 else 'upper'
                        key=adverse_key if scenario=='adverse' else ('upper' if adverse_key=='lower' else 'lower')
                        exit_prices[j]=event[key];retired[symbol]=pd.Timestamp(event['settled_at'])
                        settlement_fee+=abs(active_lots[j])*exit_prices[j]*.0008*multiplier
                    if not np.isfinite(exit_prices).all():raise ValueError('unresolved execution or settlement')
                    fee+=settlement_fee
                    price_pnl = float(active_lots@(exit_prices-entry[loc]))
                    # VWAP execution follows the settlement at the minute boundary:
                    # the outgoing position pays at exit, the incoming one does not.
                    times = realized_funding.index
                    mask = (times > asof+pd.Timedelta(minutes=10)) & (times <= exit_day+pd.Timedelta(minutes=10))
                    rates = realized_funding.loc[mask, symbols].fillna(0.)
                    for symbol,settled_at in retired.items():rates.loc[rates.index>settled_at,symbol]=0.
                    marks = closes.reindex(rates.index-pd.Timedelta(hours=1))[symbols].set_axis(rates.index)
                    if (marks.isna() & rates.ne(0)).any().any(): raise ValueError('missing funding mark')
                    funding_pnl = float(-(rates*marks).fillna(0.).sum().to_numpy()@active_lots)
                    before = equity; equity += price_pnl+funding_pnl-fee
                    records.append({'asof': asof, 'case': case, 'cost_multiplier': multiplier, 'settlement_scenario':scenario,
                        'net_return': equity/before-1, 'equity': equity, 'fee': fee/before,
                        'funding_pnl': funding_pnl/before, 'price_pnl': price_pnl/before,
                        'settlement_fee':settlement_fee/before, 'gross_notional': float(np.abs(lots*entry).sum()), 'solver': status})
                    positions = {s: float(q) for s, q in zip(union, lots) if abs(q) > 1e-12 and s not in retired}
                accounts.extend(records)
    all_accounts=pd.DataFrame(accounts)
    all_accounts.to_parquet(out/'settlement_sensitivity.parquet',index=False)
    account=all_accounts.loc[all_accounts.settlement_scenario.eq('adverse')].copy(); account.to_parquet(out/'accounts.parquet', index=False)
    decisions=pd.DataFrame(decisions)
    all_decisions=decisions.copy()
    decisions=decisions.loc[decisions.settlement_scenario.eq('adverse')].copy(); decisions.to_parquet(out/'decisions.parquet',index=False)
    pnl_error = float((all_accounts.net_return-all_accounts.price_pnl-all_accounts.funding_pnl+all_accounts.fee).abs().max())
    active = all_decisions.loc[all_decisions.baseline != 0].copy(); ratio = active.weight/active.baseline
    bound_error = max(float((.875-ratio).clip(lower=0).max()),float((ratio-1.125).clip(lower=0).max()))
    budgets = all_decisions.assign(side=np.sign(all_decisions.baseline)).groupby(['asof','case','cost_multiplier','settlement_scenario','side'])[['baseline','weight']].sum()
    budget_error = float((budgets.baseline-budgets.weight).abs().max())
    results = []
    for (case, multiplier), group in account.groupby(['case', 'cost_multiplier']):
        results.append({'case': case, 'cost_multiplier': multiplier, **metrics(group.net_return)})
    pd.DataFrame(results).to_csv(out/'comparison.csv', index=False)
    primary = account.loc[account.cost_multiplier == 1.].pivot(index='asof', columns='case', values='net_return')
    diff = primary.optimizer_trained_alpha-primary.optimizer_zero_alpha
    interval_sensitivity={}
    for scenario,group in all_accounts.loc[all_accounts.cost_multiplier.eq(1.)].groupby('settlement_scenario'):
        pivot=group.pivot(index='asof',columns='case',values='net_return')
        interval_sensitivity[scenario]={'checks':performance_checks(pivot),
            'incremental_bootstrap':bootstrap_difference(pivot.optimizer_trained_alpha-pivot.optimizer_zero_alpha),
            'metrics':{case:metrics(pivot[case]) for case in cases}}
    checks={'at_least_60_days':len(primary)>=60,
        'both_sides_all_refits':all(r['supported_sides']==[-1,1] for r in refits),
        **performance_checks(primary),
        'settlement_bound_robustness':not delistings or all(all(v['checks'].values()) for v in interval_sensitivity.values()),
        'optimizer_solved_every_day':bool(all_accounts.loc[all_accounts['case'].ne('baseline_equal'),'solver'].eq('optimal').all()),
        'accounting_and_budget_audit':pnl_error<1e-10 and bound_error<1e-7 and budget_error<1e-7,
        'temporal_and_input_checks':not problems}
    report = {'accepted': all(checks.values()), 'checks': checks, 'metrics': results,
        'alpha_policy':alpha_policy(),
        'delisting_events':list(delistings.values()),
        'solver_outcomes':all_accounts.groupby(['settlement_scenario','case','cost_multiplier','solver']).size().rename('days').reset_index().to_dict('records'),'settlement_interval_sensitivity':interval_sensitivity,
        'incremental_bootstrap': bootstrap_difference(diff), 'refits': refits,
        'training_days': 365, 'minimum_side_days': 362, 'test_start': design['test_start'],
        'test_end_inclusive': design['test_end_inclusive'], 'validation_label_available_at': str(days[-1]+pd.Timedelta(days=1,hours=1,minutes=1)),
        'panel_sha256': file_hash(root/'alpha_panel.parquet'), 'design_sha256': file_hash(root/'validation_design.json'),
        'risk_manifest_sha256': file_hash(root/'risk/manifest.json'),
        'published_overlap_audit':json.loads((root/'published_overlap_audit.json').read_text()) if (root/'published_overlap_audit.json').exists() else None,
        'historical_scope':json.loads((root/'historical_scope.json').read_text()) if (root/'historical_scope.json').exists() else None,
        'market_repairs_sha256':file_hash(root/'market_repairs.json') if (root/'market_repairs.json').exists() else None,
        'execution_minutes_manifest_sha256': file_hash(root/'execution_minutes_manifest.json'),
        'entry_benchmark': '00:10–01:00 UTC checksummed official fixed-window VWAP',
        'label_coverage':json.loads((root/'label_coverage.json').read_text()),
        'funding_max_boundary_jitter_ms':float(funding_jitter.max()*1000),
        'accounting_audit': {'maximum_pnl_return_error':pnl_error,'maximum_bound_error':bound_error,'maximum_side_budget_error':budget_error},
        'limitations': json.loads((root/'panel_manifest.json').read_text())['limitations']+[
            'Account simulation uses current 3000 USD budget cap, fractional quantities and preceding hourly funding marks; exchange step sizes/slippage are not reproduced.',
            'Fixed 00:10–01:00 VWAP is a valuation/execution approximation; minute participation, market impact, partial fills and the real sub-10-minute live schedule are not reproduced.',
            'Automatic delisting settlement uses official index OHLC-derived bounds, not an exact settlement price; both endpoints are tested, settlement fees charged, and positions removed at cessation. Missing 24h alpha labels remain excluded.',
            'Positive incremental mean does not by itself establish statistical significance.'],
        'completed_at': datetime.now(timezone.utc).isoformat()}
    write_json(out/'validation.json', report)
    print(json.dumps({'accepted': report['accepted'], 'checks': checks, 'increment': report['incremental_bootstrap']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--market', type=Path, required=True); p.add_argument('--market-tail', type=Path)
    p.add_argument('--settlements', type=Path)
    a = p.parse_args(); execution_inputs(a); evaluate(a)


if __name__ == '__main__': main()
