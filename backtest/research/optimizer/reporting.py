from __future__ import annotations

from pathlib import Path
import html
import numpy as np
import pandas as pd

from .contracts import write_json
from .validation import daily_returns, summary, paired_bootstrap


def create_report(curves, config, data, out, splits):
    start = pd.Timestamp(config.first_test)
    # Last calendar day lacks the following execution/funding boundary.
    end = data.times[-1].floor('D')
    daily = {name: daily_returns(curve) for name, curve in curves.items()}
    daily = {name: d.loc[(d.index >= start) & (d.index < end)] for name, d in daily.items()}
    base = daily['B0_baseline']
    rows, yearly, sensitivity = [], [], []
    for name, d in daily.items():
        row = {'model': name, **summary(d)}
        if name not in {'B0_baseline'} and not name.startswith('stress_'):
            row['mean_difference_bps'] = paired_bootstrap(d, base, draws=300)['mean_difference_bps']
        rows.append(row)
        for year, part in d.groupby(d.index.year):
            yearly.append({'model': name, 'year': int(year), **summary(part)})
        if name in ['B0_baseline', 'B5_full', 'walk_forward']:
            for label, dates in [('exclude_UNFI_day', ['2022-06-07']),
                                  ('exclude_2025_10_11', ['2025-10-11'])]:
                part = d.loc[~d.index.isin(pd.to_datetime(dates))]
                sensitivity.append({'model': name, 'scenario': label, **summary(part)})
            attribution = out/name/'pnl_attribution.parquet'
            if attribution.exists():
                unfi = pd.read_parquet(attribution, filters=[('symbol', '==', 'UNFI-USDT')])
                contribution = unfi.groupby(unfi.time.dt.floor('D')).net_pnl.sum()
                part = d.copy()
                part['net_return'] -= contribution.reindex(part.index, fill_value=0)/part.before
                sensitivity.append({'model': name, 'scenario': 'exclude_UNFI_coin_frozen_trades', **summary(part)})
        d.assign(model=name).reset_index().to_parquet(out/name/'daily_attribution.parquet', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out/'ablation_results.csv', index=False)
    pd.DataFrame(yearly).to_csv(out/'yearly_results.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out/'robustness_results.csv', index=False)
    paired = {name: {str(block): paired_bootstrap(daily[name], base, config.seed, block=block)
                    for block in [7, 14, 30]} for name in ['B5_full', 'walk_forward'] if name in daily}
    write_json(out/'paired_bootstrap.json', paired)
    write_json(out/'walk_forward_splits.json', splits)
    forecasts = pd.read_parquet(out/'forecasts.parquet')
    forecasts = forecasts.loc[(forecasts['asof'] >= start) & (forecasts['asof'] < end)]
    calibration = []
    for (asof, side), group in forecasts.groupby(['asof', 'side']):
        valid = group.dropna(subset=['future_return', 'alpha'])
        if len(valid) > 2:
            for prediction in ['alpha', 'alpha_bins']:
                ic = float(valid[prediction].corr(valid.future_return, method='spearman')) if valid[prediction].nunique() > 1 else None
                calibration.append({'asof': asof, 'side': int(side), 'prediction': prediction,
                    'rank_ic': ic, 'symbols': len(valid)})
    calibration_frame = pd.DataFrame(calibration)
    calibration_frame.to_csv(out/'alpha_calibration.csv', index=False)
    diagnostics = {}
    for name in curves:
        p = out/name/'decisions.parquet'
        if p.exists():
            dec = pd.read_parquet(p)
            dec = dec.loc[(dec['asof'] >= start) & (dec['asof'] < end)]
            if len(dec):
                groups = dec.groupby('asof')
                diagnostics[name] = {'decisions': int(groups.ngroups),
                    'fallback_days': int(dec.loc[dec.status != 'optimal', 'asof'].nunique()),
                    'max_constraint_residual': float(dec.residual.max()),
                    'mean_solve_seconds': float(groups.solve_seconds.first().mean()),
                    'mean_absolute_weight_tilt': float((dec.w_target-dec.w0).abs().mean()),
                    'max_abs_weight': float(dec.w_target.abs().max())}
                diagnostics[name]['mean_target_btc_beta'] = float(
                    (dec.w_target*dec.btc_beta).groupby(dec['asof']).sum().mean())
                diagnostics[name]['mean_target_eth_beta'] = float(
                    (dec.w_target*dec.eth_beta).groupby(dec['asof']).sum().mean())
    write_json(out/'solver_diagnostics.json', diagnostics)
    manifest = {'evaluation_start': str(start), 'evaluation_end_exclusive': str(end),
        'results': rows, 'paired_tests': paired,
        'alpha_rankic': {str(k): (None if pd.isna(v) else float(v)) for k, v in
            calibration_frame.groupby(['prediction', 'side']).rank_ic.mean().items()},
        'funding_sources': forecasts.funding_source.value_counts().to_dict(),
        'funding_schedule_status': forecasts.funding_status.value_counts().to_dict(),
        'solver': diagnostics,
        'limitations': ['v2/600h historical research; not a v3 live validation',
            'previously studied history is not an untouched strategy holdout',
            'legacy source funding pivot cannot distinguish zero settlement from no event; older schedules inferred',
            'point-in-time publication latency and original NaN/delist handling are compatibility assumptions',
            'objective predicts immediate turnover only; replay charges every hourly resize',
            'covariance and one-day alpha are first-order approximations to hourly rebalancing',
            'bootstrap intervals do not correct for multiple model selection'], 'real_orders': False}
    manifest['limitations'].append('UNFI coin removal is fixed-path contribution sensitivity, not an executable portfolio rerun')
    write_json(out/'summary.json', manifest)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=['Net equity on common evaluation dates', 'Drawdown', 'Rolling 30-day mean net advantage (bp/day)'])
    for name in ['B0_baseline', 'B1_inverse_vol', 'B5_full', 'walk_forward']:
        if name not in daily:
            continue
        d = daily[name]; nav = (1+d.net_return).cumprod()
        fig.add_trace(go.Scatter(x=d.index, y=nav, name=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=nav/nav.cummax()-1, name=name, showlegend=False), row=2, col=1)
        advantage = (d.net_return-base.net_return).rolling(30).mean()*10000
        fig.add_trace(go.Scatter(x=d.index, y=advantage, name=name, showlegend=False), row=3, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_layout(height=1000, title='Select Coin portfolio optimizer research', template='plotly_white')
    formatted = table.copy()
    for col in ['annual_return', 'annual_volatility', 'max_drawdown', 'cvar_95']:
        if col in formatted:
            formatted[col] = formatted[col].map(lambda v: f'{v:.2%}' if pd.notna(v) else '')
    description = ('<p>固定原选币及逐日预算，保留小时再平衡。全部结果扣除实际交易成本和资金费。'
        '主结果为回溯样本外研究；walk_forward 仅使用之前 90 日验证段选择风险偏好，持仓连续递推。</p>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Select Coin optimizer</title>'
    page += '<style>body{font-family:system-ui;margin:32px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}.table{overflow:auto}</style>'
    page += '<h1>Select Coin 仓位优化研究</h1>'+description
    page += '<p>评价期 '+html.escape(str(start.date()))+' 至 '+html.escape(str((end-pd.Timedelta(days=1)).date()))+'</p>'
    page += '<div class="table">'+formatted.to_html(index=False, float_format=lambda v: f'{v:.4f}')+'</div>'
    page += fig.to_html(full_html=False, include_plotlyjs=True)
    page += '<h2>研究限制</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in manifest['limitations'])+'</ul></html>'
    (out/'optimizer_report.html').write_text(page)
    return manifest
