"""Net-account evaluation, attribution diagnostics and standalone second-round figures."""
from __future__ import annotations

import html
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from .contracts import write_json
from .refinements import EXPERIMENTS, BY_NAME
from .validation import daily_returns, summary, paired_bootstrap, METRIC_DEFINITIONS


DISPLAY = ['B0_baseline', 'B5_full', 'mix_25', 'baseline_band10', 'baseline_band25', 'cost_only_band10', 'wf_sortino']
LABELS = {e.name: e.label for e in EXPERIMENTS} | {
    'wf_sortino': 'Walk-forward: Sortino selection', 'wf_sharpe': 'Walk-forward: Sharpe selection',
    'stress_12bp_baseline': '12bp cost: original', 'stress_12bp_cost_band': '12bp cost: turnover + band'}


def diagnose_allocations(flat, data, out, start, end):
    x = flat.loc[(flat['asof'] >= start) & (flat['asof'] < end)].copy()
    refined = pd.read_parquet(out/'refined_alpha.parquet')
    x = x.merge(refined[['asof', 'symbol', 'alpha_refined']], on=['asof', 'symbol'], validate='one_to_one')
    x['vol_bucket'] = np.ceil(x.groupby(['asof', 'side']).predicted_vol.rank(pct=True)*5).clip(1, 5).astype(int)
    x['directional_price_return'] = x.side*x.future_return
    realized = np.full(len(x), np.nan)
    indices = {t: i for i, t in enumerate(data.times)}
    for asof, ids in x.groupby('asof').groups.items():
        i = indices.get(asof)
        if i is not None and i+24 < len(data.times):
            columns = x.loc[ids, 'column'].to_numpy(int)
            events = data.prices['funding_rate'][i+1:i+25, columns]
            realized[x.index.get_indexer(ids)] = np.sum(events, axis=0)
    x['realized_funding_rate'] = realized
    x['funding_cost_per_unit'] = x.side*x.expected_funding
    x['future_net_per_unit'] = x.directional_price_return-x.side*x.realized_funding_rate
    x['carry_bucket'] = pd.cut(x.funding_cost_per_unit,
        [-np.inf, -.001, -.0001, .0001, .001, np.inf],
        labels=['receive >10bp', 'receive 1-10bp', 'near zero', 'pay 1-10bp', 'pay >10bp'])
    calibration, quantiles = [], []
    for (asof, side), group in x.groupby(['asof', 'side']):
        for feature in ['main_rank', 'small_rank', 'main', 'small', 'alpha', 'alpha_bins', 'alpha_refined']:
            g = group.dropna(subset=[feature, 'future_return'])
            if len(g) > 2:
                ic = g[feature].corr(g.future_return, method='spearman') if g[feature].nunique() > 1 else np.nan
                calibration.append({'asof': asof, 'side': side, 'feature': feature, 'rank_ic_price': ic, 'symbols': len(g)})
            if len(g) >= 3 and g[feature].nunique() > 1:
                bucket = np.ceil(g[feature].rank(pct=True)*3).clip(1, 3).astype(int)
                for k, part in g.groupby(bucket):
                    quantiles.append({'asof': asof, 'side': side, 'feature': feature, 'bucket': int(k),
                        'directional_price_bps': float(part.directional_price_return.mean()*10000),
                        'net_before_trading_cost_bps': float(part.future_net_per_unit.mean()*10000)})
    calibration = pd.DataFrame(calibration)
    calibration.to_csv(out/'pool_rank_ic_daily.csv', index=False)
    ic_summary = calibration.groupby(['side', 'feature']).rank_ic_price.agg(['mean', 'std', 'count']).reset_index()
    ic_summary.to_csv(out/'pool_rank_ic_summary.csv', index=False)
    pd.DataFrame(quantiles).groupby(['side', 'feature', 'bucket']).agg(
        mean_directional_price_bps=('directional_price_bps', 'mean'),
        mean_net_before_trading_cost_bps=('net_before_trading_cost_bps', 'mean'), days=('asof', 'size')).reset_index().to_csv(
            out/'pool_return_buckets.csv', index=False)
    tilt_rows = []
    for name in ['B5_full', 'mix_25', 'funding_pay_mix25', 'no_alpha_mix25',
                 'refined_alpha_mix25', 'cost_only', 'tracking_risk']:
        dec = pd.read_parquet(out/name/'decisions.parquet')
        d = dec.merge(x[['asof', 'symbol', 'side', 'future_return', 'directional_price_return',
                        'vol_bucket', 'carry_bucket']], on=['asof', 'symbol'], validate='one_to_one')
        d['magnitude_tilt'] = (d.w_target-d.w0)*d.side
        d['static_price_delta_bps'] = (d.w_target-d.w0)*d.future_return*10000
        d['action'] = np.select([d.magnitude_tilt > 1e-6, d.magnitude_tilt < -1e-6], ['increase', 'decrease'], 'unchanged')
        d['year'] = d['asof'].dt.year
        for dimension in ['vol_bucket', 'carry_bucket', 'action']:
            groups = d.groupby(['asof', 'side', dimension], observed=True).agg(
                static_price_delta_bps=('static_price_delta_bps', 'sum'),
                magnitude_tilt=('magnitude_tilt', 'sum'),
                directional_price_return=('directional_price_return', 'mean'))
            groups = groups.reset_index()
            groups['year'] = groups['asof'].dt.year
            for year, part in [('all', groups), *list(groups.groupby('year'))]:
                for (side, bucket), g in part.groupby(['side', dimension], observed=True):
                    tilt_rows.append({'model': name, 'year': str(year), 'side': int(side),
                        'dimension': dimension, 'bucket': str(bucket), 'days_present': len(g),
                        'mean_static_price_delta_bps': float(g.static_price_delta_bps.mean()),
                        'mean_magnitude_tilt': float(g.magnitude_tilt.mean()),
                        'mean_directional_price_bps': float(g.directional_price_return.mean()*10000)})
    pd.DataFrame(tilt_rows).to_csv(out/'tilt_diagnostics.csv', index=False)
    carry = x.groupby(['asof', 'side', 'carry_bucket'], observed=True).agg(
        price=('directional_price_return', 'mean'), net=('future_net_per_unit', 'mean'),
        forecast_cost=('funding_cost_per_unit', 'mean'))
    carry.reset_index().groupby(['side', 'carry_bucket'], observed=True).agg(
        price_return=('price', 'mean'), net_before_trade_cost=('net', 'mean'),
        forecast_funding_cost=('forecast_cost', 'mean'), days=('asof', 'size')).reset_index().to_csv(
            out/'funding_price_joint_diagnosis.csv', index=False)
    return ic_summary


def executed_attribution(curves, daily, out):
    rows = []
    base = None
    for name in ['B0_baseline', 'B5_full', 'cost_only_band10', 'wf_sortino', 'wf_sharpe']:
        p = out/name/'pnl_attribution.parquet'
        if not p.exists():
            continue
        a = pd.read_parquet(p); a['date'] = a.time.dt.floor('D')
        a = a.loc[a.date.isin(daily[name].index)]
        a = a.groupby(['date', 'symbol'])[['price_pnl', 'fee', 'funding_cost', 'net_pnl']].sum().reset_index()
        for col in ['price_pnl', 'fee', 'funding_cost', 'net_pnl']:
            a[col] = a[col]/a.date.map(daily[name].before)*10000
        a['year'] = a.date.dt.year
        if name == 'B0_baseline':
            base = a.copy()
        if base is None:
            continue
        combined = a.merge(base, on=['date', 'symbol'], how='outer', suffixes=('', '_base')).fillna(0.)
        combined['year'] = pd.to_datetime(combined.date).dt.year
        for col in ['price_pnl', 'fee', 'funding_cost', 'net_pnl']:
            combined[col+'_delta'] = combined[col]-combined[col+'_base']
        for year, part in [('all', combined), *list(combined.groupby('year'))]:
            for symbol, g in part.groupby('symbol'):
                rows.append({'model': name, 'year': str(year), 'symbol': symbol,
                    **{c+'_sum_bp': float(g[c+'_delta'].sum()) for c in ['price_pnl', 'fee', 'funding_cost', 'net_pnl']}})
    pd.DataFrame(rows).to_csv(out/'executed_coin_attribution.csv', index=False)


def diagnose_risk(out, start, end):
    forecasts = pickle.loads((out/'forecast_cache.pkl').read_bytes())
    records = []
    for name in DISPLAY:
        path = out/name/'decisions.parquet'
        dec = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        groups = {t: g.set_index('symbol') for t, g in dec.groupby('asof')} if len(dec) else {}
        for asof, forecast in forecasts.items():
            if not start <= asof < end:
                continue
            b = forecast.rows.w0.to_numpy()
            w = groups[asof].w_target.reindex(forecast.rows.symbol).to_numpy() if asof in groups else b
            cov = forecast.covariance
            records.append({'model': name, 'asof': asof,
                'baseline_expected_annual_vol': float(np.sqrt(max(0., b@cov@b)*365)),
                'target_expected_annual_vol': float(np.sqrt(max(0., w@cov@w)*365)),
                'expected_annual_tracking_error': float(np.sqrt(max(0., (w-b)@cov@(w-b))*365)),
                'target_btc_beta': float(w@forecast.betas[:, 0]),
                'target_eth_beta': float(w@forecast.betas[:, 1])})
    frame = pd.DataFrame(records)
    frame.to_parquet(out/'risk_budget_daily.parquet', index=False)
    frame.groupby('model').agg(
        mean_baseline_expected_annual_vol=('baseline_expected_annual_vol', 'mean'),
        mean_target_expected_annual_vol=('target_expected_annual_vol', 'mean'),
        mean_expected_annual_tracking_error=('expected_annual_tracking_error', 'mean'),
        mean_target_btc_beta=('target_btc_beta', 'mean'),
        mean_target_eth_beta=('target_eth_beta', 'mean')).reset_index().to_csv(out/'risk_budget_summary.csv', index=False)


def create_refinement_report(curves, config, data, out, flat, splits):
    start, end = pd.Timestamp(config.first_test), data.times[-1].floor('D')
    daily = {n: daily_returns(c).loc[lambda d: (d.index >= start) & (d.index < end)] for n, c in curves.items()}
    base = daily['B0_baseline']; rows, years, sensitivities, paired = [], [], [], {}
    for name, d in daily.items():
        row = {'model': name, 'label': LABELS[name], **summary(d)}
        if not name.startswith('stress_'):
            row['mean_difference_bps'] = float((d.net_return-base.net_return).mean()*10000)
            row['fee_saving_bps'] = float((base.fee_return-d.fee_return).mean()*10000)
            row['funding_improvement_bps'] = float((base.funding_return-d.funding_return).mean()*10000)
            row['price_difference_bps'] = row['mean_difference_bps']-row['fee_saving_bps']-row['funding_improvement_bps']
        evaluated = curves[name].loc[(curves[name].time >= start) & (curves[name].time < end)]
        row['resize_suppressed_symbols_per_hour'] = float(evaluated.resize_suppressed_symbols.mean())
        row['mean_close_gross_leverage'] = float(evaluated.actual_gross_leverage.mean())
        row['max_close_gross_leverage'] = float(evaluated.actual_gross_leverage.max())
        rows.append(row)
        for year, part in d.groupby(d.index.year):
            years.append({'model': name, 'year': int(year), **summary(part)})
        d.reset_index().to_parquet(out/name/'daily_attribution.parquet', index=False)
        if not name.startswith('stress_'):
            paired[name] = {str(block): paired_bootstrap(d, base, config.seed, block=block) for block in [7, 14, 30]}
            for label, dates in [('exclude_UNFI_day', ['2022-06-07']),
                                  ('exclude_2025_10_11', ['2025-10-11'])]:
                sensitivities.append({'model': name, 'scenario': label,
                    **summary(d.loc[~d.index.isin(pd.to_datetime(dates))])})
    table = pd.DataFrame(rows)
    # Put the requested ratios beside return and drawdown, rather than at the end.
    primary = ['model', 'label', 'annual_return', 'sharpe', 'sortino', 'max_drawdown', 'annual_volatility']
    table = table[primary+[c for c in table if c not in primary]]
    table.to_csv(out/'refinement_results.csv', index=False)
    pd.DataFrame(years).to_csv(out/'yearly_results.csv', index=False)
    pd.DataFrame(sensitivities).to_csv(out/'robustness_results.csv', index=False)
    write_json(out/'paired_bootstrap.json', paired)
    write_json(out/'metric_definitions.json', METRIC_DEFINITIONS)
    ic = diagnose_allocations(flat, data, out, start, end)
    executed_attribution(curves, daily, out)
    diagnose_risk(out, start, end)
    selected_counts = {m: pd.Series([s['selected_model'] for s in items]).value_counts().to_dict()
                       for m, items in splits.items()}
    limitations = [
        'Frozen v2/600h historical research; live v3/599h has not been validated.',
        'All history has already been studied; this is retrospective out-of-sample evaluation.',
        'Paired block-bootstrap intervals are pointwise and do not correct for repeated model selection.',
        'Sortino uses net calendar-day returns, MAR=0, all-day downside RMS and 365-day annualization.',
        'Undefined Sortino is null, including periods with no negative returns.',
        'Resize bands allow intraday holdings to drift; daily selection/exit targets always execute and a 5% side-budget guard refreshes targets.',
        'All cost savings and risk changes from bands include the resulting different holding path; gross leverage is not constant between executions.',
        'Richer alpha predicts simple 24h price returns, a first-order approximation to hourly resizing.',
        'Static tilt diagnostics are not executed PnL; executed attribution uses each account own day-start equity.',
        'Funding forecasts use completed historical settlements; older zero-settlement schedules and revised historical factors remain data limitations.',
        'Funding-price bucket analysis is descriptive, without a causal interpretation.',
        'Walk-forward scores come from independently continuous candidate accounts; selected policy executes on its own continuous account.',
        'US-session timing changes and full multi-period cost forecasts are reserved for a separate experiment.',
    ]
    result = {'evaluation_start': str(start.date()), 'evaluation_end': str((end-pd.Timedelta(days=1)).date()),
        'days': len(base), 'models': len(rows), 'results': rows, 'metric_definitions': METRIC_DEFINITIONS,
        'walk_forward_selection_counts': selected_counts, 'limitations': limitations, 'real_orders': False}
    write_json(out/'summary.json', result)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=['Net equity', 'Drawdown', 'Rolling 90-day Sortino (MAR=0)'])
    for name in DISPLAY:
        d = daily[name]; nav = (1+d.net_return).cumprod()
        fig.add_trace(go.Scatter(x=d.index, y=nav, name=LABELS[name]), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=nav/nav.cummax().clip(lower=1)-1,
            name=LABELS[name], showlegend=False), row=2, col=1)
        from .validation import sortino_ratio
        sr = d.net_return.rolling(90).apply(lambda a: np.nan if (s := sortino_ratio(a)) is None else s, raw=True)
        fig.add_trace(go.Scatter(x=d.index, y=sr, name=LABELS[name], showlegend=False), row=3, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_layout(height=1100, template='plotly_white', legend={'orientation': 'h'},
        title='Select Coin: bounded tilts, resize bands and Sortino evaluation')
    maincols = ['model', 'annual_return', 'sharpe', 'sortino', 'max_drawdown', 'annual_volatility',
                'mean_difference_bps', 'price_difference_bps', 'fee_saving_bps', 'funding_improvement_bps',
                'mean_daily_fee_bps', 'mean_daily_funding_bps']
    formatted = table[maincols].copy()
    for c in ['annual_return', 'max_drawdown', 'annual_volatility']:
        formatted[c] = formatted[c].map(lambda v: f'{v:.2%}')
    yearly = pd.DataFrame(years)[['model', 'year', 'cumulative_return', 'sharpe', 'sortino', 'max_drawdown']]
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Optimizer refinements</title>'
    page += '<style>body{font-family:system-ui;margin:28px;color:#202b40}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}th{background:#eef3fa}.scroll{overflow:auto}p{max-width:1100px;line-height:1.7}</style>'
    page += '<h1>Select Coin 仓位优化：第二轮实验</h1>'
    page += f'<p>{start.date()} — {(end-pd.Timedelta(days=1)).date()}，共同评价 {len(base):,} 天。全部结果扣除实际交易费用和资金费；20 个预先固定的实验与对照。每日选币名单及目标多空预算沿用原策略。</p>'
    page += '<p><b>Sortino Ratio</b> = √365 × 日均净收益 ÷ √平均[min(日收益, 0)²]。最低目标收益为 0，分母使用全部交易日；无下行波动时记为缺失。Sharpe 与 Sortino 同时报告，二者对应的滚动选择策略分别独立回放。</p>'
    page += '<div class="scroll">'+formatted.to_html(index=False, float_format=lambda v: f'{v:.3f}', na_rep='—')+'</div>'
    page += fig.to_html(full_html=False, include_plotlyjs=True)
    page += '<h2>池内预测排序能力</h2>'+ic.to_html(index=False, float_format=lambda v: f'{v:.4f}')
    page += '<p>Rank IC 比较预测价格收益与未来价格收益，多空分别计算；它与因子的选币有效性不同。原始因子方向不同，原始因子的 IC 符号不能直接解释为组合收益。</p>'
    page += '<h2>分年表现</h2><p>cumulative_return / max_drawdown 为小数收益率。2022 与 2026 为不完整年份。</p><div class="scroll">'+yearly.to_html(index=False, float_format=lambda v: f'{v:.4f}')+'</div>'
    page += '<h2>研究限制</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in limitations)+'</ul></html>'
    (out/'optimizer_refinements.html').write_text(page)
    plot_comparison(out, daily, table)
    return result


def plot_comparison(out, daily, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), gridspec_kw={'height_ratios': [1.4, 1., 1.]})
    short = ['Original', 'Full optimizer', '25% blend', 'Original + 10% band',
             'Original + 25% band', 'Cost + band', 'WF Sortino']
    colors = ['#202b40', '#c18b2e', '#8593a8', '#865bb4', '#e0784d', '#179269', '#2d63dc']
    for name, label, color in zip(DISPLAY, short, colors):
        d = daily[name]; nav = (1+d.net_return).cumprod()
        axes[0].plot(d.index, nav, label=label, color=color, lw=1.5)
        axes[1].plot(d.index, (nav/nav.cummax().clip(lower=1)-1)*100, color=color, lw=1.)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Net equity (log scale)')
    axes[0].legend(ncol=3, frameon=False)
    axes[1].set_ylabel('Drawdown (%)')
    selected = table.set_index('model').loc[DISPLAY]
    ticks = np.arange(len(DISPLAY))
    axes[2].bar(ticks-.18, selected.sharpe, width=.36, label='Sharpe', color='#8593a8')
    axes[2].bar(ticks+.18, selected.sortino, width=.36, label='Sortino (MAR=0)', color='#2d63dc')
    axes[2].set_xticks(ticks, [s.replace(' + ', '\n+ ') for s in short], fontsize=9)
    axes[2].set_ylabel('Annualized ratio'); axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Select Coin: second-round portfolio refinements', fontsize=19)
    fig.text(.5, .948, f'{daily[DISPLAY[0]].index.min().date()} to {daily[DISPLAY[0]].index.max().date()} | Net fees and funding | Frozen v2/600h', ha='center', fontsize=10)
    fig.text(.05, .02, 'Predeclared experiment menu; previously studied history. Resize bands change the holding path.\nSortino: net daily return / all-day downside RMS, MAR=0, annualized with 365 days.', fontsize=10, color='#555')
    fig.tight_layout(rect=[0, .06, 1, .93])
    fig.savefig(out/'optimizer_refinements.png', dpi=170)
    fig.savefig(out/'optimizer_refinements.svg')
    plt.close(fig)
