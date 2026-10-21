#!/usr/bin/env python3
"""Create a standalone walk-forward validation chart and HTML report."""
import argparse
import html
import json
from pathlib import Path
import sys
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); out = a.out
    report = json.loads((out/'validation.json').read_text())
    if 'metrics' not in report:
        (out/'report.html').write_text('<meta charset="utf-8"><h1>Validation blocked by input checks</h1><pre>'+
            html.escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre>')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    account = pd.read_parquet(out/'accounts.parquet')
    daily = account.loc[account.cost_multiplier == 1.].pivot(index='asof', columns='case', values='net_return')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios':[2, 1]})
    labels = {'baseline_equal':'Original weights', 'optimizer_zero_alpha':'Optimizer: zero alpha',
              'optimizer_trained_alpha':'Optimizer: pretrained v3 alpha'}
    for name in daily:
        axes[0].plot(daily.index, (1+daily[name]).cumprod(), label=labels[name], linewidth=1.6)
    axes[0].legend(); axes[0].set_ylabel('Account NAV'); axes[0].set_title('Monthly walk-forward: fixed settings, fees and funding included')
    diff = daily.optimizer_trained_alpha-daily.optimizer_zero_alpha
    axes[1].plot(diff.index, diff.cumsum()*10000, color='#2563eb')
    axes[1].axhline(0, color='gray', linewidth=.7); axes[1].set_ylabel('Cumulative incremental bp')
    axes[1].set_title('Contribution of learned alpha versus the same optimizer with alpha=0')
    for ax in axes: ax.grid(alpha=.2)
    fig.text(.03, .015, 'Historical v3 replay; 00:10–01:00 UTC official window VWAP; fractional quantities; current $3,000 account budget convention.', fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(out/'walk_forward.png', dpi=170); plt.close(fig)
    comparison = pd.read_csv(out/'comparison.csv')
    for col in ['cumulative_return', 'max_drawdown', 'worst_day']:
        comparison[col] = comparison[col].map(lambda x:f'{x:.2%}')
    for col in ['mean_daily_bps','annualized_sharpe','sortino']:
        comparison[col] = comparison[col].map(lambda x:f'{x:.3f}')
    bootstrap = report['incremental_bootstrap']; ci = bootstrap['mean_95ci_bps']
    body = '<meta charset="utf-8"><title>V3 Alpha Walk-forward</title><style>body{max-width:1200px;margin:35px auto;font:16px system-ui;color:#172033}table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #ccd2dd}img{width:100%}pre{white-space:pre-wrap}</style>'
    body += '<h1>历史 v3 Alpha 预训练与向前验证</h1><p>验证门槛：'+('通过' if report['accepted'] else '未通过，预期 Alpha 不启用')+'</p>'
    body += '<p>训练窗口365天；每月首日重新拟合；只使用当时已成熟标签。验证区间 '+report['test_start']+' 至 '+report['test_end_inclusive']+'。原权重与优化权重按75%/25%混合。</p>'
    body += '<p>加入 Alpha 的平均日增量：%.3f bp；7天分块 Bootstrap 95%% 区间：[%.3f, %.3f] bp。该区间没有进行策略选择修正。</p>'%(bootstrap['mean_daily_increment_bps'],ci[0],ci[1])
    best=diff.idxmax()
    body += '<p>最大增量日：%s，增量 %.3f bp；剔除该日后平均增量 %.4f bp/日。</p>'%(str(best.date()),diff.loc[best]*10000,diff.drop(best).mean()*10000)
    body += '<img src="walk_forward.png">'+comparison.to_html(index=False,escape=True)
    if report.get('published_overlap_audit'):
        body += '<h2>历史重建与已发布选币交叉检查</h2><pre>'+html.escape(json.dumps(report['published_overlap_audit'],ensure_ascii=False,indent=2))+'</pre>'
    body += '<h2>门槛与限制</h2><pre>'+html.escape(json.dumps({'checks':report['checks'],'limitations':report['limitations']},ensure_ascii=False,indent=2))+'</pre>'
    body += '<h2>退市结算敏感性</h2><pre>'+html.escape(json.dumps({'events':report.get('delisting_events',[]),'endpoints':report.get('settlement_interval_sensitivity',{})},ensure_ascii=False,indent=2))+'</pre>'
    body += '<h2>逐月训练记录</h2><pre>'+html.escape(json.dumps(report['refits'],ensure_ascii=False,indent=2))+'</pre>'
    (out/'report.html').write_text(body)


if __name__ == '__main__': main()
