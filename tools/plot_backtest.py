#!/usr/bin/env python3
"""Publish one historical baseline figure without publishing its raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backtest'))
from research.optimizer.validation import daily_returns, summary, METRIC_DEFINITIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--equity', required=True, type=Path,
                   help='Audited baseline replay equity_hourly.parquet')
    args = p.parse_args()
    curve = pd.read_parquet(args.equity)
    daily = daily_returns(curve)
    if daily.empty:
        raise ValueError('no complete UTC days')
    nav = (1+daily.net_return).cumprod()
    dd = nav/nav.cummax().clip(lower=1)-1
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2.5, 1]})
    fig.patch.set_facecolor('#f7f9fc')
    for ax in axes:
        ax.set_facecolor('#f7f9fc')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].plot(nav.index, nav, color='#245eea', lw=1.7)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Net equity (start = 1; log scale)')
    axes[0].set_title('Historical v2 baseline | daily selection, hourly rebalancing', loc='left')
    axes[1].fill_between(dd.index, dd*100, 0, color='#245eea', alpha=.25)
    axes[1].plot(dd.index, dd*100, color='#245eea', lw=.8)
    axes[1].set_ylabel('Daily drawdown (%)')
    stats = summary(daily)
    fig.suptitle(f"Crypto Stat Arb | {stats['start']} to {stats['end']}",
                 fontsize=17, fontweight='bold', x=.08, ha='left')
    fig.text(.08, .035,
        f"2x leverage | 8 bp one-way fees | realized funding included | {len(daily):,} complete UTC days\n"
        f"Sharpe {stats['sharpe']:.2f} | Sortino {stats['sortino']:.2f} | daily MDD {stats['max_drawdown']:.1%}"
        " | revised historical inputs; distinct from the live previous-hour v3 signal",
        fontsize=9, color='#506079')
    fig.subplots_adjust(top=.86, bottom=.17, left=.08, right=.98, hspace=.13)
    result = ROOT/'results'
    result.mkdir(exist_ok=True)
    fig.savefig(result/'backtest_equity.png', dpi=160)
    plt.close(fig)
    with args.equity.open('rb') as f:
        source_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    (result/'backtest_summary.json').write_text(json.dumps({
        'strategy': 'historical_top50_v2_qv600_equal_weight_baseline',
        'formula': 'original same-hour quantiles v2; distinct from live previous-hour v3',
        'volume_filter_hours': 600, 'leverage': 2, 'oneway_fee_rate': .0008,
        'funding_included': True, 'optimizer_enabled': False,
        'metric_definitions': METRIC_DEFINITIONS, 'metrics': stats,
        'source_sha256': source_hash,
        'provenance': 'Original baseline replay; baseline hourly accounting parity verified locally.',
        'scope': 'Historical simulation using revised data, not live account returns.'
    }, indent=2)+'\n')
    print(json.dumps({'days': len(daily), 'sharpe': stats['sharpe'],
                      'sortino': stats['sortino'], 'daily_max_drawdown': stats['max_drawdown']}))


if __name__ == '__main__':
    main()
