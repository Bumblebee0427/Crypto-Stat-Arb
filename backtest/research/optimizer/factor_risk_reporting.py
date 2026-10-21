"""Separate covariance forecast calibration from trading-strategy performance."""
from __future__ import annotations

import html
import json
import numpy as np
import pandas as pd

from .contracts import write_json
from .validation import daily_returns, summary, paired_bootstrap, METRIC_DEFINITIONS
from .factor_covariance import VARIANTS
from .paper_factors import CORE_FACTORS, EXTRA_FACTORS


def gaussian_covariance_score(covariance, innovations):
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError('nonpositive covariance determinant')
    return float((logdet+innovations@np.linalg.solve(covariance,innovations))/len(innovations))


def risk_calibration(data, forecasts, risks, risk_config, start, end):
    rows = []
    returns = data.daily_returns
    for asof, old in forecasts.items():
        if not start <= asof < end or asof not in risks:
            continue
        symbols = old.rows.symbol.tolist()
        future = asof+pd.Timedelta(days=1)
        if future not in returns.index:
            continue
        history = returns.loc[(returns.index <= asof) &
            (returns.index > asof-pd.Timedelta(days=risk_config.window_days)),symbols]
        # One common past-only mean reference for ALL covariance variants.
        mean = history.mean().to_numpy()
        outcome = returns.loc[future,symbols].to_numpy()
        innovation = outcome-mean
        observed = np.isfinite(innovation)
        if observed.sum()<2:
            continue
        weights = old.rows.w0.to_numpy()
        for name, cov in [('legacy',old.covariance)]+[(n,r['covariance']) for n,r in risks[asof].items()]:
            sub = cov[np.ix_(observed,observed)]; z = innovation[observed]
            port_variance = float(weights@cov@weights)
            error = float(weights@innovation) if observed.all() else np.nan
            realized = float(weights@outcome) if np.isfinite(outcome).all() else np.nan
            info = old.risk_info if name=='legacy' else risks[asof][name]['info']
            rows.append({'asof':asof,'outcome_available_at':future,'model':name,
                'matrix_qlike':gaussian_covariance_score(sub,z),
                'diagonal_qlike':float(np.mean(np.log(np.diag(sub))+z*z/np.diag(sub))),
                'portfolio_variance':port_variance,'portfolio_innovation':error,
                'portfolio_return':realized,
                'portfolio_qlike':float(np.log(port_variance)+error*error/port_variance) if np.isfinite(error) and port_variance>0 else np.nan,
                'variance_ratio':float(error*error/port_variance) if np.isfinite(error) and port_variance>0 else np.nan,
                'normal_5pct_breach':float(error < -1.6448536269514722*np.sqrt(port_variance)) if np.isfinite(error) else np.nan,
                'symbols':len(symbols),'observed_outcomes':int(observed.sum()),
                'short_history_symbols':int(np.sum(np.array(info['samples'])<risk_config.min_observations)),
                'condition':float(np.linalg.cond(cov))})
    return pd.DataFrame(rows)


def paired_mean_interval(a,b,seed=20261020,block=7,draws=1000):
    both=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna()
    delta=(both.a-both.b).to_numpy()
    if len(delta)<block:
        return {'days':len(delta)}
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(draws):
        starts=rng.integers(0,len(delta)-block+1,int(np.ceil(len(delta)/block)))
        ids=np.concatenate([np.arange(s,s+block) for s in starts])[:len(delta)]
        samples.append(float(delta[ids].mean()))
    return {'days':len(delta),'block_days':block,'mean_difference':float(delta.mean()),
        'mean_difference_95ci':np.quantile(samples,[.025,.975]).tolist(),'selection_adjusted':False}


def create_report(curves, config, risk_config, data, forecasts, risks, out, end):
    start=pd.Timestamp(config.first_test)
    daily={n:daily_returns(c).loc[lambda d:(d.index>=start)&(d.index<end)] for n,c in curves.items()}
    results=[];yearly=[];robust=[];paired={}
    for name,d in daily.items():
        result={'model':name,**summary(d)}
        reference='legacy_full' if name.endswith('_full') else 'legacy_mix25_band10'
        if name in {'B0_baseline','baseline_band10'}:
            reference='B0_baseline'
        result['same_overlay_reference']=reference
        result['net_advantage_bps']=float((d.net_return-daily[reference].net_return).mean()*10000)
        result['fee_saving_bps']=float((daily[reference].fee_return-d.fee_return).mean()*10000)
        result['funding_improvement_bps']=float((daily[reference].funding_return-d.funding_return).mean()*10000)
        evaluated=curves[name].loc[(curves[name].time>=start)&(curves[name].time<end)]
        result['mean_close_gross_leverage']=float(evaluated.actual_gross_leverage.mean())
        decision=out/name/'decisions.parquet'
        if decision.exists():
            dec=pd.read_parquet(decision)
            dec=dec.loc[(dec['asof']>=start)&(dec['asof']<end)]
            result['fallback_days']=int(dec.loc[dec.status!='optimal','asof'].nunique())
        else:
            result['fallback_days']=0
        results.append(result)
        for year,g in d.groupby(d.index.year):
            yearly.append({'model':name,'year':int(year),**summary(g)})
        for label,dates in [('exclude_UNFI_day',['2022-06-07']),('exclude_2025_10_11',['2025-10-11'])]:
            robust.append({'model':name,'scenario':label,**summary(d.loc[~d.index.isin(pd.to_datetime(dates))])})
        d.reset_index().to_parquet(out/name/'daily_attribution.parquet',index=False)
        if name not in {'B0_baseline','baseline_band10'}:
            paired[name]={'reference':reference,'blocks':{str(b):paired_bootstrap(d,daily[reference],block=b) for b in [7,14,30]}}
    table=pd.DataFrame(results)
    primary=['model','annual_return','sharpe','sortino','max_drawdown','annual_volatility']
    table=table[primary+[c for c in table if c not in primary]]
    table.to_csv(out/'strategy_results.csv',index=False)
    pd.DataFrame(yearly).to_csv(out/'yearly_results.csv',index=False)
    pd.DataFrame(robust).to_csv(out/'robustness_results.csv',index=False)
    write_json(out/'strategy_paired_bootstrap.json',paired)
    write_json(out/'metric_definitions.json',METRIC_DEFINITIONS)
    calibration=risk_calibration(data,forecasts,risks,risk_config,start,end)
    calibration.to_parquet(out/'covariance_calibration_daily.parquet',index=False)
    quality=calibration.groupby('model').agg(days=('asof','size'),matrix_qlike=('matrix_qlike','mean'),
        diagonal_qlike=('diagonal_qlike','mean'),portfolio_qlike=('portfolio_qlike','mean'),
        variance_ratio=('variance_ratio','mean'),normal_5pct_breach=('normal_5pct_breach','mean'),
        mean_condition=('condition','mean'),mean_short_history_symbols=('short_history_symbols','mean')).reset_index()
    quality.to_csv(out/'covariance_quality.csv',index=False)
    calibration.assign(year=calibration['asof'].dt.year).groupby(['model','year']).agg(
        matrix_qlike=('matrix_qlike','mean'),portfolio_qlike=('portfolio_qlike','mean'),
        variance_ratio=('variance_ratio','mean'),days=('asof','size')).reset_index().to_csv(out/'covariance_quality_yearly.csv',index=False)
    base=calibration.loc[calibration.model=='legacy'].set_index('asof')
    risk_paired={}
    for name,g in calibration.groupby('model'):
        if name=='legacy':continue
        g=g.set_index('asof')
        risk_paired[name]={metric:{str(b):paired_mean_interval(g[metric],base[metric],block=b) for b in [7,14,30]}
                          for metric in ['matrix_qlike','portfolio_qlike']}
    write_json(out/'covariance_score_bootstrap.json',risk_paired)
    factors=pd.read_parquet(out/'factor_returns_daily.parquet').set_index('available_at')
    f=factors.loc[(factors.index>=start)&(factors.index<=end)]
    coverage={'individual':{n:float(f[n].notna().mean()) for n in f},
        'core_complete':float(f[CORE_FACTORS].notna().all(axis=1).mean()),
        'extended_complete':float(f[CORE_FACTORS+EXTRA_FACTORS].notna().all(axis=1).mean())}
    capsource=json.loads((out/'capitalization_sources.json').read_text())
    limitations=[
        'Supplied May-2019 paper definitions adapted to Binance perpetual prices and daily risk estimation; not an exact published-sample replication.',
        'Historical circulating-cap source ends '+capsource['last_source_date']+'; no stale-cap extension to the latest October backtest is reported.',
        'Cap date D is assumed available at D+2. Actual original publication vintages are unavailable; histories may have been revised.',
        'Weekly legs use raw historical circulating-cap weights, not equal weights, and retain formation quantities between weekly formations.',
        'Missing held constituents make an entire factor leg unavailable. Factor regressions use complete observed factor rows without filling missing returns with zero.',
        'CW market returns use the Binance-listed subset with available cap histories; delisted or uncovered tokens can reduce coverage.',
        'EWMA+LW applies classical equal-weight LW intensity to the EWMA covariance; this hybrid has no claim to classical iid weighted-sample optimality.',
        'Beta estimates are equal-weight for every variant; EWMA changes factor covariance, residual variances and residual covariance, not the regression estimator.',
        'Classical LW is applied to factor and mature residual covariance blocks; short-history priors and conservative variance floors remain active.',
        'Risk scores evaluate one-day price innovations around one common trailing mean; the projected portfolio is the original target book, not the hourly resized executed account.',
        'Conditional complete-outcome matrix scores use only observed realized assets for evaluation; allocation eligibility never uses future availability.',
        'Gaussian QLIKE is a covariance scoring rule, not a claim of normally distributed crypto returns; 5% normal breaches are diagnostic only.',
        'All history has been researched previously; bootstrap intervals are pointwise and do not correct for model selection.',
        'Live v3/599h validation and deployment remain outside this historical v2/600h study.',
    ]
    result={'evaluation_start':str(start.date()),'evaluation_end':str((end-pd.Timedelta(days=1)).date()),
        'models':len(results),'risk_variants':len(VARIANTS)+1,'days':len(daily['B0_baseline']),
        'results':results,'covariance_quality':quality.to_dict('records'),'factor_coverage':coverage,
        'limitations':limitations,'real_orders':False}
    write_json(out/'summary.json',result)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    display=['baseline_band10','legacy_mix25_band10','paper3_fixed_mix25_band10',
        'paper3_ewma_mix25_band10','paper3_lw_mix25_band10','paper3_ewma_lw_mix25_band10',
        'extended6_ewma_lw_mix25_band10','asset_lw_mix25_band10']
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
        subplot_titles=['Net equity: same 25% blend / 10% band overlay', 'Drawdown','Original target book: annualized volatility forecasts'])
    for name in display:
        d=daily[name];nav=(1+d.net_return).cumprod()
        fig.add_trace(go.Scatter(x=d.index,y=nav,name=name),row=1,col=1)
        fig.add_trace(go.Scatter(x=d.index,y=nav/nav.cummax().clip(lower=1)-1,name=name,showlegend=False),row=2,col=1)
    for name in ['legacy','paper3_fixed','paper3_ewma_lw','extended6_ewma_lw']:
        part=calibration.loc[calibration.model==name]
        fig.add_trace(go.Scatter(x=part['asof'],y=np.sqrt(part.portfolio_variance*365),name=name+' risk'),row=3,col=1)
    realized=base.portfolio_return.rolling(30).std()*np.sqrt(365)
    fig.add_trace(go.Scatter(x=realized.index,y=realized,name='Realized 30-day price volatility',line={'dash':'dash'}),row=3,col=1)
    fig.update_yaxes(type='log',row=1,col=1);fig.update_layout(height=1200,template='plotly_white')
    formatted=table[['model','annual_return','sharpe','sortino','max_drawdown','mean_daily_fee_bps',
                     'net_advantage_bps','same_overlay_reference','fallback_days']].copy()
    for col in ['annual_return','max_drawdown']:formatted[col]=formatted[col].map(lambda v:f'{v:.2%}')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Crypto factor covariance</title>'
    page+='<style>body{font-family:system-ui;margin:30px;color:#202b40}p{max-width:1150px;line-height:1.7}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}th{background:#eef3fa}.scroll{overflow:auto}</style>'
    page+='<h1>加密货币共同风险因子与协方差估计</h1>'
    page+=f'<p>共同评价期 {start.date()} — {(end-pd.Timedelta(days=1)).date()}，共 {len(daily["B0_baseline"]):,} 天。比较 {len(VARIANTS)+1} 种风险估计、{len(results)} 个独立账户。所有策略结果扣除 8bp 单边综合成本和实际资金费。</p>'
    page+='<p>依照所提供论文的 CMKT（市值加权市场）、CSMB（小市值减大市值）、CMOM（三周赢家减输家）构建周度组合，并计算日频因子收益。扩展版另加 Amihud、波动率和偏度。协方差为 BΩBᵀ＋Ψ；EWMA 半衰期固定 30 天，经典 LW 与 EWMA/LW 混合方法分别列示。</p>'
    page+='<p><b>数据覆盖：</b>流通市值源止于 '+html.escape(capsource['last_source_date'])+'；不会将旧市值外推至十月。共同区间核心三因子完整日比例 '+f'{coverage["core_complete"]:.1%}，扩展六因子完整日比例 {coverage["extended_complete"]:.1%}。缺失因子日不填零。</p>'
    page+='<h2>协方差预测质量</h2><p>matrix_qlike / portfolio_qlike 越低越好；variance_ratio 为平均“实际价格创新平方 / 预测方差”，理想校准接近 1。矩阵分数与组合分数分别评价资产整体和原策略目标权重方向上的风险。</p>'
    page+='<div class="scroll">'+quality.to_html(index=False,float_format=lambda v:f'{v:.4f}')+'</div>'
    page+='<h2>账户结果：Sharpe 与 Sortino</h2><p>Sortino 使用 MAR=0、全部 UTC 日的下行平方均值，按 365 日年化。full 与 full 对比；mix25_band10 与同样执行规则对比。原权重＋band10 是独立执行对照。</p><div class="scroll">'+formatted.to_html(index=False,float_format=lambda v:f'{v:.4f}')+'</div>'
    page+=fig.to_html(full_html=False,include_plotlyjs=True)
    page+='<h2>定义与来源</h2><p><a href="https://www.nber.org/papers/w25882">Liu–Tsyvinski–Wu, Common Risk Factors in Cryptocurrency</a>；<a href="https://scikit-learn.org/1.7/modules/generated/sklearn.covariance.LedoitWolf.html">Ledoit–Wolf 实现与定义</a>；<a href="https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a">RiskMetrics 指数加权参考</a>。</p>'
    page+='<h2>研究限制</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in limitations)+'</ul></html>'
    (out/'factor_covariance_report.html').write_text(page)
    plot_comparison(out,table,quality,daily)
    return result


def plot_comparison(out,table,quality,daily):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    display=['baseline_band10','legacy_mix25_band10','paper3_fixed_mix25_band10',
        'paper3_ewma_lw_mix25_band10','extended6_ewma_lw_mix25_band10','asset_lw_mix25_band10']
    labels=['Original + band','BTC/ETH','Paper 3 factors','Paper 3 + EWMA/LW','Extended 6 + EWMA/LW','Direct asset LW']
    colors=['#202b40','#9b8f7f','#bf8a32','#2e64dc','#179269','#865bb4']
    fig,axes=plt.subplots(3,1,figsize=(14,13),gridspec_kw={'height_ratios':[1.4,1.,1.]})
    for name,label,color in zip(display,labels,colors):
        d=daily[name];nav=(1+d.net_return).cumprod()
        axes[0].plot(d.index,nav,label=label,color=color,lw=1.5)
        axes[1].plot(d.index,(nav/nav.cummax().clip(lower=1)-1)*100,color=color,lw=1.)
    axes[0].set_yscale('log');axes[0].set_ylabel('Net equity (log)');axes[0].legend(ncol=3,frameon=False)
    axes[1].set_ylabel('Drawdown (%)')
    selected=table.set_index('model').loc[display];ticks=np.arange(len(display))
    axes[2].bar(ticks-.18,selected.sharpe,.36,label='Sharpe',color='#8593a8')
    axes[2].bar(ticks+.18,selected.sortino,.36,label='Sortino (MAR=0)',color='#2e64dc')
    axes[2].set_xticks(ticks,[s.replace(' + ','\n+ ') for s in labels],fontsize=9)
    axes[2].set_ylabel('Annualized ratio');axes[2].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Crypto factor covariance: daily risk and portfolio allocation',fontsize=18)
    first,last=daily[display[0]].index.min().date(),daily[display[0]].index.max().date()
    fig.text(.5,.948,f'{first} to {last} | Frozen v2/600h | Net costs and funding | Common capitalization coverage',ha='center',fontsize=10)
    fig.text(.05,.02,'Optimizer curves share 25% allocation blend and 10% resize band; original + band is the execution control.\nHistorical cap publication delay is assumed. EWMA/LW is a documented hybrid. Previously studied history.',fontsize=9,color='#555')
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(out/'factor_covariance_comparison.png',dpi=170)
    fig.savefig(out/'factor_covariance_comparison.svg');plt.close(fig)
