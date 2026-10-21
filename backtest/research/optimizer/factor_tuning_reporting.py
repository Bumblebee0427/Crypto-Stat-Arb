"""Offline, reviewable tables and figures for declared factor-risk tuning."""
import html
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from .paper_factors import CORE_FACTORS, EXTRA_FACTORS
from .contracts import write_json
from .validation import daily_returns


def label(spec):
    extras=[s for s in spec.factors if s not in CORE_FACTORS]
    return '3F'+('+'+'+'.join(s.replace('CILLIQ10','ILLIQ').replace('CVOL30','VOL').replace('CSKEW7','SKEW') for s in extras) if extras else '')


def render_report(out,specs,end,start):
    table=pd.read_csv(out/'strategy_results.csv').set_index('model')
    mature=pd.read_csv(out/'strategy_results_mature_alpha.csv').set_index('model')
    quality=pd.read_csv(out/'covariance_quality.csv').set_index('model')
    yearly=pd.read_csv(out/'yearly_results.csv')
    paired=json.loads((out/'paired_bootstrap.json').read_text())
    rows=[]
    for s in specs:
        rows.append({'model':s.name,'factors':label(s),'window':s.window,
            'factor_half_life':s.factor_half_life,'residual_half_life':s.residual_half_life,'shrinkage':s.shrinkage,
            **table.loc[s.name].to_dict(),**{k:quality.loc[s.name,k] for k in ['matrix_qlike','portfolio_qlike','variance_ratio']}})
    combined=pd.DataFrame(rows);combined.to_csv(out/'experiment_comparison.csv',index=False)
    ablation=combined.loc[(combined.window==180)&(
        ((combined.shrinkage=='lw_identity')&combined.factor_half_life.isna())|
        ((combined.shrinkage=='cv_diagonal')&(combined.factor_half_life==30)&(combined.residual_half_life==30)))].copy()
    ablation.to_csv(out/'factor_ablation_results.csv',index=False)
    decay=combined.loc[combined.factors.isin(['3F','3F+ILLIQ+VOL+SKEW'])&(combined.shrinkage=='cv_diagonal')]
    decay.to_csv(out/'decay_grid_results.csv',index=False)
    targets=combined.loc[combined.factors.isin(['3F','3F+ILLIQ+VOL+SKEW'])&(combined.window==180)&
        ((combined.factor_half_life.isna())|((combined.factor_half_life==30)&(combined.residual_half_life==30)))]
    targets.to_csv(out/'shrinkage_target_results.csv',index=False)
    winner=combined.sort_values('sortino',ascending=False).iloc[0]
    loss_winner=combined.sort_values('portfolio_qlike').iloc[0]
    baseline='legacy_mix25_band10'
    insights={'retrospective_best_sortino':winner['model'],'retrospective_best_portfolio_qlike':loss_winner['model'],
        'legacy_sortino':float(table.loc[baseline,'sortino']),
        'rolling_all_sortino':float(table.loc['rolling_all','sortino']),
        'rolling_all_delta_daily_bps':paired['rolling_all']['7']['mean_difference_bps'],
        'rolling_all_delta_95ci_daily_bps':paired['rolling_all']['7']['mean_difference_95ci_bps'],
        'multiple_comparisons_adjusted':False}
    write_json(out/'findings.json',insights)
    figures=[]
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Factor ablation: net Sortino','Factor ablation: portfolio QLIKE (lower is better)'])
    for method in ['lw_identity','cv_diagonal']:
        g=ablation.loc[ablation.shrinkage==method]
        fig.add_trace(go.Bar(x=g.factors,y=g.sortino,name=method,legendgroup=method),row=1,col=1)
        fig.add_trace(go.Bar(x=g.factors,y=g.portfolio_qlike,name=method,legendgroup=method,showlegend=False),row=1,col=2)
    fig.update_layout(height=500,title='All eight combinations of Amihud / volatility / skewness',barmode='group')
    figures.append(fig)
    fig=make_subplots(rows=2,cols=2,subplot_titles=['3F: net Sortino','6F: net Sortino','3F: portfolio QLIKE','6F: portfolio QLIKE'])
    for col,family in enumerate(['3F','3F+ILLIQ+VOL+SKEW'],1):
        g=decay.loc[(decay.factors==family)&(decay.factor_half_life==decay.residual_half_life)]
        for row,metric in [(1,'sortino'),(2,'portfolio_qlike')]:
            matrix=g.pivot(index='window',columns='factor_half_life',values=metric).reindex(index=[90,180,365],columns=[14,30,60])
            fig.add_trace(go.Heatmap(z=matrix.values,x=['14d','30d','60d'],y=['90d','180d','365d'],
                text=matrix.round(3).values,texttemplate='%{text}',colorscale='Blues',showscale=False),row=row,col=col)
    fig.update_layout(height=650,title='History windows and matched factor / residual decay')
    figures.append(fig)
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Separate decay: 3F','Separate decay: 6F'])
    for col,family in enumerate(['3F','3F+ILLIQ+VOL+SKEW'],1):
        g=decay.loc[(decay.factors==family)&(decay.window==180)]
        matrix=g.pivot(index='residual_half_life',columns='factor_half_life',values='sortino').reindex(index=[14,30,60],columns=[14,30,60])
        fig.add_trace(go.Heatmap(z=matrix.values,x=['factor14d','factor30d','factor60d'],y=['residual14d','residual30d','residual60d'],
            text=matrix.round(3).values,texttemplate='%{text}',colorscale='Blues',showscale=False),row=1,col=col)
    fig.update_layout(height=440,title='Separate factor / residual half-lives: net Sortino; blank = not tested')
    figures.append(fig)
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Shrinkage targets: Sortino','Shrinkage targets: portfolio QLIKE'])
    for family in ['3F','3F+ILLIQ+VOL+SKEW']:
        g=targets.loc[targets.factors==family].copy()
        g['setting']=g.shrinkage+np.where(g.factor_half_life.isna(),' / equal',' / EWMA30')
        for col,metric in [(1,'sortino'),(2,'portfolio_qlike')]:
            fig.add_trace(go.Bar(x=g.setting,y=g[metric],name=family,legendgroup=family,showlegend=col==1),row=1,col=col)
    fig.update_layout(height=560,barmode='group',title='Standard LW, standardized LW, and chronologically validated shrinkage')
    figures.append(fig)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['Net NAV: common capitalization period','Drawdown'],vertical_spacing=.12)
    selected=['baseline_band10',baseline,'core_w180_f0_r0_lw_identity','rolling_all','rolling_core','rolling_six']
    for name in selected:
        d=daily_returns(pd.read_parquet(out/'accounts'/name/'equity_hourly.parquet'))
        d=d.loc[(d.index>=start)&(d.index<end)];nav=(1+d.net_return).cumprod()
        fig.add_trace(go.Scatter(x=nav.index,y=nav,name=name,legendgroup=name),row=1,col=1)
        fig.add_trace(go.Scatter(x=nav.index,y=nav/nav.cummax().clip(lower=1)-1,name=name,legendgroup=name,showlegend=False),row=2,col=1)
    fig.update_yaxes(type='log',row=1,col=1);fig.update_layout(height=750,title='Fixed controls and monthly past-only model selection')
    figures.append(fig)
    fig=go.Figure()
    for name in selected:
        g=yearly.loc[yearly.model==name]
        fig.add_trace(go.Bar(x=g.year.astype(str),y=g.sortino,name=name))
    fig.update_layout(height=480,barmode='group',title='Year-by-year Sortino; 2021 includes learned-alpha cold start, 2026 ends April14')
    figures.append(fig)
    for i,fig in enumerate(figures):
        fig.write_html(out/f'figure_{i+1}.html',include_plotlyjs=True)
    # Standard plotting produces standalone publication/export assets.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for method,offset,color in [('lw_identity',-.18,'#2563eb'),('cv_diagonal',.18,'#16a085')]:
        g=ablation.loc[ablation.shrinkage==method].reset_index(drop=True);x=np.arange(len(g))
        axes[0].bar(x+offset,g.sortino,width=.36,label=method,color=color)
        axes[1].bar(x+offset,g.portfolio_qlike,width=.36,label=method,color=color)
    for ax in axes:ax.set_xticks(x,g.factors,rotation=35,ha='right');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    axes[0].set_title('Net Sortino');axes[1].set_title('Portfolio QLIKE (lower is better)')
    fig.suptitle('Factor ablation | 2021-01-02 to '+str((end-pd.Timedelta(days=1)).date()))
    fig.tight_layout();fig.savefig(out/'factor_ablation.png',dpi=180);fig.savefig(out/'factor_ablation.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,family in zip(axes,['3F','3F+ILLIQ+VOL+SKEW']):
        for window,g in decay.loc[(decay.factors==family)&(decay.factor_half_life==decay.residual_half_life)].groupby('window'):
            g=g.sort_values('factor_half_life');ax.plot(g.factor_half_life,g.sortino,marker='o',label=f'{window}d window')
        ax.set_title(family);ax.set_xlabel('Factor and residual half-life (days)');ax.set_ylabel('Net Sortino');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'decay_tuning.png',dpi=180);fig.savefig(out/'decay_tuning.svg');plt.close(fig)
    def formatted(frame):
        frame=frame.copy()
        for col in ['annual_return','max_drawdown']:
            if col in frame:frame[col]=frame[col].map(lambda x:f'{x:.2%}')
        for col in ['sharpe','sortino','portfolio_qlike','variance_ratio']:
            if col in frame:frame[col]=frame[col].map(lambda x:f'{x:.4f}')
        return frame.to_html(index=False,escape=True)
    content=''.join(fig.to_html(full_html=False,include_plotlyjs=False) for fig in figures)
    head=f'''<!doctype html><meta charset="utf-8"><title>Factor risk tuning 2021–2026</title>
    <style>body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:32px;color:#172338}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border:1px solid #dce3ee}}.note{{background:#eef3fb;padding:18px;line-height:1.7}}h1,h2{{color:#173b6c}}.table{{overflow-x:auto}}</style>
    <script>{get_plotlyjs()}</script><h1>因子风险模型：增删实验、EWMA 衰减和收缩目标</h1>
    <p>主要评估：{start.date()}—{(end-pd.Timedelta(days=1)).date()}；{int(table.iloc[0]['days'])} 个完整交易日；52 个风险配置、58 个独立账户。</p>
    <div class="note">所有账户使用同一原始选币、方向、多空预算、alpha、资金费预测、2 倍初始目标杠杆及单边 8bp 成本。Optimizer 混合比例 25%，日内调整阈值 10%。<br>
    2021 年允许风险和资金费参与分配；学习型 alpha 只在过去训练样本充分后启用，之前为 0。另附训练充分阶段的独立统计。<br>
    历史流通市值截止 2026-04-13；主要模型比较截止 2026-04-14。2026-04-15 至 10-07 的补充回放统一切换 legacy BTC/ETH 风险，不能解读为论文因子的后续验证。<br>
    论文周度因子按日估计风险；Amihud10、volatility30、skewness7 的信号窗口固定，本轮调整风险估计窗口和衰减。历史修订数据不等同于当时在线版本。<br>
    因子增删统一使用六个因子均有观测的历史行；避免新增因子同时改变缺失值样本而混淆贡献。不同窗口自然使用不同长短的过去历史。<br>
    原选币池含部分股票、商品代理；本轮保留原池，未新增相应专门风险因子。XAU 市值映射及年末周度定义沿用原研究并见 manifest。<br>
    等权 LW 使用 sklearn 原公式；标准化 LW 先估相关矩阵再还原各币方差。EWMA 收缩在当时历史中前 80% 拟合、后 20% 验证强度，不宣称是解析加权 LW 最优解。残差先由当前过去窗口回归得到，因此内部验证不是完整端到端交叉拟合；月度外层选择只用已经实现的未来一天误差。<br>
    所有固定配置全样本排名均为回顾性比较。滚动选择每月只看此前 180 个日历天、至少 60 个共同完整观测的原始目标组合 QLIKE；不按未来交易收益挑参数。95% bootstrap 区间未校正多重比较，不是未接触过的最终留出样本。</div>'''
    controls=table.loc[selected,['annual_return','sharpe','sortino','max_drawdown']].reset_index()
    retrospective=combined.sort_values('sortino',ascending=False).head(12)
    text=f'''<h2>固定控制与因果滚动选择</h2><div class="table">{formatted(controls)}</div>
    <p>全样本 Sortino 最高的固定配置：{html.escape(str(winner['model']))}。它属于回顾性结果。组合风险预测损失最低：{html.escape(str(loss_winner['model']))}。</p>
    <p>rolling_all 相对 legacy 同一执行规则，平均净增益 {insights['rolling_all_delta_daily_bps']:.3f} bp/天；7 天分块 bootstrap 95% 区间 {insights['rolling_all_delta_95ci_daily_bps']}。</p>
    <h2>因子增删：相同风险配置下比较</h2><div class="table">{formatted(ablation[['factors','shrinkage','annual_return','sharpe','sortino','max_drawdown','portfolio_qlike','variance_ratio']])}</div>
    <h2>固定配置回顾性前 12 名</h2><div class="table">{formatted(retrospective[['model','annual_return','sharpe','sortino','max_drawdown','portfolio_qlike']])}</div>
    <h2>训练充分阶段（2022-01-02 起）</h2><div class="table">{formatted(mature.loc[selected,['annual_return','sharpe','sortino','max_drawdown']].reset_index())}</div>
    <h2>全部配置</h2><div class="table">{formatted(combined)}</div>
    <p>复核文件：experiment_definition.json、experiment_comparison.csv、factor_ablation_results.csv、decay_grid_results.csv、shrinkage_target_results.csv、yearly_results.csv、robustness_results.csv、paired_bootstrap.json、rolling_schedules.json、verification.json。</p>'''
    (out/'factor_risk_tuning_report.html').write_text(head+content+text)
