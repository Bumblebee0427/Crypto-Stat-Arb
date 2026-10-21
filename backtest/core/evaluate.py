import itertools

import numpy as np
import pandas as pd


def _max_consecutive(mask) -> int:
    values = [len(list(v)) for k, v in itertools.groupby(mask) if k]
    return max(values) if values else 0


def _infer_periods_per_year(metric_equity: pd.DataFrame, ret_series: pd.Series, eval_period: str | None) -> float:
    if eval_period in {'1D', 'D', 'DAY', 'DAILY'}:
        return 365.0

    ts = pd.to_datetime(metric_equity['candle_begin_time'], errors='coerce')
    delta = ts.diff().dropna().median()
    if pd.notna(delta) and delta > pd.Timedelta(0):
        return float(pd.Timedelta(days=365) / delta)

    total_days = (ts.max() - ts.min()) / pd.Timedelta(days=1)
    return (len(ret_series) / total_days) * 365 if total_days > 0 else np.nan


def _build_daily_metric_equity(equity: pd.DataFrame, net_col: str, pct_col: str) -> pd.DataFrame:
    temp = equity.copy()
    temp['candle_begin_time'] = pd.to_datetime(temp['candle_begin_time'], errors='coerce')
    temp = temp[temp['candle_begin_time'].notna()].sort_values('candle_begin_time')
    temp.set_index('candle_begin_time', inplace=True)

    daily = pd.DataFrame(index=temp.resample('D').last().index)
    daily[net_col] = pd.to_numeric(temp[net_col], errors='coerce').resample('D').last()
    daily[pct_col] = daily[net_col].pct_change()

    if 'equity' in temp.columns:
        daily['equity'] = pd.to_numeric(temp['equity'], errors='coerce').resample('D').last()
    if '是否爆仓' in temp.columns:
        daily['是否爆仓'] = pd.to_numeric(temp['是否爆仓'], errors='coerce').resample('D').max()
    else:
        daily['是否爆仓'] = 0

    if {'turnover', 'equity'}.issubset(temp.columns):
        turnover_ratio = (
            pd.to_numeric(temp['turnover'], errors='coerce')
            / pd.to_numeric(temp['equity'], errors='coerce').replace(0, np.nan)
        )
        daily['_turnover_ratio'] = turnover_ratio.replace([np.inf, -np.inf], np.nan).resample('D').sum(min_count=1)

    daily.dropna(subset=[net_col], inplace=True)
    daily.reset_index(inplace=True)
    return daily.rename(columns={'index': 'candle_begin_time'})


def _add_drawdown_columns(equity: pd.DataFrame, net_col: str, prefix: str) -> None:
    net_value = pd.to_numeric(equity[net_col], errors='coerce')
    equity[f'{prefix}max2here'] = net_value.expanding().max()
    equity[f'{prefix}dd2here'] = net_value / equity[f'{prefix}max2here'] - 1


# 计算策略评价指标
def strategy_evaluate(equity, net_col='多空资金曲线', pct_col='本周期多空涨跌幅', eval_period: str | None = None):
    """
    回测评价函数
    :param equity: 资金曲线数据
    :param net_col: 资金曲线列名
    :param pct_col: 周期涨跌幅列名
    :param eval_period: 评价周期。1D 时将小时资金曲线聚合为日频后统计 Sharpe/胜率/Turnover 等周期指标。
    :return:
    """
    prefix = net_col.split('资金曲线')[0]
    _add_drawdown_columns(equity, net_col, prefix)

    equity = equity.copy()
    equity['candle_begin_time'] = pd.to_datetime(equity['candle_begin_time'], errors='coerce')
    equity = equity[equity['candle_begin_time'].notna()].sort_values('candle_begin_time').reset_index(drop=True)

    eval_key = str(eval_period).upper() if eval_period else None
    metric_equity = _build_daily_metric_equity(equity, net_col, pct_col) if eval_key in {'1D', 'D', 'DAY', 'DAILY'} else equity.copy()

    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # 将数字转为百分数
    def num_to_pct(value):
        return '%.2f%%' % (value * 100) if pd.notna(value) else np.nan

    # ===计算累积净值
    results.loc[0, '累积净值'] = round(metric_equity[net_col].iloc[-1], 2)

    # ===计算年化收益
    elapsed = metric_equity['candle_begin_time'].iloc[-1] - metric_equity['candle_begin_time'].iloc[0]
    elapsed_days = elapsed / pd.Timedelta(days=1)
    annual_return = (metric_equity[net_col].iloc[-1]) ** (365 / elapsed_days) - 1 if elapsed_days > 0 else np.nan
    results.loc[0, '年化收益'] = num_to_pct(annual_return)

    # ===计算最大回撤
    _add_drawdown_columns(metric_equity, net_col, prefix)
    end_date, max_draw_down = tuple(metric_equity.sort_values(by=[f'{prefix}dd2here']).iloc[0][['candle_begin_time', f'{prefix}dd2here']])
    start_date = metric_equity[metric_equity['candle_begin_time'] <= end_date].sort_values(by=net_col, ascending=False).iloc[0]['candle_begin_time']
    results.loc[0, '最大回撤'] = num_to_pct(max_draw_down)
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2) if abs(max_draw_down) > 1e-12 else np.nan

    mean_back_zf = 1 / (1 + metric_equity[f'{prefix}dd2here']) - 1
    mean_fix_zf = mean_back_zf.mean()
    max_fix_zf = 1 / (1 + max_draw_down) - 1
    results.loc[0, '修复涨幅（均/最大）'] = f"{num_to_pct(mean_fix_zf)} / {num_to_pct(max_fix_zf)}"
    if pd.notna(annual_return) and annual_return > -1 and annual_return != 0:
        results.loc[0, '修复时间（均/最大）'] = (
            f"{round(np.log10(1 + mean_fix_zf) / np.log10(1 + annual_return) * 365, 1)} / "
            f"{round(np.log10(1 + max_fix_zf) / np.log10(1 + annual_return) * 365, 1)}"
        )
    else:
        results.loc[0, '修复时间（均/最大）'] = np.nan

    # ===周期收益类指标：1D 回测使用日频收益，其他回测保持原始频率
    ret_series = pd.to_numeric(metric_equity[pct_col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    win_returns = ret_series[ret_series > 0]
    loss_returns = ret_series[ret_series <= 0]

    results.loc[0, '盈利周期数'] = float(len(win_returns))
    results.loc[0, '亏损周期数'] = float(len(loss_returns))
    results.loc[0, '胜率'] = num_to_pct(len(win_returns) / len(ret_series)) if len(ret_series) > 0 else np.nan
    results.loc[0, '每周期平均收益'] = num_to_pct(ret_series.mean()) if len(ret_series) > 0 else np.nan
    if len(win_returns) > 0 and len(loss_returns) > 0 and abs(loss_returns.mean()) > 1e-12:
        results.loc[0, '盈亏收益比'] = round(win_returns.mean() / loss_returns.mean() * (-1), 2)
    else:
        results.loc[0, '盈亏收益比'] = np.nan
    if '是否爆仓' in metric_equity.columns and 1 in metric_equity['是否爆仓'].to_list():
        results.loc[0, '盈亏收益比'] = 0
    results.loc[0, '单周期最大盈利'] = num_to_pct(ret_series.max()) if len(ret_series) > 0 else np.nan
    results.loc[0, '单周期大亏损'] = num_to_pct(ret_series.min()) if len(ret_series) > 0 else np.nan

    results.loc[0, '最大连续盈利周期数'] = float(_max_consecutive(ret_series > 0))
    results.loc[0, '最大连续亏损周期数'] = float(_max_consecutive(ret_series <= 0))

    # ===其他评价指标
    results.loc[0, '收益率标准差'] = num_to_pct(ret_series.std()) if len(ret_series) > 1 else np.nan

    # ===Sharpe Ratio（年化，rf=0）
    sharpe = np.nan
    if len(ret_series) > 1:
        std_ret = ret_series.std(ddof=1)
        if std_ret > 1e-12:
            periods_per_year = _infer_periods_per_year(metric_equity, ret_series, eval_key)
            if np.isfinite(periods_per_year) and periods_per_year > 0:
                sharpe = ret_series.mean() / std_ret * np.sqrt(periods_per_year)
    results.loc[0, 'Sharpe Ratio'] = round(float(sharpe), 4) if np.isfinite(sharpe) else np.nan

    # ===平均Turnover：1D 回测为日 turnover，其他保持原始周期 turnover
    if eval_key in {'1D', 'D', 'DAY', 'DAILY'} and '_turnover_ratio' in metric_equity.columns:
        turnover_ratio = pd.to_numeric(metric_equity['_turnover_ratio'], errors='coerce')
    elif 'turnover' in metric_equity.columns:
        turnover_ratio = (
            pd.to_numeric(metric_equity['turnover'], errors='coerce')
            / pd.to_numeric(metric_equity.get('equity'), errors='coerce').replace(0, np.nan)
        )
    else:
        turnover_ratio = pd.Series(dtype='float64')
    turnover_ratio = turnover_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    avg_turnover = turnover_ratio.mean() if len(turnover_ratio) > 0 else np.nan
    results.loc[0, '平均Turnover'] = num_to_pct(avg_turnover) if np.isfinite(avg_turnover) else np.nan

    # ===交易成本：手续费与资金费
    if 'fee' in equity.columns:
        fee_sum = pd.to_numeric(equity['fee'], errors='coerce').fillna(0).sum()
        results.loc[0, '总手续费'] = f'{fee_sum:,.2f}'

    if 'funding_fee' in equity.columns:
        funding_sum = pd.to_numeric(equity['funding_fee'], errors='coerce').fillna(0).sum()
        results.loc[0, '总资金费'] = f'{funding_sum:,.2f}'

        initial_capital = np.nan
        if {'equity', net_col}.issubset(equity.columns):
            equity_num = pd.to_numeric(equity['equity'], errors='coerce')
            net_num = pd.to_numeric(equity[net_col], errors='coerce').replace(0, np.nan)
            initial_values = (equity_num / net_num).replace([np.inf, -np.inf], np.nan).dropna()
            if len(initial_values) > 0:
                initial_capital = initial_values.iloc[0]

        if np.isfinite(initial_capital) and abs(initial_capital) > 1e-12:
            results.loc[0, '累计资金费率'] = num_to_pct(funding_sum / initial_capital)

    # ===每年、每月收益率
    temp = metric_equity.copy()
    temp.set_index('candle_begin_time', inplace=True)
    year_return = temp[[pct_col]].resample(rule='YE').apply(lambda x: (1 + x.dropna()).prod() - 1)
    month_return = temp[[pct_col]].resample(rule='ME').apply(lambda x: (1 + x.dropna()).prod() - 1)
    quarter_return = temp[[pct_col]].resample(rule='QE').apply(lambda x: (1 + x.dropna()).prod() - 1)

    def num2pct(x):
        if str(x) != 'nan':
            return str(round(x * 100, 2)) + '%'
        else:
            return x

    year_return['涨跌幅'] = year_return[pct_col].apply(num2pct)
    month_return['涨跌幅'] = month_return[pct_col].apply(num2pct)
    quarter_return['涨跌幅'] = quarter_return[pct_col].apply(num2pct)

    return results.T, year_return, month_return, quarter_return
