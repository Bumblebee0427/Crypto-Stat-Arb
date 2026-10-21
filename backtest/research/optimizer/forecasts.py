"""Shared point-in-time forecasts, never shared account states."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .alpha import fit_alpha, predict_alpha
from .funding import forecast_funding, load_settlements
from .risk import estimate_risk
from .contracts import write_json


@dataclass
class DayForecast:
    rows: pd.DataFrame
    alpha: np.ndarray
    alpha_bins: np.ndarray
    alpha_supported: np.ndarray
    covariance: np.ndarray
    risk_info: dict
    betas: np.ndarray
    funding: np.ndarray
    funding_info: list
    fitted_at: pd.Timestamp
    label_cutoff: pd.Timestamp | None


def build_forecasts(data, config, out):
    events = load_settlements(data, out/'prepared')
    event_groups = {s: g for s, g in events.groupby('symbol', sort=False)}
    forecasts, flat, models = {}, [], []
    model, month = None, None
    for index, (asof, rows) in enumerate(data.panel.groupby('asof', sort=True)):
        rows = rows.reset_index(drop=True)
        current_month = asof.to_period('M').start_time
        if current_month != month:
            model = fit_alpha(data.panel, current_month, config); month = current_month
            models.append({'fitted_at': model.fitted_at, 'label_cutoff': model.label_cutoff,
                'training_days': model.training_days,
                'models': {s: m.to_dict() for s, m in model.models.items()}})
        alpha, supported = predict_alpha(rows, model)
        bins, _ = predict_alpha(rows, model, method='bins')
        history = data.daily_history(asof, config.risk_days)
        covariance, info, betas = estimate_risk(history, rows.symbol.tolist(), config)
        funding, funding_info = [], []
        for s in rows.symbol:
            group = event_groups.get(s, events.iloc[:0])
            value, details = forecast_funding(group, asof, config)
            funding.append(value); funding_info.append(details)
        forecasts[asof] = DayForecast(rows, alpha, bins, supported, covariance, info, betas,
            np.array(funding), funding_info, model.fitted_at, model.label_cutoff)
        for j, r in rows.iterrows():
            flat.append({**r.to_dict(), 'alpha': alpha[j], 'alpha_bins': bins[j],
                'alpha_supported': bool(supported[j]), 'predicted_vol': float(np.sqrt(covariance[j, j])),
                'risk_samples': info['samples'][j], 'covariance_min_eigenvalue': info['min_eigenvalue'],
                'btc_beta': float(betas[j, 0]), 'eth_beta': float(betas[j, 1]),
                'expected_funding': funding[j], 'funding_status': funding_info[j]['status'],
                'funding_source': funding_info[j]['source'],
                'funding_available_at': funding_info[j]['available_at'],
                'funding_events': json.dumps(funding_info[j]['events']),
                'fitted_at': model.fitted_at, 'training_label_cutoff': model.label_cutoff})
        if index % 250 == 0:
            print('forecast day', str(asof), index, flush=True)
    pd.DataFrame(flat).to_parquet(out/'forecasts.parquet', index=False)
    write_json(out/'alpha_models.json', models)
    return forecasts
