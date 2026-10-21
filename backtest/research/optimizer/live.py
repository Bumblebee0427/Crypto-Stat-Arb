"""Point-in-time daily live adapter for the fixed three-factor EWMA14/90 model.

Computation only. Orders remain in the existing daily executor. The numeraire
is the strategy cash budget, matching the 2x research allocation convention.
V2 alpha labels are deliberately excluded from the live v3 training sample.
"""
from dataclasses import asdict
from datetime import timedelta

import numpy as np
import pandas as pd

from .alpha import fit_alpha, predict_alpha, alpha_fingerprint
from .contracts import ResearchConfig
from .factor_risk_tuning import RiskSpec, estimate_grid
from .funding import forecast_funding
from .paper_factors import CORE_FACTORS, EXTRA_FACTORS
from .solve import solve_weights

MODEL = 'core_w90_f14_r14_cv_diagonal'
SPEC = RiskSpec(MODEL, tuple(CORE_FACTORS), 90, 14, 14, 'cv_diagonal')
V3 = 'ref_notional_previous_utc_hour_closed_both_v3'
BLEND = .25


def naive_utc(value):
    value = pd.Timestamp(value)
    return value.tz_convert('UTC').tz_localize(None) if value.tzinfo else value


def allocation(snapshot, history, factors, settlements, positions, marks, budget,
               provenance, *, alpha_panel=None):
    asof = naive_utc(snapshot['window_end'])
    if snapshot.get('formula_version') != V3:
        raise ValueError('live adapter requires the current v3 base signal')
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('invalid strategy budget')
    history, factors = history.copy(), factors.copy()
    for frame in [history, factors]:
        frame.index = pd.DatetimeIndex([naive_utc(t) for t in frame.index])
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError('duplicate or unsorted live history')
        if frame.empty or frame.index.max() != asof:
            raise ValueError('daily history must end at the decision boundary')
    if not np.isfinite(factors.loc[asof, CORE_FACTORS+EXTRA_FACTORS].to_numpy(float)).all():
        raise ValueError('latest factor return unavailable')
    cap_date = naive_utc(provenance['capitalization_latest_source_date'])
    if not asof-pd.Timedelta(days=5) <= cap_date <= asof-pd.Timedelta(days=2):
        raise ValueError('capitalization source is stale or violates publication delay')
    if provenance.get('cap_publication_delay_days') != 2:
        raise ValueError('capitalization availability rule changed')
    if naive_utc(provenance['maximum_available_at']) > asof:
        raise ValueError('future observation in live input provenance')
    selected = sorted(s for s, w in snapshot['allocations'].items() if w)
    risk = estimate_grid(history, factors, selected, asof, [SPEC])[MODEL]
    if risk['info'].get('factor_observations', 0) < 60:
        raise ValueError('fewer than 60 complete factor observations in 90-day window')
    if naive_utc(risk['info']['max_observation_at']) != asof:
        raise ValueError('risk model has no latest complete observation')
    cfg = ResearchConfig(risk_days=90)
    rows = pd.DataFrame({'symbol': selected})
    rows['asof'] = asof
    rows['side'] = [1 if snapshot['allocations'][s] > 0 else -1 for s in selected]
    rows['main'] = [snapshot['factors'][s]['main'] for s in selected]
    rows['small'] = [snapshot['factors'][s]['small_long' if sign > 0 else 'small_short']
                     for s, sign in zip(selected, rows.side)]
    for feature in ['main', 'small']:
        rows[feature+'_rank'] = rows.groupby('side')[feature].rank(pct=True)*2-1
    alpha = np.zeros(len(selected)); supported = np.zeros(len(selected), dtype=bool)
    alpha_info = {'status': 'v3_training_warmup_zero_alpha', 'training_days': 0}
    if alpha_panel is not None and len(alpha_panel):
        panel = alpha_panel.loc[alpha_panel.formula_version == V3].copy()
        for col in ['asof', 'label_available_at']:
            panel[col] = pd.to_datetime(panel[col], utc=True).dt.tz_localize(None)
        model = fit_alpha(panel, asof.to_period('M').start_time, cfg)
        alpha, supported = predict_alpha(rows, model)
        alpha_info = {'status': 'v3_trained' if supported.any() else 'v3_training_warmup_zero_alpha',
                      'training_days': model.training_days, 'fitted_at': str(model.fitted_at),
                      'label_cutoff': None if pd.isna(model.label_cutoff) else str(model.label_cutoff),
                      'supported_sides': sorted(model.models), 'model_fingerprint':alpha_fingerprint(model)}
        if set(model.models) != {-1, 1}:
            alpha[:] = 0.; supported[:] = False
            alpha_info['status'] = 'v3_training_warmup_zero_alpha'
    events = settlements.copy()
    events['time'] = pd.to_datetime(events.time, utc=True).dt.tz_localize(None)
    events['event_hour'] = events.time.dt.floor('h')
    events = events.sort_values(['symbol', 'time'])
    funding, funding_info = [], []
    for symbol in selected:
        value, info = forecast_funding(events.loc[events.symbol == symbol], asof, cfg)
        funding.append(value); funding_info.append(info)
    union = sorted(set(selected) | {s for s, q in positions.items() if float(q)})
    loc = {s: i for i, s in enumerate(union)}
    picked = np.array([loc[s] for s in selected])
    n = len(union)
    base = np.array([snapshot['leverage']*snapshot['allocations'].get(s, 0.) for s in union])
    previous = np.array([float(positions.get(s, 0.))*float(marks[s])/budget
                         if float(positions.get(s, 0.)) else 0. for s in union])
    cov = np.zeros((n, n)); cov[np.ix_(picked, picked)] = risk['covariance']
    mu, carry = np.zeros(n), np.zeros(n)
    mu[picked], carry[picked] = alpha, funding
    fixed = np.zeros(n, dtype=bool)
    fixed[picked] = (np.asarray(risk['info']['samples']) < cfg.min_risk_days) & ~supported
    fixed[picked] |= [v['status'].startswith('missing_history') or v.get('stale', False)
                      for v in funding_info]
    solution = solve_weights(base, previous, mu, cov, carry, cfg, fixed=fixed, cost_rate=.0008)
    if solution.status != 'optimal':
        raise ValueError('live solver did not produce an optimal feasible solution')
    weights = base+BLEND*(solution.weights-base)
    # Remove solver noise on forced exits before serializing an actionable artifact.
    weights[base == 0] = 0.
    allocations = {s: float(weights[loc[s]]/snapshot['leverage']) if s in loc else 0.
                   for s in snapshot['allocations']}
    return {'allocations': allocations, 'model': MODEL, 'blend': BLEND,
            'solver_status': solution.status, 'solver_residual': solution.residual,
            'solver_seconds': solution.elapsed_seconds, 'solver_terms': solution.terms,
            'risk': risk['info'], 'alpha_status': alpha_info['status'], 'alpha': alpha_info,
            'funding': dict(zip(selected, funding)),
            'fixed_symbols': [union[i] for i in np.flatnonzero(fixed)],
            'settings': {'risk_spec': asdict(SPEC), 'objective': asdict(cfg),
                         'cash_budget_numeraire': True, 'execution_frequency': 'daily',
                         'hourly_resize_overlay': False},
            'maximum_available_at': asof.isoformat()+'+00:00'}
