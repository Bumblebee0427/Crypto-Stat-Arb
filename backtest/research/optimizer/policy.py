from dataclasses import replace
import json
import numpy as np
import pandas as pd

from .contracts import digest
from .solve import solve_weights


SPECS = {
    'B1_inverse_vol': {'inverse_vol': True, 'use_alpha': False, 'use_cost': False, 'use_funding': False},
    'B2_variance': {'use_alpha': False, 'use_cost': False, 'use_funding': False},
    'B3_alpha_risk': {'use_cost': False, 'use_funding': False},
    'B4_alpha_risk_cost': {'use_funding': False},
    'B5_full': {},
    'full_no_alpha': {'use_alpha': False},
    'full_diagonal': {'diagonal': True},
    'full_no_cost': {'use_cost': False},
    'full_no_regularization': {'no_regularization': True},
    'full_bins': {},
}


def make_policy(data, forecasts, config, name, schedule=None, *, overrides=None, blend=1.,
                alpha_overrides=None, warmup_days=None):
    if not 0 <= blend <= 1:
        raise ValueError('blend must lie between zero and one')
    warm_until = data.panel['asof'].min()+pd.Timedelta(
        days=config.train_days if warmup_days is None else warmup_days)
    def decide(i, asof, baseline, state):
        if asof < warm_until or asof not in forecasts:
            return baseline, []
        forecast = forecasts[asof]
        picked = forecast.rows.column.to_numpy(dtype=int)
        close = data.prices['close'][i]
        # last_prices is known at decision; a missing close cannot erase residual exposure.
        mark = np.where(np.isfinite(close), close, state.last_prices)
        current = state.lots*data.steps*mark/state.equity
        union = np.union1d(picked, np.flatnonzero(current))
        mapping = {int(j): k for k, j in enumerate(union)}
        selected = np.array([mapping[int(j)] for j in picked])
        n = len(union)
        covariance = np.zeros((n, n)); covariance[np.ix_(selected, selected)] = forecast.covariance
        alpha, funding = np.zeros(n), np.zeros(n)
        alpha[selected] = forecast.alpha_bins if name == 'full_bins' else forecast.alpha
        if alpha_overrides is not None:
            alpha[selected] = alpha_overrides[asof]
        funding[selected] = forecast.funding
        betas = np.zeros((n, 2)); betas[selected] = forecast.betas
        fixed = np.zeros(n, bool)
        fixed[selected] = (np.asarray(forecast.risk_info['samples']) < config.min_risk_days) & ~forecast.alpha_supported
        spec = {**SPECS.get(name, {}), **(overrides or {})}
        if spec.get('use_funding', True):
            # An unobserved funding history must not become a cheap carry opportunity.
            fixed[selected] |= np.array([info['status'].startswith('missing_history')
                                         for info in forecast.funding_info])
        gamma = config.gamma
        if schedule is not None:
            key = max((d for d in schedule if d <= asof), default=None)
            if key is not None:
                gamma = schedule[key]
        cfg = replace(config, gamma=gamma)
        solution = solve_weights(baseline[union], current[union], alpha, covariance, funding, cfg,
            fixed=fixed, cost_rate=data.fee_rate*config.cost_multiplier, **spec)
        raw_weights = solution.weights.copy()
        applied = raw_weights.copy() if blend == 1. else baseline[union]+blend*(raw_weights-baseline[union])
        weights = np.zeros_like(baseline); weights[union] = applied
        # Bounded numerical errors must never initiate a spurious out-of-pool position.
        weights[baseline == 0] = 0.
        variance_contribution = applied*(covariance@applied)
        records = []
        input_hash = digest({'asof': asof, 'b': baseline[union].tolist(), 'pre': current[union].tolist(),
            'alpha': alpha.tolist(), 'funding': funding.tolist(), 'cov': covariance.tolist(),
            'gamma': gamma, 'label_cutoff': forecast.label_cutoff})
        for k, j in enumerate(union):
            records.append({'asof': asof, 'symbol': data.symbols[j], 'w0': baseline[j],
                'w_pre': current[j], 'w_target': weights[j], 'alpha': alpha[k], 'funding': funding[k],
                'w_solver': float(raw_weights[k]), 'blend': blend,
                'funding_mode': spec.get('funding_mode', 'signed'),
                'predicted_vol': float(np.sqrt(covariance[k, k])), 'fixed': bool(fixed[k]),
                'btc_beta': float(betas[k, 0]), 'eth_beta': float(betas[k, 1]),
                'variance_contribution': float(variance_contribution[k]),
                'gamma': gamma, 'status': solution.status, 'residual': solution.residual,
                'fallback_reason': solution.fallback_reason, 'solve_seconds': solution.elapsed_seconds,
                'expected_turnover': abs(weights[j]-current[j]), 'input_hash': input_hash,
                'terms': json.dumps({'solver_terms': solution.terms,
                    'applied_variance': float(applied@covariance@applied),
                    'applied_funding_cost': float(funding@applied),
                    'applied_expected_alpha': float(alpha@applied),
                    'applied_turnover': float(np.abs(applied-current[union]).sum())}),
                'model_fitted_at': forecast.fitted_at,
                'training_label_cutoff': forecast.label_cutoff})
        return weights, records
    return decide
