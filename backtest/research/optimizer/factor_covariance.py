"""Daily covariance forecasts from priced factors, EWMA and Ledoit–Wolf.

The EWMA+LW case is an explicitly labelled hybrid: use classical equal-weight
LW intensity to regularize the EWMA covariance. It is NOT claimed to be the
closed-form iid-optimal Ledoit–Wolf estimator for weighted observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from .paper_factors import CORE_FACTORS, EXTRA_FACTORS


VARIANTS = {
    'paper3_diag': ('paper3', 'diag'),
    'paper3_fixed': ('paper3', 'fixed'),
    'paper3_ewma': ('paper3', 'ewma_fixed'),
    'paper3_lw': ('paper3', 'lw'),
    'paper3_ewma_lw': ('paper3', 'ewma_lw'),
    'extended6_fixed': ('extended6', 'fixed'),
    'extended6_ewma': ('extended6', 'ewma_fixed'),
    'extended6_lw': ('extended6', 'lw'),
    'extended6_ewma_lw': ('extended6', 'ewma_lw'),
    'asset_lw': ('asset', 'lw'),
}


def covariance_weights(dates, asof, half_life=None):
    dates = pd.DatetimeIndex(dates); asof = pd.Timestamp(asof)
    if (dates > asof).any():
        raise ValueError('future observation in covariance estimation')
    if half_life is None:
        return np.full(len(dates), 1/len(dates))
    if half_life <= 0:
        raise ValueError('nonpositive EWMA half-life')
    age = (asof-dates).total_seconds().to_numpy()/86400.
    weights = np.exp2(-age/half_life)
    return weights/weights.sum()


def weighted_covariance(values, weights):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('covariance requires complete observed rows')
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(values) or (weights < 0).any() or not np.isclose(weights.sum(), 1.):
        raise ValueError('invalid covariance weights')
    centered = values-np.sum(values*weights[:, None], axis=0)
    return centered.T@(centered*weights[:, None])


def estimate_covariance_block(values, dates, asof, method, config, fixed_shrink=0.):
    equal = covariance_weights(dates, asof)
    temporal = method.startswith('ewma')
    weights = covariance_weights(dates, asof, config.ewma_half_life_days) if temporal else equal
    sample = weighted_covariance(values, weights)
    shrink = 0.
    if method in {'lw', 'ewma_lw'}:
        classical, shrink = ledoit_wolf(np.asarray(values, dtype=float), assume_centered=False)
        if method == 'lw':
            cov = classical
        else:
            target = np.eye(sample.shape[0])*np.trace(sample)/sample.shape[0]
            cov = (1-shrink)*sample+shrink*target
    else:
        cov = (1-fixed_shrink)*sample+fixed_shrink*np.diag(np.diag(sample))
        shrink = fixed_shrink
    return cov, {'shrinkage': float(shrink), 'effective_observations': float(1/np.sum(weights**2)),
        'time_weighting': 'EWMA' if temporal else 'equal',
        'target': 'trace_mean_identity' if method in {'lw', 'ewma_lw'} else 'diagonal',
        'lw_intensity_calibration': 'equal_weight_classical' if method in {'lw', 'ewma_lw'} else None,
        'weighted_lw_optimality_claimed': False}


def factor_regression(history, factors, symbols, config, asof):
    history = history.loc[(history.index <= asof) &
                          (history.index > asof-pd.Timedelta(days=config.window_days))]
    factors = factors.reindex(history.index)
    complete_factors = np.isfinite(factors.to_numpy(float)).all(axis=1)
    f = factors.to_numpy(float)
    returns = history.reindex(columns=symbols).to_numpy(float)
    prior = np.r_[1., np.zeros(f.shape[1]-1)]
    betas = np.tile(prior, (len(symbols), 1))
    residuals = np.full_like(returns, np.nan)
    counts = np.zeros(len(symbols), dtype=int)
    if complete_factors.sum() < config.min_observations:
        return None
    for j in range(len(symbols)):
        mask = complete_factors & np.isfinite(returns[:, j])
        counts[j] = int(mask.sum())
        if counts[j] < config.min_observations:
            continue
        x, y = f[mask], returns[mask, j]
        center_x, center_y = x.mean(axis=0), y.mean()
        x, y = x-center_x, y-center_y
        scale = max(float(np.trace(x.T@x))/len(x), 1e-8)
        beta = np.linalg.solve(x.T@x+np.eye(x.shape[1])*scale*config.ridge_observations, x.T@y)
        strength = counts[j]/(counts[j]+config.beta_prior_observations)
        betas[j] = strength*beta+(1-strength)*prior
        intercept = center_y-center_x@betas[j]
        residuals[mask, j] = returns[mask, j]-intercept-f[mask]@betas[j]
    return {'history': history, 'factors': factors, 'complete': complete_factors,
        'betas': betas, 'residuals': residuals, 'counts': counts}


def _positive_definite(covariance):
    cov = (covariance+covariance.T)/2
    if not np.isfinite(cov).all():
        raise ValueError('nonfinite covariance')
    eigen = np.linalg.eigvalsh(cov)
    if eigen.min() < -1e-9:
        raise ValueError('covariance is not PSD')
    cov += np.eye(len(cov))*max(0., 1e-8-eigen.min())
    return cov


def covariance_from_fit(fit, factor_names, asof, method, config, symbols):
    n, k = len(symbols), len(factor_names)
    if fit is None:
        return {'covariance': np.eye(n)*config.short_history_variance,
            'betas': np.zeros((n, k)), 'info': {'samples': [0]*n, 'factor_names': factor_names,
                'fallback': 'insufficient_factor_history', 'factor_observations': 0,
                'max_observation_at': None, 'condition': 1., 'min_eigenvalue': config.short_history_variance,
                'mature_residual_symbols': 0}}
    history, factors = fit['history'], fit['factors']
    observed = fit['complete']; counts = fit['counts']; residuals = fit['residuals']
    dates = history.index
    omega, omega_info = estimate_covariance_block(factors.to_numpy(float)[observed], dates[observed],
        asof, method, config, fixed_shrink=.05 if method in {'fixed', 'ewma_fixed'} else 0.)
    variances = np.full(n, np.nan)
    for j in range(n):
        mask = np.isfinite(residuals[:, j])
        if mask.sum() >= config.min_observations:
            w = covariance_weights(dates[mask], asof,
                config.ewma_half_life_days if method.startswith('ewma') else None)
            variances[j] = weighted_covariance(residuals[mask, j:j+1], w)[0, 0]
    prior = max(float(np.nanmedian(variances)) if np.isfinite(variances).any() else config.short_history_variance, .02**2)
    for j in range(n):
        if np.isfinite(variances[j]):
            strength = counts[j]/(counts[j]+config.beta_prior_observations)
            variances[j] = max(strength*variances[j]+(1-strength)*prior, config.variance_floor)
        else:
            variances[j] = max(prior*2, config.short_history_variance)
    idio = np.diag(variances)
    mature = np.flatnonzero(counts >= max(60, int(observed.sum()*.9)))
    common = np.isfinite(residuals[:, mature]).all(axis=1) if len(mature) else np.zeros(len(dates), bool)
    residual_info = None
    if method != 'diag' and len(mature) >= 2 and common.sum() >= 60:
        block, residual_info = estimate_covariance_block(residuals[common][:, mature], dates[common],
            asof, method, config)
        if method in {'fixed', 'ewma_fixed'}:
            block = .5*block+.5*np.diag(variances[mature])
            residual_info['shrinkage'] = .5
        idio[np.ix_(mature, mature)] = block
    else:
        mature = np.array([], dtype=int)
    # Preserve conservative per-coin idiosyncratic floors after block shrinkage.
    uplift = np.maximum(variances-np.diag(idio), 0.)
    idio += np.diag(uplift)
    cov = _positive_definite(fit['betas']@omega@fit['betas'].T+idio)
    return {'covariance': cov, 'betas': fit['betas'], 'info': {
        'samples': counts.tolist(), 'factor_names': list(factor_names),
        'factor_observations': int(observed.sum()), 'max_observation_at': str(dates[observed].max()),
        'min_eigenvalue': float(np.linalg.eigvalsh(cov).min()), 'condition': float(np.linalg.cond(cov)),
        'mature_residual_symbols': len(mature), 'factor_covariance': omega_info,
        'residual_covariance': residual_info, 'idio_floor_uplift': float(uplift.sum()),
        'regression_time_weighting': 'equal; loadings held identical across covariance-weighting variants',
        'fallback': 'conservative_prior_for_short_history'}}


def direct_asset_lw(history, symbols, asof, config):
    frame = history.loc[(history.index <= asof) & (history.index > asof-pd.Timedelta(days=config.window_days))].reindex(columns=symbols)
    values = frame.to_numpy(float); counts = np.isfinite(values).sum(axis=0)
    variances = np.full(len(symbols), config.short_history_variance)
    for j in range(len(symbols)):
        valid = np.isfinite(values[:, j])
        if valid.sum() >= config.min_observations:
            variances[j] = max(float(np.var(values[valid, j])), config.variance_floor)
    cov = np.diag(variances)
    mature = np.flatnonzero(counts >= max(60, int(len(frame)*.9)))
    common = np.isfinite(values[:, mature]).all(axis=1) if len(mature) else np.zeros(len(frame), bool)
    shrink = None
    if len(mature) >= 2 and common.sum() >= 60:
        block, shrink = ledoit_wolf(values[common][:, mature])
        cov[np.ix_(mature, mature)] = block
    cov += np.diag(np.maximum(variances-np.diag(cov), 0.))
    cov = _positive_definite(cov)
    return {'covariance': cov, 'betas': np.zeros((len(symbols), 0)), 'info': {
        'samples': counts.tolist(), 'factor_names': [], 'factor_observations': 0,
        'max_observation_at': str(frame.index.max()), 'min_eigenvalue': float(np.linalg.eigvalsh(cov).min()),
        'condition': float(np.linalg.cond(cov)), 'shrinkage': None if shrink is None else float(shrink),
        'mature_residual_symbols': len(mature), 'fallback': 'diagonal_for_short_history'}}


def estimate_variants(history, factor_returns, symbols, asof, config):
    asof = pd.Timestamp(asof)
    fits = {family: factor_regression(history, factor_returns[names], symbols, config, asof)
            for family, names in [('paper3', CORE_FACTORS), ('extended6', CORE_FACTORS+EXTRA_FACTORS)]}
    results = {}
    for name, (family, method) in VARIANTS.items():
        if family == 'asset':
            results[name] = direct_asset_lw(history, symbols, asof, config)
        else:
            names = CORE_FACTORS if family == 'paper3' else CORE_FACTORS+EXTRA_FACTORS
            results[name] = covariance_from_fit(fits[family], names, asof, method, config, symbols)
    return results
