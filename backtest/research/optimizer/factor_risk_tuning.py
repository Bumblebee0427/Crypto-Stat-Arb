"""Declared factor ablations, separate risk decay and causal shrinkage targets.

CV shrinkage is explicitly not the classical Ledoit--Wolf formula. It chooses
intensity on the final chronological fifth of the available estimation window,
then refits the covariance using all observations known at the decision.
"""
from dataclasses import dataclass, asdict, replace
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from .paper_factors import PaperRiskConfig, CORE_FACTORS, EXTRA_FACTORS
from .factor_covariance import (covariance_weights, weighted_covariance,
    factor_regression, _positive_definite)


@dataclass(frozen=True)
class RiskSpec:
    name: str
    factors: tuple
    window: int = 180
    factor_half_life: float | None = None
    residual_half_life: float | None = None
    shrinkage: str = 'lw_identity'
    group: str = 'ablation'


def experiment_specs():
    specs = {}
    def add(factors, window=180, fh=None, rh=None, shrink='lw_identity', group='ablation'):
        family = 'core' if not factors else 'core_'+'_'.join(factors)
        name = f'{family}_w{window}_f{fh or 0}_r{rh or 0}_{shrink}'
        if name not in specs:
            specs[name] = RiskSpec(name, tuple(CORE_FACTORS+list(factors)), window, fh, rh, shrink, group)
    # All 2^3 extensions: singleton additions and six-factor leave-one-out.
    for k in range(4):
        for extras in combinations(EXTRA_FACTORS, k):
            add(extras)
            add(extras, fh=30, rh=30, shrink='cv_diagonal')
    # Window/decay comparisons in both three- and six-factor families.
    for extras in [[], EXTRA_FACTORS]:
        for window in [90, 180, 365]:
            for half in [14, 30, 60]:
                add(extras, window, half, half, 'cv_diagonal', 'decay')
        for fh, rh in [(14, 60), (60, 14), (30, 60), (60, 30)]:
            add(extras, 180, fh, rh, 'cv_diagonal', 'decay')
        for half in [None, 30]:
            for shrink in ['cv_identity', 'cv_diagonal', 'cv_constant_correlation']:
                add(extras, 180, half, half, shrink, 'target')
        add(extras, shrink='lw_correlation', group='target')
    return list(specs.values())


def target_matrix(sample, target):
    p = len(sample)
    if target == 'identity':
        return np.eye(p)*np.trace(sample)/p
    if target == 'diagonal':
        return np.diag(np.diag(sample))
    if target != 'constant_correlation':
        raise ValueError('unknown shrinkage target')
    vol = np.sqrt(np.maximum(np.diag(sample), 1e-16))
    corr = sample/np.outer(vol, vol)
    rho = (corr.sum()-np.trace(corr))/(p*(p-1)) if p > 1 else 0.
    rho = np.clip(rho, -1/max(p-1, 1)+1e-8, 1-1e-8)
    return np.outer(vol, vol)*((1-rho)*np.eye(p)+rho*np.ones((p,p)))


def covariance_block(values, dates, asof, half_life, method):
    x = np.asarray(values, float)
    if len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('need complete historical observations')
    dates = pd.DatetimeIndex(dates)
    w = covariance_weights(dates, asof, half_life)
    sample = weighted_covariance(x, w)
    info = {'half_life': half_life, 'effective_observations': float(1/(w@w)),
        'method': method, 'max_observation_at': str(dates.max()), 'weighted_lw_optimality_claimed': False}
    if method.startswith('lw_'):
        if half_life is not None:
            raise ValueError('classical LW requires equal weights; use CV for EWMA')
        if method == 'lw_identity':
            cov, delta = ledoit_wolf(x)
        elif method == 'lw_correlation':
            vol = np.sqrt(np.maximum(np.diag(sample), 1e-16))
            corr, delta = ledoit_wolf((x-x.mean(axis=0))/vol)
            cov = corr*np.outer(vol, vol)
        else:
            raise ValueError('unknown LW estimator')
        info.update(shrinkage=float(delta), target='identity' if method=='lw_identity' else 'diagonal_via_standardized_LW')
        return cov, info
    target = method.removeprefix('cv_')
    cut = max(3, int(len(x)*.8))
    cut = min(cut, len(x)-1)
    train, valid = x[:cut], x[cut:]
    train_w = covariance_weights(dates[:cut], dates[cut-1], half_life)
    train_cov = weighted_covariance(train, train_w)
    train_target = target_matrix(train_cov, target)
    innovations = valid-np.sum(train*train_w[:,None], axis=0)
    realized = innovations.T@innovations/len(innovations)
    best, best_loss = 1., np.inf
    for delta in [0., .1, .25, .5, .75, 1.]:
        cov = (1-delta)*train_cov+delta*train_target
        cov = cov+np.eye(len(cov))*1e-10
        sign, logdet = np.linalg.slogdet(cov)
        loss = (logdet+np.trace(np.linalg.solve(cov, realized)))/len(cov) if sign > 0 else np.inf
        if loss < best_loss:
            best, best_loss = delta, loss
    info.update(shrinkage=best, target=target, cv_train_last=str(dates[cut-1]),
        cv_validation_first=str(dates[cut]), cv_validation_last=str(dates[-1]),
        cv_validation_observations=len(valid), cv_qlike=float(best_loss))
    return (1-best)*sample+best*target_matrix(sample, target), info


def fit_spec(history, factor_returns, symbols, asof, spec, fit=None):
    cfg = PaperRiskConfig(window_days=spec.window)
    if fit is None:
        fit = factor_regression(history, factor_returns[list(spec.factors)], symbols, cfg, asof)
    n, k = len(symbols), len(spec.factors)
    if fit is None:
        return {'covariance': np.eye(n)*cfg.short_history_variance, 'betas': np.zeros((n,k)),
            'info': {'samples': [0]*n, 'factor_names': list(spec.factors),
                'max_observation_at': None, 'fallback': 'insufficient_factor_history'}}
    observed, residuals, counts = fit['complete'], fit['residuals'], fit['counts']
    dates = fit['history'].index
    omega, omega_info = covariance_block(fit['factors'].to_numpy()[observed], dates[observed],
        asof, spec.factor_half_life, spec.shrinkage)
    variances = np.full(n, np.nan)
    for j in range(n):
        mask = np.isfinite(residuals[:,j])
        if mask.sum() >= cfg.min_observations:
            w = covariance_weights(dates[mask], asof, spec.residual_half_life)
            variances[j] = weighted_covariance(residuals[mask,j:j+1],w)[0,0]
    prior = max(float(np.nanmedian(variances)) if np.isfinite(variances).any() else cfg.short_history_variance, .02**2)
    for j in range(n):
        if np.isfinite(variances[j]):
            strength = counts[j]/(counts[j]+cfg.beta_prior_observations)
            variances[j] = max(strength*variances[j]+(1-strength)*prior, cfg.variance_floor)
        else:
            variances[j] = max(2*prior,cfg.short_history_variance)
    idio = np.diag(variances)
    mature = np.flatnonzero(counts >= max(60, int(observed.sum()*.9)))
    common = np.isfinite(residuals[:,mature]).all(axis=1) if len(mature) else np.zeros(len(dates),bool)
    residual_info = None
    if len(mature) >= 2 and common.sum() >= 60:
        block, residual_info = covariance_block(residuals[common][:,mature], dates[common], asof,
            spec.residual_half_life, spec.shrinkage)
        idio[np.ix_(mature,mature)] = block
    idio += np.diag(np.maximum(variances-np.diag(idio),0.))
    cov = _positive_definite(fit['betas']@omega@fit['betas'].T+idio)
    return {'covariance': cov, 'betas':fit['betas'], 'info': {'samples':counts.tolist(),
        'factor_names':list(spec.factors), 'max_observation_at':str(dates[observed].max()),
        'factor_covariance':omega_info,'residual_covariance':residual_info,
        'factor_observations':int(observed.sum()),'mature_residual_symbols':len(mature),
        'regression_time_weighting':'equal; window follows spec', 'fallback':None}}


def estimate_grid(history, factors, symbols, asof, specs):
    # Keep the same historical observation mask across all factor subsets.
    # Otherwise a newly added factor changes the sample as well as the model.
    factors = factors.where(factors[CORE_FACTORS+EXTRA_FACTORS].notna().all(axis=1), axis=0)
    fits = {}
    result = {}
    for spec in specs:
        key = (spec.factors,spec.window)
        if key not in fits:
            cfg=PaperRiskConfig(window_days=spec.window)
            fits[key]=factor_regression(history,factors[list(spec.factors)],symbols,cfg,asof)
        result[spec.name] = fit_spec(history,factors,symbols,asof,spec,fits[key])
    return result


def rolling_selection(scores, boundaries, candidates, default, lookback=180, minimum=60):
    """Only outcomes already observed at a monthly boundary may select a model."""
    records=[]
    for boundary in boundaries:
        boundary=pd.Timestamp(boundary)
        past=scores.loc[(scores.outcome_available_at<=boundary)&
            (scores['asof']>=boundary-pd.Timedelta(days=lookback))&scores.model.isin(candidates)]
        matrix=past.pivot(index='asof',columns='model',values='portfolio_qlike').reindex(columns=candidates).dropna()
        selected=default
        if len(matrix)>=minimum:
            selected=matrix.mean().sort_index().idxmin()
        records.append({'boundary':boundary,'model':selected,'validation_days':len(matrix),
            'latest_outcome_available_at':str(past.outcome_available_at.max()) if len(past) else None,
            'criterion':'past 180-calendar-day portfolio QLIKE; common complete outcomes; no trading-return selection'})
    return records
