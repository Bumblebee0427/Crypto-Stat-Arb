"""PSD daily factor covariance with a shrunk mature residual block."""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_risk(history, symbols, config):
    factors = history.reindex(columns=['BTC-USDT', 'ETH-USDT'])
    f = factors.to_numpy(dtype=float)
    observed_f = np.isfinite(f).all(axis=1)
    if observed_f.sum() < config.min_risk_days:
        diagonal = np.full(len(symbols), .1**2)
        return np.diag(diagonal), {'samples': [0]*len(symbols), 'fallback': 'insufficient_market_history',
                                  'min_eigenvalue': .01, 'condition': 1., 'mature_residual_symbols': 0}, np.zeros((len(symbols), 2))
    omega = np.cov(f[observed_f], rowvar=False)
    omega = .95*omega+.05*np.diag(np.diag(omega))
    data = history.reindex(columns=symbols).to_numpy(dtype=float)
    n = len(symbols); betas = np.tile([1., 0.], (n, 1))
    residual = np.full_like(data, np.nan)
    variances = np.full(n, np.nan); counts = []
    for j in range(n):
        mask = observed_f & np.isfinite(data[:, j]); count = int(mask.sum()); counts.append(count)
        if count >= config.min_risk_days:
            x = f[mask]-f[mask].mean(axis=0); y = data[mask, j]-data[mask, j].mean()
            scale = max(float(np.trace(x.T@x))/len(x), 1e-8)
            raw_beta = np.linalg.solve(x.T@x+np.eye(2)*scale*5, x.T@y)
            strength = count/(count+20.)
            betas[j] = strength*raw_beta+(1-strength)*np.array([1., 0.])
            residual[mask, j] = data[mask, j]-f[mask]@betas[j]
            variances[j] = np.var(residual[mask, j], ddof=1)
    prior = float(np.nanmedian(variances)) if np.isfinite(variances).any() else .1**2
    prior = max(prior, .02**2)
    for j in range(n):
        count = counts[j]
        if np.isfinite(variances[j]):
            strength = count/(count+20.)
            variances[j] = max(strength*variances[j]+(1-strength)*prior, .01**2)
        else:
            variances[j] = max(prior*2, .1**2)
    idio = np.diag(variances)
    mature = np.flatnonzero(np.asarray(counts) >= max(60, int(len(history)*.9)))
    common = np.isfinite(residual[:, mature]).all(axis=1) if len(mature) else np.zeros(len(history), bool)
    if len(mature) >= 2 and common.sum() >= 60:
        sample = np.cov(residual[common][:, mature], rowvar=False)
        block = (1-config.risk_shrink)*sample+config.risk_shrink*np.diag(variances[mature])
        idio[np.ix_(mature, mature)] = block
    else:
        mature = []
    cov = betas@omega@betas.T+idio
    cov = (cov+cov.T)/2
    eigen = np.linalg.eigvalsh(cov)
    if not np.isfinite(cov).all() or eigen.min() < -1e-10:
        raise ValueError('invalid covariance')
    cov += np.eye(n)*max(0., 1e-8-eigen.min())
    return cov, {'samples': counts, 'fallback': 'factor_prior_for_short_history',
                 'min_eigenvalue': float(np.linalg.eigvalsh(cov).min()),
                 'condition': float(np.linalg.cond(cov)), 'mature_residual_symbols': len(mature)}, betas
