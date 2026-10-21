"""Convex signed-notional allocation, with independent feasibility checks."""
from __future__ import annotations

import time
import warnings
import cvxpy as cp
import numpy as np

from .contracts import AllocationDecision
from .costs import transaction_cost


def solve_weights(baseline, current, alpha, covariance, funding, config, *,
                  use_alpha=True, use_cost=True, use_funding=True, diagonal=False,
                  no_regularization=False, inverse_vol=False, fixed=None, cost_rate=.0008,
                  funding_mode='signed', funding_scale=1., tracking_gamma=0.):
    started = time.monotonic()
    b, previous = np.asarray(baseline), np.asarray(current)
    n = len(b)
    if n == 0:
        return AllocationDecision(b.copy(), 'empty', {}, 0., 0.)
    if not all(np.isfinite(x).all() for x in [b, previous, alpha, covariance, funding]):
        raise ValueError('nonfinite optimization input')
    covariance = (covariance+covariance.T)/2
    if np.linalg.eigvalsh(covariance).min() < -1e-9:
        raise ValueError('covariance not PSD')
    cov = np.diag(np.diag(covariance)) if diagonal else covariance
    mu = np.array(alpha) if use_alpha else np.zeros(n)
    if funding_mode not in {'signed', 'pay_only'} or min(funding_scale, tracking_gamma) < 0:
        raise ValueError('invalid refinement objective')
    f = np.array(funding)*funding_scale if use_funding else np.zeros(n)
    if funding_mode == 'pay_only':
        # Signs are fixed: retain only expected payments, without a reward for receipts.
        f = np.where(np.sign(b)*f > 0, f, 0.)
    eta = 0. if no_regularization else config.eta
    signs = np.sign(b)
    x = cp.Variable(n)
    constraints = [cp.multiply(signs, x) >= np.abs(b)*config.lower_multiple,
                   cp.multiply(signs, x) <= np.abs(b)*config.upper_multiple]
    for side in [-1, 1]:
        mask = signs == side
        if mask.any():
            constraints.append(cp.sum(x[mask]) == b[mask].sum())
    if (b == 0).any():
        constraints.append(x[b == 0] == 0)
    if fixed is not None and np.any(fixed):
        constraints.append(x[fixed] == b[fixed])
    scale = 1/np.maximum(np.abs(b), 1e-8)
    scale[b == 0] = 0
    cost = cost_rate*cp.norm1(x-previous) if use_cost else 0
    if inverse_vol:
        target = np.zeros(n)
        for side in [-1, 1]:
            mask = signs == side
            if mask.any():
                inv = 1/np.sqrt(np.diag(cov)[mask])
                target[mask] = b[mask].sum()*inv/inv.sum()
        objective = -cp.sum_squares(cp.multiply(scale, x-target))
    else:
        objective = mu@x-config.gamma/2*cp.quad_form(x, cp.psd_wrap(cov))-cost-f@x
        objective -= eta/2*cp.sum_squares(cp.multiply(scale, x-b))
        if tracking_gamma:
            objective -= tracking_gamma/2*cp.quad_form(x-b, cp.psd_wrap(cov))
    # Scale objective to improve absolute solver tolerances while preserving economics.
    problem = cp.Problem(cp.Maximize(objective*10000), constraints)
    reason = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The default value of raise_error')
            problem.solve(solver='OSQP', eps_abs=1e-9, eps_rel=1e-9, max_iter=30000,
                          polishing=True, warm_start=False)
        if problem.status != 'optimal' or x.value is None:
            raise ValueError('solver status '+str(problem.status))
        w = np.asarray(x.value).reshape(-1)
        residual = max(float(np.max(np.maximum(np.abs(b)*config.lower_multiple-signs*w, 0))),
                       float(np.max(np.maximum(signs*w-np.abs(b)*config.upper_multiple, 0))),
                       max([abs(w[signs == s].sum()-b[signs == s].sum()) for s in [-1, 1]]))
        if (b == 0).any():
            residual = max(residual, float(np.max(np.abs(w[b == 0]))))
        if fixed is not None and np.any(fixed):
            residual = max(residual, float(np.max(np.abs(w[fixed]-b[fixed]))))
        if not np.isfinite(w).all() or residual > 1e-7:
            raise ValueError('allocation residual '+str(residual))
        status = 'optimal'
    except (cp.error.SolverError, ValueError) as exc:
        w, status, residual, reason = b.copy(), 'baseline_fallback', 0., str(exc)
    terms = {'expected_alpha': float(np.asarray(alpha)@w),
             'variance': float(w@cov@w), 'risk_penalty': float(config.gamma/2*(w@cov@w)),
             'expected_transaction_cost': transaction_cost(w, previous, cost_rate),
             'expected_funding_cost': float(np.asarray(funding)@w),
             'funding_objective_cost': float(f@w),
             'tracking_risk_penalty': float(tracking_gamma/2*((w-b)@cov@(w-b))),
             'regularization': float(eta/2*np.sum(((w-b)*scale)**2))}
    return AllocationDecision(w, status, terms, residual, time.monotonic()-started, reason)
