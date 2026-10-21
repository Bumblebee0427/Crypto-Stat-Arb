"""Predeclared second-round allocation and execution experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import numpy as np
import pandas as pd

from .policy import make_policy
from .contracts import write_json


@dataclass(frozen=True)
class Experiment:
    name: str
    label: str
    policy: str | None = 'B5_full'
    blend: float = 1.
    overrides: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    resize_threshold: float = 0.
    refined_alpha: bool = False


EXPERIMENTS = [
    Experiment('B0_baseline', 'Original weights', policy=None),
    Experiment('B5_full', 'First-round full objective'),
    Experiment('mix_10', '10% optimizer blend', blend=.1),
    Experiment('mix_25', '25% optimizer blend', blend=.25),
    Experiment('mix_50', '50% optimizer blend', blend=.5),
    Experiment('no_alpha_mix25', '25% blend, zero allocation alpha', blend=.25,
               overrides={'use_alpha': False}),
    Experiment('funding_pay_mix25', '25% blend, payment-only funding', blend=.25,
               overrides={'funding_mode': 'pay_only'}),
    Experiment('funding_half_mix25', '25% blend, half funding forecast', blend=.25,
               overrides={'funding_scale': .5}),
    Experiment('refined_alpha_mix25', '25% blend, richer shrunk price alpha', blend=.25,
               refined_alpha=True),
    Experiment('refined_pay_mix25', '25% blend, richer alpha + payment-only funding', blend=.25,
               refined_alpha=True, overrides={'funding_mode': 'pay_only'}),
    Experiment('cost_only', 'Bounded turnover control',
               overrides={'use_alpha': False, 'use_funding': False},
               config={'gamma': 0., 'eta': .001, 'lower_multiple': .8, 'upper_multiple': 1.2}),
    Experiment('tracking_risk', 'Turnover control + active risk',
               overrides={'use_alpha': False, 'use_funding': False, 'tracking_gamma': 10.},
               config={'gamma': 0., 'eta': .001, 'lower_multiple': .8, 'upper_multiple': 1.2}),
    Experiment('baseline_band10', 'Original weights + 10% resize band', policy=None, resize_threshold=.1),
    Experiment('baseline_band25', 'Original weights + 25% resize band', policy=None, resize_threshold=.25),
    Experiment('mix25_band10', '25% blend + 10% resize band', blend=.25, resize_threshold=.1),
    Experiment('cost_only_band10', 'Turnover control + 10% resize band', resize_threshold=.1,
               overrides={'use_alpha': False, 'use_funding': False},
               config={'gamma': 0., 'eta': .001, 'lower_multiple': .8, 'upper_multiple': 1.2}),
]

BY_NAME = {e.name: e for e in EXPERIMENTS}
# Fixed BEFORE second-round outcomes are read; include the original as a candidate.
SELECTION_CANDIDATES = ['B0_baseline', 'mix_25', 'funding_pay_mix25', 'cost_only', 'baseline_band10']


def policy_for(experiment, data, forecasts, config, refined):
    if experiment.policy is None:
        return None
    cfg = replace(config, **experiment.config)
    return make_policy(data, forecasts, cfg, experiment.policy, overrides=experiment.overrides,
                       blend=experiment.blend,
                       alpha_overrides=refined if experiment.refined_alpha else None)


def build_refined_alpha(flat, config, out):
    """Monthly past-only ridge on ranks, factor levels, volatility and known funding.

    Targets are PRICE returns only; funding stays a separate objective/ledger term.
    Preprocessing is fitted on mature training rows, never on test labels.
    """
    from core.model.ridge_regression import fit_ridge_regression
    x = flat.copy()
    x['log_vol'] = np.log(x.predicted_vol.clip(lower=1e-6))
    features = ['main_rank', 'small_rank', 'main', 'small', 'log_vol', 'expected_funding']
    records, models, values = [], [], {}
    models_by_side, month = {}, None
    for asof, rows in x.groupby('asof', sort=True):
        current_month = asof.to_period('M').start_time
        if month != current_month:
            month = current_month; models_by_side = {}
            train = x.loc[(x['asof'] >= month-pd.Timedelta(days=config.train_days)) &
                          (x.label_available_at <= month)].dropna(subset=features+['future_return'])
            for side, part in train.groupby('side'):
                if part['asof'].nunique() < config.train_days-3:
                    continue
                inputs = part[features].to_numpy(float)
                lo, hi = np.quantile(inputs, [.01, .99], axis=0)
                inputs = np.clip(inputs, lo, hi)
                mean, scale = inputs.mean(axis=0), inputs.std(axis=0)
                scale = np.maximum(scale, 1e-8)
                y = part.future_return-part.groupby('asof').future_return.transform('mean')
                ylo, yhi = y.quantile([.01, .99]); y = y.clip(ylo, yhi)
                weights = 1/part.groupby('asof')['asof'].transform('size')
                fitted = fit_ridge_regression((inputs-mean)/scale, y.to_numpy(),
                    alpha=10., feature_names=features, sample_weight=weights.to_numpy())
                cutoff = part.label_available_at.max()
                models_by_side[int(side)] = (fitted, lo, hi, mean, scale, max(abs(ylo), abs(yhi)), cutoff)
                models.append({'fitted_at': month, 'training_label_cutoff': cutoff,
                    'side': int(side), 'days': int(part['asof'].nunique()),
                    'features': features, 'ridge_penalty': 10., 'alpha_shrink': .25,
                    'feature_clip_low': lo.tolist(), 'feature_clip_high': hi.tolist(),
                    'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(), 'model': fitted.to_dict()})
        prediction = np.zeros(len(rows))
        for side in [-1, 1]:
            pos = np.flatnonzero(rows.side.to_numpy() == side)
            supported = side in models_by_side
            if supported and len(pos):
                fitted, lo, hi, mean, scale, bound, cutoff = models_by_side[side]
                inputs = rows.iloc[pos][features].to_numpy(float)
                valid = np.isfinite(inputs).all(axis=1)
                if valid.any():
                    yhat = fitted.predict((np.clip(inputs[valid], lo, hi)-mean)/scale)
                    yhat -= yhat.mean()
                    prediction[pos[valid]] = np.clip(yhat, -bound, bound)*.25
            for k in pos:
                r = rows.iloc[k]
                records.append({'asof': asof, 'symbol': r.symbol, 'side': side,
                    'alpha_refined': float(prediction[k]), 'fitted_at': month,
                    'training_label_cutoff': models_by_side[side][-1] if supported else pd.NaT,
                    'supported': bool(supported and np.isfinite(r[features].to_numpy(float)).all())})
        values[asof] = prediction
    pd.DataFrame(records).to_parquet(out/'refined_alpha.parquet', index=False)
    write_json(out/'refined_alpha_models.json', models)
    return values


def select_refinement_schedule(curves, config, last, metric='sortino'):
    from .validation import daily_returns, summary
    if metric not in {'sortino', 'sharpe'}:
        raise ValueError('unknown selection metric')
    daily = {n: daily_returns(curves[n]) for n in SELECTION_CANDIDATES}
    schedule, splits = {}, []
    for boundary in pd.date_range(config.first_test, last, freq=f'{config.test_days}D'):
        start = boundary-pd.Timedelta(days=config.validation_days)
        scores = {}
        for name, frame in daily.items():
            past = frame.loc[(frame.index >= start) & (frame.index < boundary)]
            if len(past) >= 60:
                scores[name] = summary(past)
        valid = {n: s[metric] for n, s in scores.items() if s[metric] is not None and np.isfinite(s[metric])}
        tolerance = .1 if metric == 'sortino' else .05
        if valid:
            best = max(valid.values())
            # Prefer the original, then fewer changed components when scores are close.
            chosen = next(n for n in SELECTION_CANDIDATES if valid.get(n, -np.inf) >= best-tolerance)
        else:
            chosen = 'B0_baseline'
        schedule[boundary] = chosen
        splits.append({'test_start': boundary, 'test_end_exclusive': boundary+pd.Timedelta(days=config.test_days),
            'validation_start': start, 'validation_end_exclusive': boundary,
            'selected_model': chosen, 'selection_metric': metric, 'validation_scores': scores,
            'candidate_count': len(SELECTION_CANDIDATES), 'account_reset': False,
            'training': 'monthly trailing mature-label window', 'tie_tolerance': tolerance})
    return schedule, splits


def scheduled_policy(schedule, data, forecasts, config, refined):
    policies = {n: policy_for(BY_NAME[n], data, forecasts, config, refined) for n in SELECTION_CANDIDATES}
    boundaries = sorted(schedule)
    def chosen(asof):
        past = [t for t in boundaries if t <= asof]
        return schedule[past[-1]] if past else 'B0_baseline'
    def decide(i, asof, baseline, state):
        name = chosen(asof); policy = policies[name]
        if policy is None:
            return baseline, []
        weights, records = policy(i, asof, baseline, state)
        for record in records:
            record['selected_model'] = name
        return weights, records
    def threshold(asof):
        return BY_NAME[chosen(asof)].resize_threshold
    return decide, threshold


def experiment_manifest(config):
    return {'experiments': [asdict(e) for e in EXPERIMENTS],
        'selection_candidates': SELECTION_CANDIDATES, 'selection_metrics': ['sortino', 'sharpe'],
        'stress_cases': {'oneway_cost_bps': 12., 'fixed_candidate': 'cost_only_band10'},
        'config': asdict(config), 'resize_side_budget_guard': .05,
        'selection_note': 'Predeclared small menu. Previously studied history remains retrospective OOS.',
        'real_orders': False}
