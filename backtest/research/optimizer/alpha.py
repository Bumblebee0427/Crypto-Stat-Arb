"""Train-only, leg-conditional price-return forecasts in decimal units."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from core.model.ridge_regression import fit_ridge_regression
from .contracts import digest

FEATURES = ['main_rank', 'small_rank']


@dataclass
class AlphaModel:
    fitted_at: pd.Timestamp
    label_cutoff: pd.Timestamp | None
    models: dict
    clips: dict
    bins: dict
    training_days: int
    shrink: float


def alpha_fingerprint(model):
    return digest({'models':{str(k):v.to_dict() for k,v in model.models.items()},
                   'clips':model.clips, 'shrink':model.shrink})


def fit_alpha(panel, cutoff, config):
    cutoff = pd.Timestamp(cutoff)
    train = panel.loc[(panel['asof'] < cutoff) &
                      (panel['asof'] >= cutoff-pd.Timedelta(days=config.train_days)) &
                      (panel.label_available_at <= cutoff)].copy()
    train = train.dropna(subset=FEATURES+['future_return'])
    models, clips, bins = {}, {}, {}
    days = int(train['asof'].nunique())
    for side, part in train.groupby('side'):
        if part['asof'].nunique() < config.train_days-3:
            continue
        # Common leg/day intercept has no allocation value under fixed budgets.
        y = part.future_return-part.groupby('asof').future_return.transform('mean')
        lo, hi = y.quantile([.01, .99]); y = y.clip(lo, hi)
        weights = 1/part.groupby('asof')['asof'].transform('size')
        models[int(side)] = fit_ridge_regression(part[FEATURES].to_numpy(), y.to_numpy(),
            alpha=config.ridge_penalty, feature_names=FEATURES, sample_weight=weights.to_numpy())
        clips[int(side)] = float(max(abs(lo), abs(hi)))
        b = np.minimum((part.main_rank.to_numpy()+1)*1.5, 2).astype(int)
        means = []
        for k in range(3):
            m = b == k
            means.append(float(np.average(y.to_numpy()[m], weights=weights.to_numpy()[m])) if m.any() else 0.)
        bins[int(side)] = means
    end = train.label_available_at.max() if len(train) else None
    return AlphaModel(cutoff, end, models, clips, bins, days, config.alpha_shrink)


def predict_alpha(rows, model, method='ridge'):
    if len(rows) and rows['asof'].min() < model.fitted_at:
        raise ValueError('future-trained model')
    result = np.zeros(len(rows))
    supported = np.zeros(len(rows), dtype=bool)
    for side in [-1, 1]:
        pos = np.flatnonzero(rows.side.to_numpy() == side)
        if side not in model.models or not len(pos):
            continue
        x = rows.iloc[pos][FEATURES].to_numpy()
        valid = np.isfinite(x).all(axis=1)
        use = pos[valid]
        if method == 'ridge':
            predictions = model.models[side].predict(x[valid])
        elif method == 'bins':
            bucket = np.minimum((x[valid, 0]+1)*1.5, 2).astype(int)
            predictions = np.asarray(model.bins[side])[bucket]
        else:
            raise ValueError('unknown alpha model')
        if len(use):
            predictions = predictions-predictions.mean()
            result[use] = np.clip(predictions, -model.clips[side], model.clips[side])*model.shrink
            supported[use] = True
    return result, supported
