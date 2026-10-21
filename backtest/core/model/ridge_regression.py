from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _as_2d_float(values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape={array.shape}")
    return array


def _as_1d_float(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape={array.shape}")
    return array


def _validated_sample_weight(sample_weight: np.ndarray | Sequence[float] | None, n: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)
    weight = _as_1d_float(sample_weight, "sample_weight")
    if len(weight) != n:
        raise ValueError(f"sample_weight length {len(weight)} != sample count {n}")
    if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("sample_weight must be finite, non-negative, and have positive sum")
    return weight


@dataclass(frozen=True)
class RidgeRegressionModel:
    """Small dependency-free ridge model with train-only feature scaling.

    The intercept is not penalized. Coefficients are stored on the standardized
    feature scale so their relative magnitudes are directly comparable.
    """

    feature_names: tuple[str, ...]
    alpha: float
    intercept: float
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def predict(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        values = _as_2d_float(x)
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {values.shape[1]} features, expected {len(self.feature_names)}"
            )
        standardized = (values - self.feature_mean) / self.feature_scale
        return self.intercept + standardized @ self.coefficients

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "alpha": float(self.alpha),
            "intercept": float(self.intercept),
            "coefficients_standardized": self.coefficients.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }


def fit_ridge_regression(
    x: np.ndarray | Sequence[Sequence[float]],
    y: np.ndarray | Sequence[float],
    *,
    alpha: float,
    feature_names: Iterable[str] | None = None,
    sample_weight: np.ndarray | Sequence[float] | None = None,
    penalty_weights: np.ndarray | Sequence[float] | None = None,
    min_scale: float = 1e-12,
) -> RidgeRegressionModel:
    """Fit weighted ridge regression using a stable closed-form solve.

    The optimized objective is weighted mean squared error plus
    ``alpha * sum(penalty_weights * beta**2)``. By default every coefficient
    has penalty weight one. A zero weight leaves a coefficient unpenalized.
    Feature means/scales and the target intercept are estimated from the
    supplied training observations only.
    """

    values = _as_2d_float(x)
    target = _as_1d_float(y, "y")
    if len(values) != len(target):
        raise ValueError(f"X rows {len(values)} != y length {len(target)}")
    if len(values) == 0:
        raise ValueError("cannot fit ridge regression on an empty sample")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite and non-negative, got {alpha}")

    if penalty_weights is None:
        coefficient_penalties = np.ones(values.shape[1], dtype=np.float64)
    else:
        coefficient_penalties = _as_1d_float(penalty_weights, "penalty_weights")
        if len(coefficient_penalties) != values.shape[1]:
            raise ValueError(
                f"penalty_weights length {len(coefficient_penalties)} != feature count {values.shape[1]}"
            )
        if not np.isfinite(coefficient_penalties).all() or (coefficient_penalties < 0).any():
            raise ValueError("penalty_weights must be finite and non-negative")

    weight = _validated_sample_weight(sample_weight, len(values))
    finite = np.isfinite(values).all(axis=1) & np.isfinite(target) & np.isfinite(weight) & (weight > 0)
    values = values[finite]
    target = target[finite]
    weight = weight[finite]
    if len(values) == 0:
        raise ValueError("no finite positive-weight observations remain")

    weight_sum = float(weight.sum())
    mean = np.sum(values * weight[:, None], axis=0) / weight_sum
    centered = values - mean
    variance = np.sum(centered * centered * weight[:, None], axis=0) / weight_sum
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale >= min_scale, scale, 1.0)
    standardized = centered / scale

    intercept = float(np.sum(target * weight) / weight_sum)
    centered_target = target - intercept
    sqrt_weight = np.sqrt(weight / weight_sum)
    weighted_x = standardized * sqrt_weight[:, None]
    weighted_y = centered_target * sqrt_weight
    gram = weighted_x.T @ weighted_x
    rhs = weighted_x.T @ weighted_y
    penalty = float(alpha) * np.diag(coefficient_penalties)
    try:
        coefficients = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients, *_ = np.linalg.lstsq(gram + penalty, rhs, rcond=None)

    if feature_names is None:
        names = tuple(f"x{i}" for i in range(values.shape[1]))
    else:
        names = tuple(str(name) for name in feature_names)
        if len(names) != values.shape[1]:
            raise ValueError(f"feature_names length {len(names)} != feature count {values.shape[1]}")

    return RidgeRegressionModel(
        feature_names=names,
        alpha=float(alpha),
        intercept=intercept,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        feature_mean=np.asarray(mean, dtype=np.float64),
        feature_scale=np.asarray(scale, dtype=np.float64),
    )
