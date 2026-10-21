from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchConfig:
    train_days: int = 365
    risk_days: int = 180
    min_risk_days: int = 20
    ridge_penalty: float = 1.0
    alpha_shrink: float = 0.5
    risk_shrink: float = 0.5
    funding_delay_minutes: int = 5
    funding_method: str = 'mean_revert'
    lower_multiple: float = 0.5
    upper_multiple: float = 1.5
    gamma: float = 2.0
    eta: float = 0.00001
    initial_equity: float | None = None
    cost_multiplier: float = 1.0
    validation_days: int = 90
    test_days: int = 90
    first_test: str = '2022-04-01'
    seed: int = 20261019

    def __post_init__(self):
        if not 0 <= self.lower_multiple <= 1 <= self.upper_multiple:
            raise ValueError('baseline must be feasible under relative bounds')
        if min(self.gamma, self.eta, self.cost_multiplier) < 0:
            raise ValueError('penalties cannot be negative')
        if self.funding_method not in {'last', 'mean_revert'}:
            raise ValueError('unknown funding forecast')
        if not 0 <= self.alpha_shrink <= 1 or not 0 <= self.risk_shrink <= 1:
            raise ValueError('invalid shrinkage')
        if self.initial_equity is not None and self.initial_equity <= 0:
            raise ValueError('initial equity must be positive')
        if min(self.train_days, self.risk_days, self.min_risk_days, self.test_days, self.validation_days) <= 0:
            raise ValueError('invalid history or evaluation window')


@dataclass
class AccountState:
    equity: float
    lots: np.ndarray
    targets: np.ndarray
    last_prices: np.ndarray


@dataclass
class AllocationDecision:
    weights: np.ndarray
    status: str
    terms: dict
    residual: float
    elapsed_seconds: float
    fallback_reason: str | None = None


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
                                   allow_nan=False) + '\n')
    temporary.replace(path)
