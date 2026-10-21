import numpy as np


def transaction_cost(weights, current, rate):
    """Full traded notional/equity; no half-turnover convention."""
    return float(np.sum(np.abs(np.asarray(weights)-np.asarray(current)))*rate)
