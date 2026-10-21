"""Synthetic snapshot shared by the exported selection tests."""
from datetime import datetime, timezone

from data_center.base_strategy import STRATEGY_VERSION, FORMULA_VERSION
from data_center.contract import SCHEMA_VERSION, seal

NOW = datetime(2026, 9, 15, 0, 10, tzinfo=timezone.utc)


def snapshot():
    return seal({"schema_version": SCHEMA_VERSION, "strategy_version": STRATEGY_VERSION,
                 "formula_version": FORMULA_VERSION, "window_end": "2026-09-15T00:00:00+00:00",
                 "available_at": "2026-09-15T00:01:00+00:00", "generated_at": "2026-09-15T00:02:00+00:00",
                 "execution_deadline": "2026-09-15T00:30:00+00:00", "valid_until": "2026-09-16T00:00:00+00:00",
                 "status": "complete", "blockers": [], "coverage": {"expected": 100, "observed": 100},
                 "allocations": {"ADAUSDT": .5, "XRPUSDT": -.5}, "gross_allocation": 1., "leverage": 2.})
