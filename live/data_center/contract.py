"""Dependency-free snapshot validation shared with the remote trading consumer."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from data_center.base_strategy import STRATEGY_VERSION, FORMULA_VERSION
SCHEMA_VERSION = "select_coin.targets.v1"


def timestamp(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("timestamps require UTC offset")
    return value.astimezone(timezone.utc)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(payload):
    body = {k: v for k, v in payload.items() if k not in ("snapshot_id", "checksum")}
    checksum = digest(body)
    return {**body, "snapshot_id": checksum, "checksum": checksum}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate(snapshot, now=None, actionable=True, expected_strategy=STRATEGY_VERSION,
             expected_formula=FORMULA_VERSION):
    now = now or datetime.now(timezone.utc)
    if seal(snapshot) != snapshot:
        raise ValueError("snapshot checksum mismatch")
    if snapshot.get("schema_version") != SCHEMA_VERSION or snapshot.get("strategy_version") != expected_strategy:
        raise ValueError("unsupported strategy/schema version")
    if snapshot.get("formula_version") != expected_formula:
        raise ValueError("unsupported formula version")
    window_end = timestamp(snapshot["window_end"])
    available = timestamp(snapshot["available_at"])
    generated = timestamp(snapshot["generated_at"])
    deadline = timestamp(snapshot["execution_deadline"])
    expiry = timestamp(snapshot["valid_until"])
    if not window_end <= available <= generated <= now:
        raise ValueError("future or inconsistent snapshot timestamps")
    if not window_end < deadline <= expiry or (expiry - window_end).total_seconds() > 86400:
        raise ValueError("invalid snapshot validity bounds")
    if now >= expiry:
        raise ValueError("stale snapshot")
    if actionable and now >= deadline:
        raise ValueError("missed rebalance execution deadline")
    if actionable and (snapshot.get("status") != "complete" or snapshot.get("blockers")):
        raise ValueError("snapshot not ready: " + ", ".join(snapshot.get("blockers", [])))
    if actionable and snapshot.get('diagnostic_only'):
        raise ValueError('diagnostic preview cannot be executed')
    coverage = snapshot.get("coverage", {})
    if actionable and (not coverage.get("expected") or coverage.get("observed") != coverage.get("expected")):
        raise ValueError("incomplete universe coverage")
    if snapshot.get("leverage") != 2.0:
        raise ValueError("baseline leverage mismatch")
    weights = snapshot.get("allocations")
    if not isinstance(weights, dict) or (actionable and not weights):
        raise ValueError("missing allocations")
    for symbol, weight in weights.items():
        if not re.fullmatch(r"[A-Z0-9\u3400-\u4dbf\u4e00-\u9fff]+USDT", symbol) or symbol in ("BTCUSDT", "ETHUSDT"):
            raise ValueError("invalid or blacklisted target symbol")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ValueError("nonfinite allocation")
    gross = sum(abs(v) for v in weights.values())
    if gross > 1.0 + 1e-10 or abs(gross - snapshot.get("gross_allocation", -1)) > 1e-10:
        raise ValueError("invalid gross allocation")
    return snapshot
