"""Build whole-universe snapshots from finalized, versioned daily inputs.

Inputs are deliberately isolated from legacy dashboard quantity factors.
TOP50 ranks only symbols with valid history and market inputs. Per-symbol
exclusions are recorded; shared input failures and an insufficient pool block.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_center.base_strategy import (
    BASE_LEVERAGE, FORMULA_VERSION, MAIN, SMALL, SOURCE_VERSION,
    STRATEGY_VERSION, finite, select, symbol_name,
)
from data_center.contract import SCHEMA_VERSION, atomic_json, digest, seal, timestamp


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def build(root, universe_path, day, now=None, profile='base', preview=False,
          factor_root=None, formula_version=FORMULA_VERSION, strategy_version=None,
          formula_code_sha256=None, btc_input=None, universe_input=None):
    root = Path(root)
    factor_root = Path(factor_root) if factor_root is not None else root
    if profile not in ('base','top50'):raise ValueError('unknown strategy profile')
    if strategy_version is None:
        strategy_version=STRATEGY_VERSION
        if profile=='top50':
            from data_center.top50_strategy import STRATEGY_VERSION as strategy_version
    from data_center.quality_tolerance import assess, POLICY_VERSION, MAX_MISSING_FRACTION
    now = now or datetime.now(timezone.utc)
    end = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(days=1)
    if end > now:
        raise ValueError("cannot build an unclosed daily signal")
    blockers, missing, factors, input_hashes = [], {}, {}, {}
    universe = universe_input if universe_input is not None else read_json(universe_path, {})
    symbols = sorted({symbol_name(s) for s in universe.get("tradable_symbols", [])}
                     - {"BTCUSDT", "ETHUSDT"})
    onboard = {symbol_name(s): d for s, d in universe.get("symbol_onboard_dates", {}).items()}
    symbols = [s for s in symbols if onboard.get(s, day) <= day]
    btc = btc_input if btc_input is not None else read_json(root / "inputs/btc_hourly.json", {})
    deferred_first_candle = []
    deferred_listing_day = []
    deferred_new_symbols = {}
    requested_symbols = list(symbols)
    excluded_symbols = {}
    young = set()
    retired_from_expected = set()
    if profile == 'top50':
        from data_center.top50_strategy import LAGGED_FORMULA_VERSION
        if formula_version == LAGGED_FORMULA_VERSION:
            from data_center.lagged_universe import readiness, new_symbols
            work=root.parent/'captured_daily_stream_v1/work'/day
            ready, deferred_new_symbols = readiness(universe,day,work if work.exists() else None)
            deferred_listing_day = sorted(s for s in deferred_new_symbols if onboard.get(s)==day)
            young = new_symbols(universe,day)
            symbols = [s for s in symbols if s in ready]
            excluded_symbols.update(deferred_new_symbols)
        known = btc.get("first_candle_by_symbol", {})
        deferred_first_candle = [s for s in symbols if s not in known]
        symbols = [s for s in symbols if s in known]
        excluded_symbols.update({s:["missing_first_usable_hourly_candle"] for s in deferred_first_candle})
    if not symbols:
        blockers.append("missing_universe")
    universe_time = universe.get("fetched_at") or universe.get("updated_at")
    if not universe_time or not 0 <= (now - timestamp(universe_time)).total_seconds() <= 36 * 3600:
        blockers.append("stale_or_unversioned_universe")

    # Hourly BTC close keys are candle BEGIN times. At D 23:00 selection,
    # shift1 uses D 22:00, with the comparison exactly 720 hours earlier.
    near = end - timedelta(hours=2)
    far = near - timedelta(days=30)
    closes = btc.get("closes", {})
    near_close, far_close = closes.get(near.isoformat()), closes.get(far.isoformat())
    trend = None
    if finite(near_close) and finite(far_close) and near_close > 0 and far_close > 0:
        trend = near_close / far_close - 1.
    else:
        blockers.append("missing_btc_exact_hourly_closes")
    input_hashes["btc_hourly"] = digest(btc)
    observed = 0
    computed_inputs = 0
    for symbol in symbols:
        history = []
        errors = []
        for offset in range(11, -1, -1):
            row_day = (end - timedelta(days=offset + 1)).date().isoformat()
            if row_day < onboard.get(symbol, row_day):
                continue  # The frozen base has min_kline_num=0, not a 12-day listing-age filter.
            try:
                row = read_json(factor_root / "inputs/daily" / row_day / f"{symbol}.json")
                if not row:
                    marker = read_json(factor_root/'inputs/live_daily'/row_day/'complete.json', {})
                    if (profile == 'top50' and not preview
                            and symbol in marker.get('excluded_symbols', {})):
                        errors.append(f"{row_day}:source_excluded:" + ','.join(marker['excluded_symbols'][symbol]))
                        continue
                    row = read_json(factor_root / 'inputs/live_daily' / row_day / f'{symbol}.json')
                if row and not isinstance(row, dict):
                    raise ValueError('daily input is not an object')
            except (OSError, ValueError, TypeError, AttributeError):
                errors.append(f"{row_day}:malformed_daily_input")
                continue
            if not row:
                if (profile == 'top50' and formula_version == LAGGED_FORMULA_VERSION
                        and row_day == onboard.get(symbol)):
                    # No sealed listing-day row exists for the lagged formula.
                    # Start rolling from available post-listing daily values;
                    # the recipe still has no minimum listing-age filter.
                    continue
                errors.append(f"{row_day}:missing")
                continue
            if row.get("symbol") != symbol or row.get("day") != row_day:
                errors.append(f"{row_day}:identity_mismatch")
                continue
            code_ok = (formula_code_sha256 is None
                       or (row.get('formula_code_sha256') == formula_code_sha256
                           and row.get('candidate_only') is False))
            if row.get('formula_version') == formula_version and row.get('source_version') == SOURCE_VERSION and code_ok:
                computed_inputs += 1
            quality_ok=(row.get('source_complete') is True and row.get('missing_trade_ids')==0)
            if profile=='top50':
                try:
                    quality_ok=assess(row,symbol,row_day,now,expected_formula=formula_version)['passes_tolerance']
                except (ValueError, KeyError, TypeError, OverflowError):
                    quality_ok=False
            if (row.get("formula_version") != formula_version or row.get("source_version") != SOURCE_VERSION
                    or not code_ok
                    or row.get("status") != "final" or not quality_ok
                    or not isinstance(row.get("minute_count"), int) or not 1 <= row["minute_count"] <= 1440
                    ):
                errors.append(f"{row_day}:quality_or_version_mismatch")
                if not (preview and row.get('formula_version')==formula_version and row.get('source_version')==SOURCE_VERSION and code_ok
                        and row.get('status')=='final' and isinstance(row.get('minute_count'),int) and 1<=row['minute_count']<=1440):
                    continue
            if timestamp(row["available_at"]) > now:
                errors.append(f"{row_day}:future_revision")
                continue
            input_hashes[f"{symbol}/{row_day}"] = digest(row)
            history.append(row)
        if errors:
            if profile == 'top50' and not preview:
                excluded_symbols[symbol]=errors
                if symbol in young:deferred_new_symbols[symbol]=errors
                retired_from_expected.add(symbol)
                continue
            missing[symbol] = errors
            if not preview or any(not e.endswith(':quality_or_version_mismatch') for e in errors):continue
        else:observed += 1
        main_values = [r[MAIN] for r in history if finite(r.get(MAIN))]
        small_values = [r[SMALL] for r in history if finite(r.get(SMALL))]
        if not main_values or not small_values:
            if profile == 'top50' and not preview:
                excluded_symbols[symbol]=['undefined_rolling_factors']
                retired_from_expected.add(symbol)
                observed-=1
            continue
        if len(history) == 12 and (len(main_values) < 12 or len(small_values) < 8):
            if profile == 'top50' and not preview:
                excluded_symbols[symbol]=['need_earlier_valid_factor_observations']
                if symbol in young:deferred_new_symbols[symbol]=excluded_symbols[symbol]
                retired_from_expected.add(symbol)
                observed-=1
                continue
            missing[symbol] = ["need_earlier_valid_factor_observations"]
            continue
        weights = range(1, len(main_values) + 1)
        main = sum(v * w for v, w in zip(main_values, weights)) / sum(weights)
        mean = lambda n: sum(small_values[-n:]) / len(small_values[-n:])
        # Exact BTC warm-up is per symbol in the historical framework. The
        # market-input producer must provide its first usable hourly candle.
        first = btc.get("first_candle_by_symbol", {}).get(symbol)
        if first is None:
            missing.setdefault(symbol, []).append("missing_first_usable_hourly_candle")
            symbol_trend = None
        else:
            try:
                symbol_trend = trend if timestamp(first) <= far else None
            except (ValueError, TypeError, AttributeError):
                if profile == 'top50' and not preview:
                    excluded_symbols[symbol]=['invalid_first_usable_hourly_candle']
                    retired_from_expected.add(symbol)
                    observed-=1
                    continue
                raise
        factors[symbol] = {"main": main, "small_long": mean(8), "small_short": mean(4), "btc": symbol_trend}
        if profile=='top50':
            try:
                volume=read_json(root/'inputs/volume25'/day/f'{symbol}.json',{})
                if not isinstance(volume,dict):raise ValueError('invalid volume input')
                volume_time=timestamp(volume['available_at']) if volume.get('available_at') else None
            except (OSError, ValueError, TypeError, AttributeError):
                volume={};volume_time=None
            if (volume.get('complete') is not True or volume.get('hours')!=599 or volume.get('symbol')!=symbol
                or volume.get('day')!=day
                or volume.get('window_end')!=(end-timedelta(hours=1)).isoformat()
                or volume.get('target_window_end')!=end.isoformat()
                or volume.get('volume_lag_hours')!=1
                or not finite(volume.get('quote_volume_mean')) or not volume.get('available_at')
                or volume_time is None or volume_time>now):
                factors.pop(symbol)
                if not preview:
                    excluded_symbols[symbol]=['volume25_incomplete_or_stale']
                    if symbol in young:deferred_new_symbols[symbol]=excluded_symbols[symbol]
                    retired_from_expected.add(symbol)
                    observed-=1
                else:
                    missing.setdefault(symbol,[]).append('volume25_incomplete_or_stale')
            else:
                factors[symbol]['quote_volume_mean']=volume['quote_volume_mean']
                input_hashes['volume25/'+symbol]=digest(volume)
    symbols=[s for s in symbols if s not in retired_from_expected]
    if observed != len(symbols) or missing:
        blockers.append("factor_history_or_market_metadata_incomplete")
    if len(factors) < 20:
        blockers.append("eligible_universe_below_20")

    receipt = read_json(root / ('top50/validation.json' if profile=='top50' else 'validation.json'), {})
    checks = ("source_event_parity", "quantile_interpolation", "window_boundary_parity",
              "daily_factor_parity", "target_parity", "availability_no_lookahead")
    code_hashes = {name: digest((Path(__file__).parent / name).read_text())
                   for name in ("base_strategy.py", "reference_formula_v2.py", "factor_worker.py", "publisher.py")}
    if profile=='top50':
        code_hashes.update({name:digest((Path(__file__).parent/name).read_text()) for name in ('top50_strategy.py','volume_inputs.py','quality_tolerance.py')})
    receipt_valid = (receipt.get("strategy_version") == strategy_version
            and receipt.get("formula_version") == formula_version
            and receipt.get("source_version") == SOURCE_VERSION
            and receipt.get("code_hashes") == code_hashes
            and all(receipt.get("checks", {}).get(c) is True for c in checks)
            and receipt.get("evidence"))
    if profile != 'top50' and not receipt_valid:
        blockers.append("formula_and_source_parity_not_certified")
    selected = select(factors, trend, volume_top_pct=.5 if profile=='top50' else None)
    allocations = selected["allocations"]
    if not selected["longs"] or not selected["shorts"]:
        blockers.append("empty_strategy_leg")
    if preview:blockers.append('diagnostic_preview_not_actionable')
    diagnostic_universe_complete=bool(symbols) and trend is not None and not any(
        any(not error.endswith(':quality_or_version_mismatch') for error in errors)
        for errors in missing.values())
    return seal({
        "schema_version": SCHEMA_VERSION, "strategy_version": strategy_version,
        'diagnostic_only':preview,'volume_hours':599 if profile=='top50' else None,
        'volume_lag_hours':1 if profile=='top50' else None,
        'volume_top_pct':.5 if profile=='top50' else None,
        'source_certification':'user_waived_20260925' if profile=='top50' else ('certified' if receipt_valid else 'missing'),
        'diagnostic_universe_complete':diagnostic_universe_complete,
        'source_quality_policy': {'version':POLICY_VERSION,'max_missing_fraction_inclusive':MAX_MISSING_FRACTION} if profile=='top50' else {'version':'strict_source_complete_zero_gaps'},
        "formula_version": formula_version, "source_version": SOURCE_VERSION,
        "universe_version": digest(universe), "signal_day": day,
        "window_start": (end - timedelta(days=1)).isoformat(),
        "window_end": end.isoformat(), "available_at": now.isoformat(),
        "generated_at": now.isoformat(), "execution_deadline": (end + timedelta(minutes=30)).isoformat(),
        "valid_until": (end + timedelta(days=1)).isoformat(),
        "status": "blocked" if blockers else "complete", "blockers": blockers,
        "coverage": {"requested": len(requested_symbols), "expected": len(symbols), "observed": observed, "eligible": len(factors),
                     "excluded_symbols": excluded_symbols,
                     "missing": missing, "deferred_first_candle": deferred_first_candle,
                     "deferred_listing_day": deferred_listing_day,
                     "deferred_new_symbols": deferred_new_symbols,
                     "required_history_days": 12, "computed_daily_inputs": computed_inputs},
        "leverage": BASE_LEVERAGE, "allocations": allocations,
        "gross_allocation": sum(abs(v) for v in allocations.values()),
        "factors": factors, "selection": selected,
        "input_manifest_hash": digest(input_hashes), "validation_hash": digest(receipt),
        "revision": digest(input_hashes),
        "source_watermark": end.isoformat() if symbols and observed == len(symbols) else None,
        "submitted_to_exchange": False,
    })


def publish(root, snapshot):
    root = Path(root)
    previous = read_json(root / "latest.json")
    if previous and timestamp(previous["window_end"]) > timestamp(snapshot["window_end"]):
        raise ValueError("refusing to move latest backwards")
    atomic_json(root / "snapshots" / f"{snapshot['snapshot_id']}.json", snapshot)
    atomic_json(root / "latest.json", snapshot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--universe", required=True)
    p.add_argument("--day", default=None)
    p.add_argument('--profile',choices=['base','top50'],default='base')
    a = p.parse_args()
    day = a.day or (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    snapshot = build(a.root, a.universe, day,profile=a.profile)
    output=Path(a.root)/'top50' if a.profile=='top50' else Path(a.root)
    publish(output, snapshot)
    if a.profile=='top50':
        atomic_json(output/'preview.json',build(a.root,a.universe,day,profile='top50',preview=True))
    print(json.dumps({k: snapshot[k] for k in ("signal_day", "snapshot_id", "status", "blockers")}))


if __name__ == "__main__":
    main()
