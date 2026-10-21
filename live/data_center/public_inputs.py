"""Small public-only hourly market inputs for the frozen base strategy."""
import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from data_center.base_strategy import symbol_name
from data_center.contract import atomic_json, timestamp


def first_candle(symbol, start):
    params = {"symbol": symbol, "interval": "1h", "startTime": int(start.timestamp() * 1000), "limit": 1}
    url = "https://fapi.binance.com/fapi/v1/klines?" + urlencode(params)
    with urlopen(url, timeout=15) as response:
        rows = json.load(response)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise ValueError("missing public hourly candle")
    return rows[0]


def fetch_btc_closes(near):
    closes = {}
    for instant in (near, near - timedelta(days=30)):
        row = first_candle("BTCUSDT", instant)
        if (int(row[0]) != int(instant.timestamp() * 1000)
                or int(row[6]) >= datetime.now(timezone.utc).timestamp() * 1000):
            raise ValueError("BTC candle is unclosed or has an unexpected timestamp")
        closes[instant.isoformat()] = float(row[4])
    return closes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--universe", required=True)
    p.add_argument("--max-new-symbols", type=int, default=20)
    args = p.parse_args()
    now = datetime.now(timezone.utc)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # At 23:00 UTC the next signal's 22:00 UTC candle has already closed.
    # Fetch it before midnight so the boundary publisher can use it immediately.
    if now.hour == 23:
        end += timedelta(days=1)
    near = end - timedelta(hours=2)
    root = Path(args.root)
    path = root / "inputs/btc_hourly.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    first = previous.get("first_candle_by_symbol", {})
    closes = fetch_btc_closes(near)
    universe = json.loads(Path(args.universe).read_text())
    onboards = {symbol_name(s): d for s, d in universe.get("symbol_onboard_dates", {}).items()}
    # Keep newly listed symbols out of the UTC-boundary signal window.
    new_symbol_limit = (0 if (now.hour == 23 and now.minute >= 50)
                        or (now.hour == 0 and now.minute < 8)
                        else args.max_new_symbols)
    queried, failures = 0, {}
    for raw_symbol in universe.get("tradable_symbols", []):
        symbol = symbol_name(raw_symbol)
        if symbol in first or queried >= new_symbol_limit:
            continue
        queried += 1
        try:
            start = timestamp(onboards.get(symbol, "1970-01-01") + "T00:00:00+00:00")
            row = first_candle(symbol, start)
            first[symbol] = datetime.fromtimestamp(int(row[0]) / 1000, timezone.utc).isoformat()
        except (ValueError, OSError) as exc:
            failures[symbol] = type(exc).__name__
            # Fail a batch early on exchange throttling or network outage.
            if getattr(exc, "code", None) in (418, 429):
                break
        time.sleep(.05)
    atomic_json(path, {"source": "binance_public_fapi_1h", "closes": closes,
                       "first_candle_by_symbol": first, "fetched_at": now.isoformat(), "failures": failures})
    print(json.dumps({"target_boundary_utc": end.isoformat(), "btc_closes": len(closes),
                      "first_candle_symbols": len(first), "queried": queried, "failures": failures}))


if __name__ == "__main__":
    main()
