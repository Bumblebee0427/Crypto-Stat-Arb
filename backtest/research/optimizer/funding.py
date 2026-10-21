"""Past-only funding forecasts, with explicit schedule and availability assumptions."""
from __future__ import annotations

import os

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import json
from .contracts import digest, file_hash, write_json


def load_settlements(data, cache):
    path = cache/'settlements.parquet'
    receipt = cache/'settlements_identity.json'
    identity = digest(data.manifest)
    if path.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['source_identity'] == identity and saved['sha256'] == file_hash(path):
            return pd.read_parquet(path)
    frames = []
    # The pivot retains nonzero historical payments, but not zero-rate settlement events.
    for j, symbol in enumerate(data.symbols):
        rates = data.prices['funding_rate'][:, j]
        mask = np.isfinite(rates) & (rates != 0)
        frames.append(pd.DataFrame({'symbol': symbol, 'time': data.times[mask], 'rate': rates[mask],
                                     'interval_hours': np.nan, 'source': 'legacy_nonzero_pivot'}))
    root = Path(os.environ.get('SELECT_COIN_FUNDING_ROOT', Path(__file__).resolve().parents[2] / 'data/fundingRate'))
    for symbol in data.symbols:
        name = symbol.replace('-', '')
        for p in sorted((root/name).glob('*-fundingRate-*.zip')):
            with zipfile.ZipFile(p) as z:
                f = pd.read_csv(z.open(z.namelist()[0]))
            if not {'calc_time', 'last_funding_rate', 'funding_interval_hours'}.issubset(f):
                continue
            times = pd.to_datetime(f.calc_time, unit='ms')
            frames.append(pd.DataFrame({'symbol': symbol, 'time': times,
                'rate': f.last_funding_rate.to_numpy(), 'interval_hours': f.funding_interval_hours.to_numpy(),
                'source': 'settled_raw_archive'}))
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows['event_hour'] = all_rows.time.dt.floor('h')
    # Raw event metadata supersedes the inferred legacy event at the same hour.
    all_rows = all_rows.drop_duplicates(['symbol', 'event_hour'], keep='last').sort_values(['symbol', 'time'])
    all_rows.to_parquet(path, index=False)
    write_json(receipt, {'source_identity': identity, 'sha256': file_hash(path),
        'scope': 'Frozen settlement snapshot; --refresh-funding explicitly rebuilds it',
        'legacy_zero_event_limitation': True})
    return all_rows


def forecast_funding(events, asof, config):
    asof = pd.Timestamp(asof)
    cutoff = asof-pd.Timedelta(minutes=config.funding_delay_minutes)
    past = events.loc[(events.time <= cutoff) & (events.time > asof-pd.Timedelta(days=7))]
    if past.empty:
        return 0., {'status': 'missing_history_zero_prior', 'available_at': None,
                     'interval_hours': None, 'events': [], 'source': 'missing'}
    last = past.iloc[-1]
    interval = last.interval_hours
    inferred = not np.isfinite(interval)
    if inferred:
        differences = past.event_hour.diff().dt.total_seconds().div(3600).dropna()
        choices = differences[differences.isin([1., 4., 8.])].tail(12)
        interval = float(choices.mode().iloc[0]) if len(choices) else 8.
    interval = float(interval)
    if not 0 < interval <= 24:
        raise ValueError('invalid historical funding interval')
    rate = float(last.rate)
    if config.funding_method == 'mean_revert':
        hours = past.interval_hours.fillna(interval)
        intensity = past.rate/hours
        rate = .5*rate+.5*float(intensity.ewm(halflife=3).mean().iloc[-1])*interval
    event = last.event_hour
    horizon = asof+pd.Timedelta(days=1)
    schedule = []
    while event < horizon:
        event += pd.Timedelta(hours=interval)
        if asof < event <= horizon:
            schedule.append(event.isoformat())
    return rate*len(schedule), {'status': 'inferred_schedule' if inferred else 'settled_metadata',
        'available_at': (last.time+pd.Timedelta(minutes=config.funding_delay_minutes)).isoformat(),
        'interval_hours': interval, 'events': schedule, 'source': last.source,
        'stale': bool(asof-last.time > pd.Timedelta(hours=interval*2))}
