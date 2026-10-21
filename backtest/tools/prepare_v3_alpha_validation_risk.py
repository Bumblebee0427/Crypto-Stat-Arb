#!/usr/bin/env python3
"""Construct historical validation risk inputs with the live risk definitions."""
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.paper_factors import PaperRiskConfig, load_capitalization, build_factors
from research.optimizer.contracts import file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--cap-root', type=Path, required=True)
    p.add_argument('--live-root', type=Path)
    a = p.parse_args(); out = a.history/'risk'; out.mkdir(exist_ok=True)
    closes = pd.read_parquet(a.history/'daily_closes.parquet').loc['2026-03-01':]
    quotes = pd.read_parquet(a.history/'daily_quotes.parquet').reindex(closes.index)
    for x in [closes, quotes]: x.columns = [s[:-4]+'-USDT' for s in x.columns]
    if a.live_root:
        closes = pd.read_parquet(a.live_root/'closes.parquet').combine_first(closes).sort_index()
        quotes = pd.read_parquet(a.live_root/'quotes.parquet').combine_first(quotes).sort_index()
    cfg = PaperRiskConfig(window_days=90, ewma_half_life_days=14)
    cap, ids, available, _ = load_capitalization(a.cap_root, closes, cfg, out)
    factors = build_factors(closes, quotes, cap, ids, available, cfg, out=out)
    returns = closes.pct_change(fill_method=None)
    returns.columns = [s.replace('-', '') for s in returns.columns]
    returns.to_parquet(out/'returns.parquet'); factors.to_parquet(out/'factors.parquet')
    write_json(out/'manifest.json', {'factor_file_sha256': file_hash(out/'factors.parquet'),
        'return_file_sha256': file_hash(out/'returns.parquet'),
        'first_complete_factor': str(factors.dropna().index.min()),
        'last_complete_factor': str(factors.dropna().index.max()),
        'definition': 'live estimate_grid core3; all6 observation mask; official historical cap with source date +2 delay',
        'publication_vintages_available': False})
    print((out/'manifest.json').read_text())


if __name__ == '__main__': main()
