#!/usr/bin/env python3
"""Run the isolated historical optimizer; never publishes signals or submits orders."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import pickle
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.optimizer.contracts import ResearchConfig, file_hash, write_json, digest
from research.optimizer.dataset import prepare, DEFAULT_MANIFEST
from research.optimizer.replay import parity, replay
from research.optimizer.forecasts import build_forecasts
from research.optimizer.policy import make_policy, SPECS
from research.optimizer.validation import select_schedule
from research.optimizer.reporting import create_report
from research.optimizer.funding import load_settlements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-stress', action='store_true')
    parser.add_argument('--refresh-funding', action='store_true')
    args = parser.parse_args()
    cfg = ResearchConfig(**json.loads(args.config.read_text())) if args.config else ResearchConfig()
    out = args.out.resolve()
    source = json.loads(args.manifest.read_text())
    frozen = Path(source['result_folder']).resolve().parents[2]
    if out == frozen or frozen in out.parents:
        raise ValueError('research output cannot overwrite the frozen source workspace')
    out.mkdir(parents=True, exist_ok=True)
    data = prepare(args.manifest, out/'prepared')
    code = {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT/'research/optimizer').glob('*.py')}
    code[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    runtime = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'numba', 'cvxpy', 'osqp', 'scipy', 'pyarrow']}
    identity = {'config': asdict(cfg), 'source': data.manifest, 'code': code, 'runtime': runtime}
    write_json(out/'manifest.json', identity)
    print('verify original hourly accounting', flush=True)
    checked = parity(data, out/'baseline')
    print(json.dumps(checked), flush=True)
    if args.parity_only:
        return
    if args.refresh_funding:
        (out/'prepared/settlements_identity.json').unlink(missing_ok=True)
        (out/'forecast_identity.json').unlink(missing_ok=True)
    load_settlements(data, out/'prepared')
    identity['funding_snapshot_sha256'] = file_hash(out/'prepared/settlements.parquet')
    write_json(out/'manifest.json', identity)
    forecast_key = digest({'config': asdict(cfg), 'source': data.manifest,
        'funding_snapshot_sha256': identity['funding_snapshot_sha256'],
        'code': {k: v for k, v in code.items() if Path(k).name in
                 ['alpha.py', 'risk.py', 'funding.py', 'forecasts.py', 'dataset.py', 'contracts.py']}})
    fp, fm = out/'forecast_cache.pkl', out/'forecast_identity.json'
    if args.resume and fp.exists() and fm.exists() and json.loads(fm.read_text())['key'] == forecast_key:
        forecasts = pickle.loads(fp.read_bytes())
        print('reuse verified forecast cache', flush=True)
    else:
        forecasts = build_forecasts(data, cfg, out)
        temporary = fp.with_suffix('.pkl.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(forecasts, stream, protocol=5)
        temporary.replace(fp)
        write_json(fm, {'key': forecast_key})
    # Account calculations depend on replay/policy/solve; report edits do not invalidate them.
    account_code = {k: v for k, v in code.items() if Path(k).name in
                    ['replay.py', 'policy.py', 'solve.py', 'contracts.py', 'costs.py']}
    curves = {}
    def run(name, config=cfg, policy_name=None, schedule=None, ledger=False):
        target = out/name
        key = digest({'forecast': forecast_key, 'account_code': account_code,
                      'config': asdict(config), 'policy': policy_name, 'schedule':
                      None if schedule is None else {str(k): v for k, v in schedule.items()}, 'ledger': ledger})
        marker = target/'run_identity.json'
        if args.resume and marker.exists() and json.loads(marker.read_text())['key'] == key:
            import pandas as pd
            curve = pd.read_parquet(target/'equity_hourly.parquet')
            print('reuse completed', name, flush=True)
        else:
            print('replay', name, flush=True); started = time.monotonic()
            policy = make_policy(data, forecasts, config, policy_name, schedule) if policy_name else None
            curve, audit = replay(data, policy, out=target, initial_equity=config.initial_equity,
                                  cost_multiplier=config.cost_multiplier, ledger=ledger)
            write_json(marker, {'key': key, 'seconds': time.monotonic()-started, **audit})
            print('finished', name, round(time.monotonic()-started, 2), flush=True)
        curves[name] = curve
        return curve
    run('B0_baseline', ledger=True)
    for name in SPECS:
        run(name, policy_name=name, ledger=name == 'B5_full')
    candidates = {cfg.gamma: curves['B5_full']}
    for gamma in [cfg.gamma/4, cfg.gamma*4]:
        candidates[gamma] = run('full_gamma_'+str(gamma), replace(cfg, gamma=gamma), 'B5_full')
    schedule, splits = select_schedule(candidates, cfg, data.times[-1])
    run('walk_forward', policy_name='B5_full', schedule=schedule, ledger=True)
    if not args.skip_stress:
        stress = replace(cfg, cost_multiplier=1.5)
        run('stress_12bp_baseline', stress)
        run('stress_12bp_optimizer', stress, 'B5_full', schedule)
        small = replace(cfg, initial_equity=1000.)
        run('stress_1000usdt_baseline', small)
        run('stress_1000usdt_optimizer', small, 'B5_full', schedule)
    report = create_report(curves, cfg, data, out, splits)
    print(json.dumps({'status': 'complete', 'report': str(out/'optimizer_report.html'),
        'models': len(curves), 'evaluation_start': report['evaluation_start'],
        'evaluation_end_exclusive': report['evaluation_end_exclusive'], 'real_orders': False}), flush=True)


if __name__ == '__main__':
    main()
