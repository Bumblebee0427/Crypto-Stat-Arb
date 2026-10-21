#!/usr/bin/env python3
"""Resume reconstruction, validate, report, and install only a passing release."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.optimizer.contracts import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); history = Path(cfg['history'])
    out = Path(cfg['report_out']); out.mkdir(parents=True, exist_ok=True)
    lock = (history/'completion.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def state(phase, **detail):
        write_json(history/'completion.json', {'phase': phase, **detail,
            'updated_at': datetime.now(timezone.utc).isoformat(), 'report': str(out/'report.html')})
    def run(script, args, check=True):
        result = subprocess.run([sys.executable, str(ROOT/'tools'/script), *args])
        if check and result.returncode: raise RuntimeError(script+' exited '+str(result.returncode))
        return result.returncode
    try:
        if not cfg.get('resume_validation',False):
            state('rebuilding_history')
            args = ['--out', str(history), '--market', cfg['market'], '--workers', str(cfg.get('workers',24)), '--exchange-info', cfg['exchange_info']]
            for root in cfg['raw_roots']: args += ['--raw-root', root]
            if cfg.get('repair_symbols'):
                # Existing raw day caches are unaffected by market-input repairs.
                repair_args=['--history',str(history),'--exchange-info',cfg['exchange_info']]
                for minute_root in cfg['minute_roots']:repair_args+=['--minute-root',minute_root]
                for symbol in cfg['repair_symbols']:repair_args+=['--symbol',symbol]
                run('repair_alpha_market_history.py',repair_args)
            run('bootstrap_live_alpha_history.py', args)
            state('preparing_execution_minutes')
            args = ['--history', str(history)]
            for root in cfg['minute_roots']: args += ['--minute-root', root]
            run('prepare_alpha_execution_minutes.py', args)
            state('preparing_risk')
            run('prepare_v3_alpha_validation_risk.py', ['--history', str(history), '--cap-root', cfg['cap_root'],
                '--live-root', cfg['live_root']])
        state('walk_forward_validation')
        code = run('validate_live_alpha_history.py', ['--history', str(history), '--out', str(out),
            '--market', cfg['market'], '--market-tail', cfg['market_tail'], '--settlements', cfg['settlements']], check=False)
        if (out/'validation.json').exists(): run('report_live_alpha_validation.py', ['--out', str(out)])
        if code: raise RuntimeError('walk-forward input or execution audit failed')
        report = json.loads((out/'validation.json').read_text())
        if not report['accepted']:
            state('validation_rejected', checks=report['checks'], orders_submitted=False)
            return
        state('installing_passed_release')
        destination = Path(cfg['live_root'])/'alpha_bootstrap'
        if destination.exists():
            installed = json.loads((destination/'release.json').read_text())
            if installed['panel_sha256'] != report['panel_sha256']:
                raise ValueError('an installed release uses a different training panel; active release retained')
        if not destination.exists():
            run('publish_live_alpha_release.py', ['--history', str(history), '--validation', str(out/'validation.json'),
                '--live-root', cfg['live_root']])
        # The preview exercises the real producer and remote read-only account
        # checks. It cannot deliver an actionable artifact or place an order.
        run('run_live_optimizer.py', ['--config', cfg['live_config'], '--preview', '--skip-refresh'])
        preview = json.loads((Path(cfg['live_root'])/'last_run.json').read_text())
        if preview['alpha_status'] != 'v3_trained': raise ValueError('live preview did not activate trained alpha')
        release = json.loads((destination/'release.json').read_text())
        state('complete', release_id=release['release_id'], preview=preview, orders_submitted=False)
    except Exception as exc:
        state('failed', reason=type(exc).__name__+': '+str(exc)[:300], orders_submitted=False)
        raise


if __name__ == '__main__': main()
