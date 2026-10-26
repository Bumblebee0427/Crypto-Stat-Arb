"""Keep the candidate archive worker clear of production backfill and 08:00."""
import subprocess
import shutil
from pathlib import Path

from deploy.seoul_factor_backfill_gate import main as base_gate


def main():
    if shutil.disk_usage(Path(__file__).resolve().parents[1]/'data').free < 35*1024**3:
        return 1
    result = subprocess.run(
        ['systemctl', 'show', '--property=ActiveState', '--value',
         'select-coin-seoul-base-factors.service'],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or result.stdout.strip() not in ('inactive', 'failed'):
        return 1
    return base_gate(skip_unit='select-coin-seoul-lagged-backfill.service')


if __name__ == '__main__':
    raise SystemExit(main())
