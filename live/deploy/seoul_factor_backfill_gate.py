"""Reserve Seoul capture and midnight signal capacity before archive backfill."""
from datetime import datetime
from pathlib import Path
from data_center.paths import BASE
import shutil
import subprocess
from zoneinfo import ZoneInfo


def main(skip_unit=None):
    hour = datetime.now(ZoneInfo('Asia/Shanghai')).hour
    if 6 <= hour < 10:
        return 1
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        if key == 'MemAvailable':
            memory[key] = int(value.split()[0]) * 1024
    if memory.get('MemAvailable', 0) < 4 * 1024**3:
        return 1
    if shutil.disk_usage(str(BASE/'data')).free < 20 * 1024**3:
        return 1
    for unit in ('select-coin-seoul-minute-producer.service',
                 'select-coin-seoul-daily-producer.service',
                 'select-coin-seoul-archive-gap-repair.service',
                 'select-coin-seoul-lagged-backfill.service'):
        if unit == skip_unit:
            continue
        result = subprocess.run(
            ['systemctl', 'show', '--property=ActiveState', '--value', unit],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0 or result.stdout.strip() not in ('inactive', 'failed'):
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
