#!/usr/bin/env python3
"""Copy only reviewed computation source files into this repository.

Review git diff and run tools/check_export.py before committing. No commit,
push, deployment, credential reading, or account-data export is performed.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_assignment(text, name, replacement):
    tree = ast.parse(text)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one assignment to {name}')
    node = matches[0]
    lines = text.splitlines(keepends=True)
    indent = ' ' * node.col_offset
    lines[node.lineno-1:node.end_lineno] = [indent + name + ' = ' + replacement + '\n']
    return ''.join(lines)


def portable(kind, name, text):
    # These recovery helpers inherit portable roots from the minute module.
    if kind == 'live' and name in {'deploy/seoul_lagged_bootstrap_minutes.py', 'deploy/seoul_lagged_gap_watch.py', 'deploy/seoul_lagged_backfill_gate.py'}:
        return text
    if kind == 'live' and name == 'deploy/seoul_factor_backfill_gate.py':
        calls = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == 'shutil.disk_usage']
        if len(calls) != 1:
            raise ValueError('backfill disk gate changed; review portable adapter')
        original = ast.get_source_segment(text, calls[0].args[0])
        return text.replace('import shutil', 'from data_center.paths import BASE\nimport shutil').replace(
            original, "str(BASE/'data')")
    if kind == 'live' and name.startswith('deploy/') and name.endswith('.py'):
        if name.endswith('seoul_lagged_daily_prefix.py'):
            text = replace_assignment(text, 'ROOT', "BASE / 'data/lagged_quantile_v3'")
            text = text.replace('ROOT = BASE', 'from data_center.paths import BASE\n\nROOT = BASE', 1)
        else:
            # Keep module-level names so production test patches still apply.
            text = replace_assignment(text, 'BASE', '_PORTABLE_BASE')
            text = text.replace('BASE = _PORTABLE_BASE',
                                'from data_center.paths import BASE as _PORTABLE_BASE\n\nBASE = _PORTABLE_BASE', 1)
    if kind == 'live' and name == 'tests/test_top50.py':
        text = text.replace('from tests.test_base_signal import snapshot,NOW',
                            'from tests.fixtures import snapshot,NOW')
    if kind == 'live' and name == 'tests/test_quality_tolerance.py':
        # Older fixtures predate the current 10% policy and 599-hour feed.
        # Update fixture inputs only; production decisions are unchanged.
        for old, new in [('row(49,1)', 'row(8,1)'), ('row(48,1)', 'row(1,1)'),
                         ('row(50,1)', 'row(9,1)')]:
            if old not in text:
                raise ValueError('quality test fixture changed; review adapter')
            text = text.replace(old, new)
        text = text.replace("'hours':600,'window_end':end.isoformat()",
            "'hours':599,'window_end':(end-timedelta(hours=1)).isoformat(),"
            "'target_window_end':end.isoformat(),'volume_lag_hours':1")
    if kind == 'backtest' and name == 'research/optimizer/dataset.py':
        text = text.replace('import json\n', 'import json\nimport os\n', 1)
        text = replace_assignment(text, 'DEFAULT_MANIFEST',
            "Path(os.environ.get('SELECT_COIN_BACKTEST_MANIFEST', Path(__file__).resolve().parents[2] / 'data/source_manifest.json'))")
    if kind == 'backtest' and name == 'research/optimizer/funding.py':
        text = text.replace('from __future__ import annotations\n',
                            'from __future__ import annotations\n\nimport os\n', 1)
        text = replace_assignment(text, 'root',
            "Path(os.environ.get('SELECT_COIN_FUNDING_ROOT', Path(__file__).resolve().parents[2] / 'data/fundingRate'))")
    if kind == 'backtest' and name == 'tools/run_optimizer_research.py':
        tree = ast.parse(text)
        guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and 'str(out).startswith' in ast.unparse(n.test)]
        if len(guards) != 1:
            raise ValueError('frozen source guard changed; review adapter')
        node = guards[0]
        lines = text.splitlines(keepends=True)
        lines[node.lineno-1:node.end_lineno] = [
            "    source = json.loads(args.manifest.read_text())\n",
            "    frozen = Path(source['result_folder']).resolve().parents[2]\n",
            "    if out == frozen or frozen in out.parents:\n",
            "        raise ValueError('research output cannot overwrite the frozen source workspace')\n"]
        text = ''.join(lines)
    if kind == 'backtest' and name in {'core/equity.py', 'core/simulator.py'}:
        text = '\n'.join(line.rstrip() for line in text.splitlines())+'\n'
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-source', type=Path, default=ROOT.parent/'select-coin-hft-pro')
    parser.add_argument('--backtest-source', type=Path, default=ROOT.parent/'select-coin-pro')
    args = parser.parse_args()
    allowlist = json.loads((ROOT/'tools/source_allowlist.json').read_text())
    records = []
    # Read and validate the entire allowlist before writing any files.
    prepared = []
    for kind, names in allowlist.items():
        source = getattr(args, kind+'_source').resolve()
        for name in names:
            p = (source/name).resolve()
            if source not in p.parents or (source/name).is_symlink():
                raise ValueError('unsafe allowlist path: '+name)
            raw = p.read_bytes()
            exported = portable(kind, name, raw.decode()).encode()
            prepared.append((ROOT/kind/name, exported))
            records.append({'path': kind+'/'+name,
                'source_sha256': hashlib.sha256(raw).hexdigest(),
                'export_sha256': hashlib.sha256(exported).hexdigest(),
                'portable_adapter': raw != exported})
    for target, content in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (ROOT/'docs/source_manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'Copied {len(records)} allowlisted source files; review git diff before commit.')


if __name__ == '__main__':
    main()
