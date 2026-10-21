#!/usr/bin/env python3
"""Inspect public files or staged blobs without displaying possible secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'access token': re.compile(r'\b(?:gh[pousr]_|github_pat_|xox[baprs]-|cdg_)[A-Za-z0-9_-]{16,}'),
    'cloud access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'credential URL': re.compile(r'[?&](?:token|access_token|key)=[A-Za-z0-9_-]{12,}', re.I),
    'private deployment path': re.compile(r'/(?:home/(?:graham|ubuntu)|data[0-9]|opt/select)[/\s\x27\x22]'),
    'private deployment address': re.compile(r'\b43[.]164[.]\d+[.]\d+\b'),
}
FORBIDDEN_SUFFIXES = {'.pkl', '.pickle', '.parquet', '.npy', '.npz', '.csv', '.log',
                      '.jsonl', '.pem', '.key', '.zip', '.gz', '.pptx', '.pdf', '.html'}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staged', action='store_true', help='Audit the exact Git index contents')
    args = p.parse_args()
    names = git('ls-files', '-z', '--cached') if args.staged else git(
        'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    names = sorted(set(n.decode() for n in names.split(b'\0') if n))
    issues, blobs = [], {}
    for name in names:
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith('.env'):
            issues.append(f'{name}: prohibited file type')
        if any(x in {'data', 'runtime', 'logs', '.venv', 'reports', '__pycache__'} for x in path.parts):
            issues.append(f'{name}: runtime directory')
        if not args.staged and (ROOT/name).is_symlink():
            issues.append(f'{name}: symlink')
            continue
        raw = git('show', ':'+name) if args.staged else (ROOT/name).read_bytes()
        blobs[name] = raw
        if len(raw) > 1_000_000:
            issues.append(f'{name}: exceeds 1 MB public-file limit')
        if path.suffix.lower() == '.png':
            if name != 'results/backtest_equity.png':
                issues.append(f'{name}: unexpected image')
            continue
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            issues.append(f'{name}: unexpected binary')
            continue
        for label, pattern in RULES.items():
            for m in pattern.finditer(text):
                line = text.count('\n', 0, m.start())+1
                issues.append(f'{name}:{line}: {label}')
    manifest = json.loads(blobs.get('docs/source_manifest.json', b'[]'))
    allowlist = json.loads(blobs.get('tools/source_allowlist.json', b'{}'))
    expected = {kind+'/'+n for kind, names_ in allowlist.items() for n in names_}
    if {r['path'] for r in manifest} != expected:
        issues.append('source manifest and allowlist differ')
    for item in manifest:
        raw = blobs.get(item['path'])
        if raw is None or hashlib.sha256(raw).hexdigest() != item['export_sha256']:
            issues.append(item['path']+': source manifest drift; sync or review and update provenance')
    pngs = [n for n in names if n.endswith('.png')]
    if pngs != ['results/backtest_equity.png']:
        issues.append('expected exactly one historical equity figure')
    if issues:
        print('\n'.join(issues))
        return 1
    print(f'PASS: {len(names)} files, {sum(map(len, blobs.values())):,} bytes; '
          f'{len(manifest)} allowlisted source files; one historical figure.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
