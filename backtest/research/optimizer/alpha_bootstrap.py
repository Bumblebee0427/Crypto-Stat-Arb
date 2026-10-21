"""Load only an explicitly validated, hash-bound historical alpha release."""
from pathlib import Path
import json
import pandas as pd

from .contracts import digest, file_hash, ResearchConfig
from .v3_history import temporal_panel_check, V3


def alpha_policy():
    cfg = ResearchConfig()
    return {'train_days': cfg.train_days, 'minimum_side_days': cfg.train_days-3,
        'ridge_penalty': cfg.ridge_penalty, 'alpha_shrink': cfg.alpha_shrink,
        'features': ['main_rank', 'small_rank'], 'refit': 'UTC_month_start',
        'formula_version': V3}


def validated_history(root, boundary):
    root = Path(root); release_path = root/'alpha_bootstrap/release.json'
    if not release_path.exists(): return None, None
    return validate_release(release_path.parent, boundary)


def validate_release(folder, boundary):
    folder = Path(folder); release_path = folder/'release.json'
    release = json.loads(release_path.read_text())
    body = {k: v for k, v in release.items() if k != 'release_id'}
    if release.get('release_id') != digest(body) or release.get('policy') != alpha_policy():
        raise ValueError('invalid alpha bootstrap release or changed policy')
    panel_path, report_path = folder/'historical_panel.parquet', folder/'validation.json'
    if file_hash(panel_path) != release['panel_sha256'] or file_hash(report_path) != release['validation_sha256']:
        raise ValueError('alpha bootstrap data or validation changed after approval')
    report = json.loads(report_path.read_text())
    if report.get('alpha_policy') != release['policy']:
        raise ValueError('alpha validation policy differs from the live policy')
    if report.get('panel_sha256') != release['panel_sha256']:
        raise ValueError('alpha validation refers to a different training panel')
    if report.get('accepted') is not True or not report.get('checks') or not all(report['checks'].values()):
        raise ValueError('alpha historical walk-forward validation did not pass')
    now = pd.Timestamp(boundary)
    if now.tzinfo: now = now.tz_convert('UTC').tz_localize(None)
    if pd.Timestamp(report['validation_label_available_at']) > now:
        raise ValueError('future validation outcomes cannot authorize alpha')
    panel = temporal_panel_check(pd.read_parquet(panel_path))
    return panel, {'release_id': release['release_id'], 'validation_end': report['test_end_inclusive'],
                   'historical_rows': len(panel), 'historical_days': panel['asof'].nunique()}
