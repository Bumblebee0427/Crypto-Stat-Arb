"""Portable storage root for the exported computation pipeline."""
import os
from pathlib import Path

BASE = Path(os.environ.get('SELECT_COIN_LIVE_ROOT', Path(__file__).resolve().parents[1])).resolve()
