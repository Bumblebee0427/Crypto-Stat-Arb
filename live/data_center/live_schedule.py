"""One UTC deadline shared by signal publication, decision and opening."""
from datetime import datetime, timedelta, timezone

LIVE_WINDOW_MINUTES = 10


def boundary_deadline(now):
    now = now.astimezone(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=LIVE_WINDOW_MINUTES)
