"""UTC boundary policy shared by publisher, optimizer and executor."""
from datetime import timedelta, timezone

LIVE_WINDOW_MINUTES = 25  # Last publication/decision; leave five minutes to submit.
OPENING_WINDOW_MINUTES = 30
OPTIMIZER_WAIT_SECONDS = 120


def boundary_deadline(now):
    now = now.astimezone(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=LIVE_WINDOW_MINUTES)
