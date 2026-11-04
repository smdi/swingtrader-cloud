"""Centralised time helpers.

IST (India) is the app's clock: market hours, "today", the evening update and
the dates in the database are IST. The UTC helpers remain for code that needs
an absolute time.
"""
from datetime import datetime, timezone, date, timedelta

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))


def now_utc() -> datetime:
    """Naive UTC 'now' — the standard for all DB writes and comparisons."""
    return datetime.now(UTC).replace(tzinfo=None)


def utc_now_aware() -> datetime:
    return datetime.now(UTC)


def ist_now() -> datetime:
    """Aware IST 'now' for logs/display only."""
    return datetime.now(UTC).astimezone(IST)


def ist_today() -> date:
    """Trading-relevant 'today' (IST date). Use instead of local date.today()."""
    return ist_now().date()


def to_naive(dt) -> datetime:
    """Convert any datetime to naive UTC (safe for None)."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def days_held(entry, exit_) -> int:
    """Whole calendar days between entry and exit (naive-safe)."""
    entry, exit_ = to_naive(entry), to_naive(exit_)
    if entry is None or exit_ is None:
        return 0
    return max((exit_ - entry).days, 0)


def format_eta(seconds) -> str:
    """Plain-words estimate of time left for progress messages."""
    if seconds is None or seconds != seconds or seconds < 0:
        return ""
    minutes = int(round(seconds / 60.0))
    if minutes < 1:
        return "under a minute left"
    if minutes < 60:
        return f"about {minutes} min left"
    hours, minutes = divmod(minutes, 60)
    return f"about {hours} h {minutes:02d} min left"
