"""NSE trading calendar.

Wraps pandas-market-calendars' XNSE calendar. Gracefully falls back to a
weekday + hardcoded NSE holiday approximation if the package is unavailable.

All functions operate on `datetime.date` (or pandas Timestamps).
"""
from datetime import date, timedelta

# Fixed NSE holidays (fallback only). Movable ones are approximated; the real
# calendar from pandas-market-calendars is authoritative when installed.
FALLBACK_HOLIDAYS = {
    (1, 26),   # Republic Day
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
    (12, 25),  # Christmas
    (1, 1),    # New Year
}


def _pmc_session():
    try:
        import pandas_market_calendars as pmc
        return pmc.get_calendar("XNSE")
    except Exception:
        return None


_cal_cache = {}


def _calendar():
    key = "default"
    if key not in _cal_cache:
        _cal_cache[key] = _pmc_session()
    return _cal_cache[key]


# NSE holidays beyond what the calendar package knows (22 Oct 2026: its XNSE
# list stops at 2026, so every 2027 weekday counted as a trading day). Kept in
# data/nse_holidays.json: NSE's own holiday list (pipeline.evening_job.fill_holidays,
# weekly) and days NSE published no file for but did for the next day (pipeline.daily).
_extra = {"path": None, "mtime": None, "days": frozenset(), "checked": 0.0}


def holidays_file():
    from pathlib import Path
    try:
        from config import BASE_DIR
    except Exception:
        return None
    return Path(BASE_DIR) / "data" / "nse_holidays.json"


def extra_holidays() -> frozenset:
    """The dates in data/nse_holidays.json (re-read at most once a minute when the file changes)."""
    import json
    import os
    import time
    now = time.monotonic()
    if _extra["path"] is None:
        _extra["path"] = holidays_file()
    path = _extra["path"]
    if path is None or now - _extra["checked"] < 60 and _extra["mtime"] is not None:
        return _extra["days"]
    _extra["checked"] = now
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _extra["mtime"], _extra["days"] = -1, frozenset()
        return _extra["days"]
    if mtime != _extra["mtime"]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _extra["days"] = frozenset(date.fromisoformat(k) for k in data)
        except (OSError, ValueError):
            pass                                  # keep what was read before
        _extra["mtime"] = mtime
    return _extra["days"]


def known_holidays() -> dict:
    """``{date: why}`` from data/nse_holidays.json."""
    import json
    path = holidays_file()
    try:
        return {date.fromisoformat(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def add_holidays(days: dict) -> int:
    """Merge ``{date: why}`` into data/nse_holidays.json; returns how many were new."""
    import json
    import os
    path = holidays_file()
    if path is None:
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    new = {d.isoformat(): why for d, why in days.items() if d.isoformat() not in data}
    if not new:
        return 0
    data.update(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(dict(sorted(data.items())), indent=1), encoding="utf-8")
    os.replace(tmp, path)
    _extra["checked"], _extra["mtime"] = 0.0, None          # read again at once
    return len(new)


def is_trading_day(d) -> bool:
    """True if `d` is a scheduled NSE trading session."""
    d = _as_date(d)
    if d in extra_holidays():
        return False
    cal = _calendar()
    if cal is not None:
        try:
            # 1-day window is a cheap membership check by index date
            sch = cal.schedule(start_date=d - timedelta(days=1),
                               end_date=d + timedelta(days=1))
            return d in sch.index.date
        except Exception:
            pass
    # fallback: weekdays minus fixed holidays
    if d.weekday() >= 5:
        return False
    return (d.month, d.day) not in FALLBACK_HOLIDAYS


def next_trading_day(d) -> date:
    day = _as_date(d) + timedelta(days=1)
    while not is_trading_day(day):
        day = day + timedelta(days=1)
    return day


def prev_trading_day(d) -> date:
    day = _as_date(d) - timedelta(days=1)
    while not is_trading_day(day):
        day = day - timedelta(days=1)
    return day


def trading_days_between(start, end) -> int:
    """Number of trading sessions in [start, end] inclusive (>= 1 if start is a
    trading day). If the calendar is unavailable, approximates business days."""
    start, end = _as_date(start), _as_date(end)
    if end < start:
        return 0
    n = 0
    day = start
    while day <= end:
        if is_trading_day(day):
            n += 1
        day = day + timedelta(days=1)
    return n


def add_trading_days(d, n: int) -> date:
    """Return the date `n` trading sessions after `d` (n >= 0)."""
    day = _as_date(d)
    count = 0
    while count < n:
        day = day + timedelta(days=1)
        if is_trading_day(day):
            count += 1
    return day


def _as_date(d):
    return d.date() if hasattr(d, "date") else d