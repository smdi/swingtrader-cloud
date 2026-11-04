"""The evening data job (Redesign v2, Phase 2).

For one trading day, in order:
  1. stock prices: NSE's daily file; if it is not published (yet), Kite's
     quotes as a backup, marked source='kite';
  2. any of the last 10 days still from Kite: replaced by NSE's file when it
     is now available;
  3. NIFTY 50: NSE's index file, else Kite;
  4. corporate actions for the surrounding weeks (a failure is logged, not fatal);
  5. the day's screen (live rules).
A day NSE answers with another day's file was a market holiday: nothing is
written. Every run is recorded in job_log.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta

from pipeline import nse_daily, screen as sc

logger = logging.getLogger("trader.pipeline")
RECHECK_DAYS = 10


def _today() -> date:
    return date.today()


HOLIDAY_MATCH = 0.9              # share of stocks whose next-day "previous close" is the close before the day


def _holiday_evidence(conn, client, day: date) -> str | None:
    """Why ``day`` (NSE: no file) was a market holiday, or None. The next
    weekday's file must exist AND give, as each stock's previous close, the
    close of the session before ``day`` - true only if nothing traded on
    ``day`` (a file that is merely late fails this). A year NSE's own holiday
    list covers is never second-guessed."""
    from core import market_calendar
    if any(d.year == day.year for d, why in market_calendar.known_holidays().items() if why.startswith("NSE")):
        return None
    before = conn.execute("SELECT max(date) FROM prices WHERE date < ? AND source = 'nse'",
                          (day.isoformat(),)).fetchone()[0]
    if not before:
        return None
    # the first later weekday that has a file (two unlisted holidays in a row: 28 Oct 2026 review)
    frame, nxt = None, day
    for _ in range(4):
        nxt += timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        if nxt > _today():
            return None
        try:
            frame = nse_daily.read_bhav(client, nxt)
            break
        except Exception:
            frame = None
    if frame is None:
        return None
    if frame is None or frame.empty or "prev_close" not in frame:
        return None
    closes = dict(conn.execute("SELECT symbol, close FROM prices WHERE date = ? AND source = 'nse'", (before,)))
    pairs = [(float(p), closes[s]) for s, p in zip(frame["symbol"], frame["prev_close"]) if s in closes and p == p]
    if len(pairs) < 50:
        return None
    same = sum(1 for p, c in pairs if abs(p - c) <= 0.011)
    if same / len(pairs) < HOLIDAY_MATCH:
        return None
    return f"no NSE file; {nxt}'s file gives {before}'s closes as previous closes ({same} of {len(pairs)} stocks)"


def run(conn, client, day: date, *, kite=None, log=logger.info) -> dict:
    from pipeline.nse_eod import NseFileNotPublished, NseFileUnavailable, NseSchemaError
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    summary = {"day": day.isoformat(), "prices": None, "nifty": None, "actions": None, "screen": None,
               "replaced_backup_days": 0}
    status = "ok"
    try:
        # 1. prices
        try:
            frame = nse_daily.read_bhav(client, day)
            summary["prices"] = f"nse:{nse_daily.write_prices(conn, frame, source='nse')}"
            _drop_backup_rows(conn, day)
            conn.commit()                  # (28 Oct 2026) no write lock held over the network calls below
        except NseSchemaError as exc:
            if "is dated" in str(exc):
                # NSE served another day's file: the market was closed. Remembered as a holiday only with
                # the same evidence as a missing file (the next day's file), never on this alone (28 Oct 2026)
                from core import market_calendar
                why = None if not market_calendar.is_trading_day(day) else _holiday_evidence(conn, client, day)
                if market_calendar.is_trading_day(day) and not why:
                    # no proof the market was closed: wait for a correct file - a trading day must
                    # never be skipped on a wrongly dated file alone (28 Oct 2026 review)
                    summary["prices"] = f"waiting: {str(exc)[:120]}"
                    status = "incomplete"
                    return summary
                if why:
                    market_calendar.add_holidays({day: why})
                summary["prices"] = "market closed"
                status = "closed"
                return summary
            raise
        except (NseFileNotPublished, NseFileUnavailable) as exc:
            why = _holiday_evidence(conn, client, day) if isinstance(exc, NseFileNotPublished) else None
            if why:
                # an unlisted holiday (the calendar package knows none after 2026)
                from core import market_calendar
                market_calendar.add_holidays({day: why})
                log(f"{day}: NSE published no file but did for the next day - marked as a market holiday")
                summary["prices"] = "market closed"
                status = "closed"
                return summary
            log(f"NSE file for {day} unavailable ({exc}); using Kite as the backup")
            symbols = _recent_symbols(conn, day)
            try:
                rows = kite.day_rows(symbols, day, check=_closes_before(conn, day)) if kite else None
            except Exception as kexc:                         # noqa: BLE001 - e.g. not logged in yet
                log(f"Kite backup for {day} not available ({str(kexc)[:120]})")
                rows = None                                    # (3 Nov 2026 review) never stops the update
            if rows is None or rows.empty:
                summary["prices"] = "missing (NSE not published, Kite unavailable)"
                status = "incomplete"
            else:
                summary["prices"] = f"kite:{nse_daily.write_prices(conn, rows, source='kite')}"
                status = "backup"
                conn.commit()
        # 2. earlier backup days
        for (old_day,) in conn.execute(
                "SELECT DISTINCT date FROM prices WHERE source = 'kite' AND date >= ? AND date < ?",
                ((day - timedelta(days=RECHECK_DAYS)).isoformat(), day.isoformat())).fetchall():
            try:
                frame = nse_daily.read_bhav(client, date.fromisoformat(old_day))
            except Exception:
                continue
            nse_daily.write_prices(conn, frame, source="nse")
            _drop_backup_rows(conn, date.fromisoformat(old_day))
            conn.commit()
            summary["replaced_backup_days"] += 1
        # 3. NIFTY 50
        try:
            nse_daily.write_index(conn, day, nse_daily.read_index_close(client, day), source="nse")
            summary["nifty"] = "nse"
        except Exception as exc:
            try:
                nse_daily.write_index(conn, day, kite.index_close(day), source="kite")
                summary["nifty"] = "kite"
            except Exception as exc2:
                summary["nifty"] = f"missing ({exc}; Kite: {exc2})"
                status = "incomplete"
        conn.commit()
        # 4. corporate actions
        try:
            events, _ = client.corporate_actions(day - timedelta(days=30), day + timedelta(days=30))
            summary["actions"] = nse_daily.write_actions(conn, events)
            from store import db as store      # the morning fill trusts the list only when this is recent
            if (store.get_meta(conn, "actions_refreshed_on") or "") < day.isoformat():   # never backwards
                store.set_meta(conn, "actions_refreshed_on", day.isoformat())
        except Exception as exc:
            summary["actions"] = f"not refreshed ({exc})"
        conn.commit()
        # 5. screen
        rows = sc.live_screen(conn, day, day)
        summary["screen"] = sc.write_screen(conn, rows, "live", day, day)
        conn.commit()
        return summary
    except Exception as exc:
        status = "error"
        summary["error"] = str(exc)[:500]
        conn.rollback()
        raise
    finally:
        conn.execute("INSERT INTO job_log (job, started_at, finished_at, status, detail) VALUES (?, ?, ?, ?, ?)",
                     ("daily_data", started, time.strftime("%Y-%m-%d %H:%M:%S"), status, json.dumps(summary)))
        conn.commit()


def _drop_backup_rows(conn, day: date) -> None:
    """(22 Oct 2026) Once NSE's file for a day is in, Kite's backup rows for
    stocks NSE did not list (or listed as invalid) go too: NSE's file is the record."""
    conn.execute("DELETE FROM prices WHERE date = ? AND source = 'kite'", (day.isoformat(),))


def has_screen(conn, day: date) -> bool:
    return conn.execute("SELECT 1 FROM screen WHERE date = ? LIMIT 1", (day.isoformat(),)).fetchone() is not None


def rebuild_screen(conn, day: date, *, log=logger.info) -> int:
    """The day's screen when its prices are in but the screen is not (22 Oct 2026:
    the update stopped between the two - the app closed or the PC slept). Commits."""
    rows = sc.live_screen(conn, day, day)
    n = sc.write_screen(conn, rows, "live", day, day)
    conn.commit()
    log(f"screen for {day} rebuilt: {n} stocks (the update had stopped before it)")
    return n


def _closes_before(conn, day: date) -> dict:
    """{symbol: (date, close)}: NSE's last close before ``day`` - the Kite backup checks its
    history against it (Kite's daily candles are adjusted for later splits and bonuses)."""
    # (3 Nov 2026 review) each stock's own last NSE close in the last 30 days, not only those of NSE's last day
    out = {}
    for sym, d, c in conn.execute(
            "SELECT p.symbol, p.date, p.close FROM prices p JOIN (SELECT symbol, max(date) AS d FROM prices "
            "WHERE source = 'nse' AND date < ? AND date >= ? GROUP BY symbol) m ON m.symbol = p.symbol AND m.d = p.date "
            "WHERE p.close > 0", (day.isoformat(), (day - timedelta(days=30)).isoformat())):
        out[sym] = (date.fromisoformat(d), float(c))
    return out


def _recent_symbols(conn, day: date) -> list:
    """Shares that traded in the last two weeks (what Kite is asked for)."""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT p.symbol FROM prices p JOIN instruments i ON i.symbol = p.symbol "
        "WHERE i.kind = 'equity' AND p.date >= ? AND p.date < ?",
        ((day - timedelta(days=14)).isoformat(), day.isoformat()))]
