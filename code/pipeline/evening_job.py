"""The v2 evening tick (Redesign v2, Phase 4): data, then the paper portfolio.

Called every 15 minutes by the scheduler; each call is cheap when there is
nothing to do. From 18:30 IST on a trading day (before that: the previous
trading day) it brings the v2 database up to the latest session, one session
at a time in order (a PC that was off catches up):
  1. pipeline.daily.run  - NSE's file, Kite as the backup (latest session only:
     Kite's quotes are today's prices);
  2. strategy.live.evening - the paper portfolio, only on NSE's prices; if a
     session still has only Kite's prices at 09:00 the next morning, it runs on
     those (Kite's rows are replaced when NSE's file arrives, the fills stay);
  3. strategy.experiments.update - the Experiments page's pretend portfolios,
     on NSE's prices only (never the book).
Before that, from 09:20 on a trading day, the morning fill: trades waiting for
today's open are made at Kite's opening prices (strategy.live.morning); the
evening checks them against NSE's file. Without Kite they wait for the evening.
"""
from __future__ import annotations

import json
import logging
import threading
import time as clock
from datetime import date, datetime, time, timedelta

logger = logging.getLogger("trader.pipeline")
READY_AT = time(18, 30)          # NSE's daily file is normally out by then
KITE_DEADLINE = time(9, 0)       # next morning: run the paper book on Kite's prices
MAX_CATCH_UP = 15                # sessions processed in one tick
MORNING_AT = time(9, 20)         # the open is settled (pre-open auction ends 09:08, trading from 09:15)
_RUNNING = threading.Lock()      # one update at a time (scheduler or "Update now")
_SAID: set = set()               # morning-fill messages already logged (once a day each)


def latest_session(now: datetime, *, is_trading_day, prev_trading_day) -> date:
    today = now.date()
    if is_trading_day(today) and now.time() >= READY_AT:
        return today
    return prev_trading_day(today)


def tick(conn, client, now: datetime, *, kite=None, calendar=None, log=logger.info, news: bool = False) -> list:
    from pipeline import daily
    from strategy import live
    if calendar is None:
        from core import market_calendar as calendar
    if news:            # red-flag company announcements: every tick, all day (a failure never stops the update)
        from pipeline import news as nw
        try:
            nw.check(conn, client, now, calendar)
        except Exception as exc:
            log(f"announcements not read: {exc}")
        from pipeline import media_news       # headlines to read (never trades), at most hourly
        try:
            media_news.refresh(conn, now)
        except Exception as exc:
            log(f"Google News not read: {exc}")
    if kite is not None and hasattr(kite, "day_opens"):
        try:
            morning_fill(conn, now, kite=kite, calendar=calendar, log=log)
        except Exception as exc:
            if (now.date(), "failed") not in _SAID:
                _SAID.add((now.date(), "failed"))
                log(f"morning fill failed, the trades wait for the evening: {exc}")
    target = latest_session(now, is_trading_day=calendar.is_trading_day,
                            prev_trading_day=calendar.prev_trading_day)
    # 1. data first: every session after the newest NSE file in the database
    #    up to the target (14 Oct: the database ended 5 Oct; the ranking needs
    #    the days in between).
    last_data = conn.execute("SELECT max(date) FROM prices WHERE source = 'nse'").fetchone()[0]
    if last_data:
        d, filled = date.fromisoformat(last_data), 0
        while filled < 2 * MAX_CATCH_UP:
            d = calendar.next_trading_day(d)
            if d >= target:        # the target session is handled below with the book
                break
            if not conn.execute("SELECT 1 FROM prices WHERE date = ? LIMIT 1", (d.isoformat(),)).fetchone():
                daily.run(conn, client, d, kite=kite, log=log)      # (2 Nov 2026) Kite's daily history
            filled += 1
    # 1b. NSE sectors for held and screened stocks not on the list (once a day; never stops the update)
    try:
        fill_sectors(conn, client, now, calendar=calendar, log=log)
    except Exception as exc:
        log(f"sectors not looked up: {exc}")
    # 1c. NSE's holiday list, once a week (the calendar package stops at 2026)
    try:
        fill_holidays(conn, client, now, log=log)
    except Exception as exc:
        log(f"NSE holiday list not read: {exc}")
    # 1c2. NSE's list of shares, once a week: new ETFs and renamed ones stay out of the ranking
    try:
        refresh_equity_list(conn, client, now, log=log)
    except Exception as exc:
        log(f"NSE's list of shares not read: {exc}")
    # 1d. a past day whose NSE file was never read (once a week)
    try:
        repair_gaps(conn, client, now, log=log)
    except Exception as exc:
        log(f"gaps not repaired: {exc}")
    # 2. the paper book, session by session
    last_done = conn.execute("SELECT max(date) FROM paper_values").fetchone()[0]
    if last_done is None:          # a new book starts on the latest session
        sessions = [target]
    else:
        sessions, d = [], date.fromisoformat(last_done)
        while len(sessions) < MAX_CATCH_UP:
            d = calendar.next_trading_day(d)
            if d > target:
                break
            sessions.append(d)
    done = []
    for day in sessions:
        has_nse = conn.execute("SELECT 1 FROM prices WHERE date = ? AND source = 'nse' LIMIT 1",
                               (day.isoformat(),)).fetchone()
        if not has_nse and _backup_recently_tried(conn, day, now) and _nse_file_out(client, day):
            _TRIED.pop(day, None)            # NSE's file is out: read it now, not in up to an hour
        if not has_nse and not _backup_recently_tried(conn, day, now):
            # (2 Nov 2026) the backup for every session: today's from Kite's quotes, earlier
            # ones from Kite's daily history (NSE's website was unreachable for days)
            _TRIED[day] = now
            out = daily.run(conn, client, day, kite=kite, log=log)
            if out.get("prices") == "market closed":
                done.append((day.isoformat(), "market closed"))
                continue
            has_nse = conn.execute("SELECT 1 FROM prices WHERE date = ? AND source = 'nse' LIMIT 1",
                                   (day.isoformat(),)).fetchone()
        deadline = datetime.combine(calendar.next_trading_day(day), KITE_DEADLINE)
        has_any = conn.execute("SELECT 1 FROM prices WHERE date = ? LIMIT 1", (day.isoformat(),)).fetchone()
        if not has_nse and (not has_any or now < deadline):
            done.append((day.isoformat(), "waiting for NSE's file"))
            break                  # later sessions wait too: the book moves in order
        if has_nse and not conn.execute("SELECT 1 FROM index_prices WHERE date = ? LIMIT 1",
                                        (day.isoformat(),)).fetchone():
            # (28 Oct 2026 review) the update stopped after the prices were saved: NIFTY, the actions
            # list and the screen are fetched again (hourly at most)
            last = _FINISH_TRIED.get(day)
            if last is None or now - last >= BACKUP_RETRY:
                _FINISH_TRIED[day] = now
                try:
                    daily.run(conn, client, day, kite=None, log=log)
                except Exception as exc:                          # noqa: BLE001
                    log(f"v2 evening {day}: finishing the day's data failed: {exc}")
        if not daily.has_screen(conn, day):
            # (22 Oct 2026) prices in, screen missing: never decide without it; a rebuild that
            # finds nothing (too few prices) is tried again hourly, not every 15 minutes
            last = _SCREEN_TRIED.get(day)
            if last is None or now - last >= BACKUP_RETRY:
                _SCREEN_TRIED[day] = now
                daily.rebuild_screen(conn, day, log=log)
            if not daily.has_screen(conn, day):
                if last is None:
                    log(f"v2 evening {day}: no screen could be made from the day's prices; waiting for NSE's file")
                done.append((day.isoformat(), "waiting for the day's screen"))
                break
        started = clock.strftime("%Y-%m-%d %H:%M:%S")
        try:
            out = live.evening(conn, day, next_session=calendar.next_trading_day)
        except Exception as exc:
            _log_job(conn, "paper_book", started, "error", {"day": day.isoformat(), "error": str(exc)[:500]})
            raise
        if out.get("status") != "already done":
            _log_job(conn, "paper_book", started, out.get("status"), out)
        done.append((day.isoformat(), out.get("status")))
        log(f"v2 evening {day}: {out}")
        if out.get("status") not in ("ok", "already done"):
            break                  # the book moves in order: a day not done holds the later ones
    # 3. experiments: pretend portfolios beside the book (strategy/experiments.py);
    #    they never touch the book, and a failure never stops the update
    try:
        run_experiments(conn, calendar=calendar, log=log)
    except Exception as exc:
        log(f"experiments not updated: {exc}")
    return done


def run_experiments(conn, *, calendar, log=logger.info):
    """Once per new session with NSE's prices: replay the experiments."""
    from strategy import experiments
    started = clock.strftime("%Y-%m-%d %H:%M:%S")
    try:
        out = experiments.update(conn, next_session=calendar.next_trading_day, log=log)
    except Exception as exc:
        _log_job(conn, "experiments", started, "error", {"error": str(exc)[:500]})
        raise
    if out.get("status") == "ok":
        _log_job(conn, "experiments", started, "ok", out)
        log(f"experiments {out['for']}: {out['values']}")
    return out


_TRIED: dict = {}                # day -> when the data job last ran for it without NSE's file
_SCREEN_TRIED: dict = {}         # day -> when its missing screen was last rebuilt
_FINISH_TRIED: dict = {}         # day -> last re-run of a day whose prices are in but NIFTY is not
BACKUP_RETRY = timedelta(minutes=60)


def _nse_file_out(client, day: date) -> bool:
    """Is NSE's file for ``day`` there now? (one download, without Kite's quotes)"""
    if client is None:
        return False
    from pipeline import nse_daily
    try:
        nse_daily.read_bhav(client, day)
        return True
    except Exception:
        return False


def _backup_recently_tried(conn, day: date, now: datetime) -> bool:
    """(22 Oct 2026) While NSE's file is missing, Kite's backup rows are in and
    the data job ran within the hour, it is not run again every 15 minutes
    (each run quotes ~2,000 stocks and rebuilds the screen)."""
    for cache in (_TRIED, _SCREEN_TRIED):
        for old in [d for d in list(cache) if d < day - timedelta(days=10)]:
            cache.pop(old, None)
    last = _TRIED.get(day)
    if last is None or now - last >= BACKUP_RETRY:
        return False
    return conn.execute("SELECT 1 FROM prices WHERE date = ? AND source = 'kite' LIMIT 1",
                        (day.isoformat(),)).fetchone() is not None


def morning_fill(conn, now: datetime, *, kite, calendar, log=logger.info):
    """From 09:20 on a trading day: the trades waiting for today's open at
    Kite's opening prices. Returns live.morning's summary, or None."""
    from strategy import live
    today = now.date()
    if not calendar.is_trading_day(today) or now.time() < MORNING_AT:
        return None
    prev = calendar.prev_trading_day(today)
    symbols = live.morning_symbols(conn, today, prev)
    if not symbols:
        return None

    for old in [k for k in _SAID.copy() if k[0] != today]:     # a copy: the stop-watch thread may add meanwhile
        _SAID.discard(old)

    def say(kind, text):
        if (today, kind) not in _SAID:
            _SAID.add((today, kind))
            log(text)

    try:
        opens = kite.day_opens(symbols, today)
    except Exception as exc:
        say("kite", f"morning fill {today}: Kite not available ({exc}); the trades wait for the evening's NSE file")
        return None
    started = clock.strftime("%Y-%m-%d %H:%M:%S")
    out = live.morning(conn, today, opens, prev)
    if out["status"] == "ok":
        _log_job(conn, "morning_fill", started, "ok", out)
        log(f"morning fill {today}: {out}")
    elif out["status"] == "left for the evening":
        say("left", f"morning fill {today}: left for the evening - {out.get('why')}")
    return out


MARKET_OPEN, MARKET_CLOSE = time(9, 15), time(15, 30)


def fill_sectors(conn, client, now: datetime, *, calendar, log=logger.info):
    """Once a day, from READY_AT: sectors for stocks still missing (pipeline.sectors)."""
    from config import BASE_DIR
    from pathlib import Path
    from pipeline import sectors
    from store import db as store
    today = now.date()
    if now.time() < READY_AT or store.get_meta(conn, "sectors_checked_on") == today.isoformat():
        return None
    store.set_meta(conn, "sectors_checked_on", today.isoformat())
    conn.commit()
    return sectors.fill_missing(conn, client, Path(BASE_DIR) / "data" / "nse_research" / "industries.csv", today,
                                log=log)


HOLIDAYS_EVERY = timedelta(days=7)


def fill_holidays(conn, client, now: datetime, *, log=logger.info):
    """Once a week: NSE's trading holidays into data/nse_holidays.json (core.market_calendar)."""
    from core import market_calendar
    from store import db as store
    last = store.get_meta(conn, "holidays_checked_on")
    if last and now.date() - date.fromisoformat(last) < HOLIDAYS_EVERY:
        return None
    if client is None or not hasattr(client, "trading_holidays"):
        return None
    store.set_meta(conn, "holidays_checked_on", now.date().isoformat())   # a failure waits a week too
    conn.commit()
    days = client.trading_holidays()
    new = market_calendar.add_holidays({d: f"NSE: {why}" for d, why in days.items()})
    if new:
        log(f"NSE holidays: {new} added ({min(days)} .. {max(days)})")
    return new


EQUITY_LIST_EVERY = timedelta(days=7)
EQUITY_LIST_MIN = 1000          # fewer shares on NSE's list: a broken download, nothing is changed


def refresh_equity_list(conn, client, now: datetime, *, log=logger.info):
    """Once a week: NSE's list of shares (EQUITY_L). A symbol trading now but not on it
    is an ETF, a bond or a rights entitlement and is kept out of the ranking (3 Nov 2026:
    SBI renamed all its ETFs on 30 Oct, and every new symbol had counted as a share).
    A stock the book holds is never reclassified."""
    from pipeline import nse_daily
    from store import db as store
    last = store.get_meta(conn, "equity_list_on")
    if (last and now.date() - date.fromisoformat(last) < EQUITY_LIST_EVERY) or client is None \
            or not hasattr(client, "equity_list"):
        return None
    store.set_meta(conn, "equity_list_on", now.date().isoformat())       # a failure waits a week too
    conn.commit()
    equity, as_of = client.equity_list(now.date(), max_age_days=7)
    if as_of != now.date():
        log(f"NSE's list of shares not downloaded (only a copy from {as_of}): not used")
        return None                       # (3 Nov 2026 review) an old copy would mark a new listing 'other'
    if len(equity) < EQUITY_LIST_MIN:
        log(f"NSE's list of shares has only {len(equity)} rows: not used")
        return None
    held = {r[0] for r in conn.execute("SELECT symbol FROM paper_positions")}
    before = {r[0] for r in conn.execute("SELECT symbol FROM instruments WHERE kind != 'equity'")}
    known = {r[0] for r in conn.execute("SELECT symbol FROM instruments")}
    nse_daily.write_equity_list(conn, {s: v for s, v in equity.items() if s in known})
    # (3 Nov 2026 review) "trading now" = on NSE's newest file: a Kite backup day lists only shares
    last_day = conn.execute("SELECT max(date) FROM prices WHERE source = 'nse'").fetchone()[0]
    if not last_day:
        return None
    still = {r[0] for r in conn.execute("SELECT symbol FROM prices WHERE date = ? AND source = 'nse'", (last_day,))}
    others = sorted(still - set(equity) - held)
    conn.executemany("UPDATE instruments SET kind = 'other' WHERE symbol = ?", [(x,) for x in others])
    conn.commit()
    now_other = {r[0] for r in conn.execute("SELECT symbol FROM instruments WHERE kind != 'equity'")}
    if now_other - before:
        log(f"NSE's list of shares: {len(now_other - before)} symbols marked as not shares "
            f"({', '.join(sorted(now_other - before)[:8])} ...)")
    return len(now_other - before)


GAPS_EVERY = timedelta(days=7)
GAP_TRIES = 3
GAPS_PER_RUN = 5               # (3 Nov 2026) days read again per weekly run (Kite-only days count)


def repair_gaps(conn, client, now: datetime, *, log=logger.info) -> list:
    """Once a week: a past trading day with NIFTY's close but no stock prices (the
    NSE file was never read - 8 Aug 2022 was such a day) is read again from NSE,
    at most two days a week and three tries per day (29 Oct 2026). The live book
    never needs these days; the research and the ranking do. (3 Nov 2026 review) A day
    with only Kite's backup rows counts too: NSE's file replaces them once it is read."""
    from pipeline import daily
    from store import db as store
    last = store.get_meta(conn, "gaps_checked_on")
    if (last and now.date() - date.fromisoformat(last) < GAPS_EVERY) or client is None:
        return []
    store.set_meta(conn, "gaps_checked_on", now.date().isoformat())
    conn.commit()
    tries = json.loads(store.get_meta(conn, "gap_tries") or "{}")
    gaps = [d for (d,) in conn.execute(
        "SELECT i.date FROM index_prices i WHERE i.index_key = 'NIFTY 50' AND i.date >= '2019-10-01' AND i.date < ? "
        "AND NOT EXISTS (SELECT 1 FROM prices p WHERE p.date = i.date AND p.source = 'nse') ORDER BY i.date",
        ((now.date() - timedelta(days=7)).isoformat(),)) if tries.get(d, 0) < GAP_TRIES]
    done = []
    for d in gaps[:GAPS_PER_RUN]:
        tries[d] = tries.get(d, 0) + 1
        store.set_meta(conn, "gap_tries", json.dumps(tries))
        conn.commit()
        try:
            out = daily.run(conn, client, date.fromisoformat(d), kite=None, log=log)
        except Exception as exc:                                  # noqa: BLE001
            out = {"prices": f"failed: {exc}"}
        done.append((d, out.get("prices")))
        log(f"gap {d}: {out.get('prices')}")
    return done


def stop_watch(conn, now: datetime, *, kite, calendar, log=logger.info) -> list:
    """Every minute in market hours: the 25% stop at Kite's live prices
    (strategy.live.check_stops_live). Returns the sales made."""
    from strategy import live
    today = now.date()
    if not calendar.is_trading_day(today) or not (MARKET_OPEN <= now.time() <= MARKET_CLOSE):
        return []
    rules = live.book_rules(conn)
    held = [r[0] for r in conn.execute("SELECT symbol FROM paper_positions")]
    if not held or not rules or not rules.stop_loss or not rules.stop_intraday or not enabled(conn):
        return []
    try:
        prices = kite.traded_prices(held, today)   # traded today only; Kite not logged in: the evening check backs up
    except Exception as exc:
        if (today, "stop_kite") not in _SAID:
            _SAID.add((today, "stop_kite"))
            log(f"stop during the day: Kite not available ({exc}); the evening's close check is the backup")
        return []
    if not prices:
        return []
    sold = live.check_stops_live(conn, today, prices, calendar.prev_trading_day(today))
    for s in sold:
        log(f"STOP during the day: sold {s['symbol']} {s['shares']:.2f} shares at {s['price']:.2f}, "
            f"received {s['received']:.0f}")
        _log_job(conn, "stop_live", now.strftime("%Y-%m-%d %H:%M:%S"), "ok", s)
    return sold


def scheduled_stop_watch():
    """Entry point for the scheduler's one-minute job."""
    from pipeline import kite_source
    from store import db as store
    from core import market_calendar
    from core.timeutil import IST
    now = datetime.now(IST).replace(tzinfo=None)
    if not market_calendar.is_trading_day(now.date()) or not (MARKET_OPEN <= now.time() <= MARKET_CLOSE):
        return []                                  # outside market hours: not even the database is opened
    path = store.default_path()
    if not path.exists():
        return []
    conn = store.connect(path)
    try:
        return stop_watch(conn, now, kite=kite_source, calendar=market_calendar)
    finally:
        conn.close()


def _log_job(conn, job, started, status, detail) -> None:
    conn.execute("INSERT INTO job_log (job, started_at, finished_at, status, detail) VALUES (?, ?, ?, ?, ?)",
                 (job, started, clock.strftime("%Y-%m-%d %H:%M:%S"), status, json.dumps(detail, default=str)))
    conn.commit()


def enabled(conn) -> bool:
    """The evening update is on unless switched off in Settings (or in .env)."""
    from config import Config
    from store import db as store
    return bool(getattr(Config, "V2_EVENING_ENABLED", True)) and store.get_meta(conn, "evening_enabled", "1") == "1"


def running() -> bool:
    return _RUNNING.locked()


def scheduled_tick(*, force: bool = False):
    """Entry point for the scheduler and the "Update now" button. Returns the
    sessions handled, or a reason it did not run."""
    from config import BASE_DIR, Config
    from pathlib import Path
    from pipeline import kite_source
    from pipeline.nse_eod import NseEodClient
    from store import db as store
    path = store.default_path()
    if not path.exists():
        logger.info("v2 evening: database %s not built yet (scripts\\build_v2_db.py)", path)
        return "no database"
    if not _RUNNING.acquire(blocking=False):
        return "already running"
    try:
        conn = store.connect(path)
        client = None
        try:
            if not force and not enabled(conn):
                return "switched off"
            from core.timeutil import IST
            now = datetime.now(IST).replace(tzinfo=None)
            client = NseEodClient(Path(BASE_DIR) / "data" / "nse_research", timeout=Config.NSE_EOD_TIMEOUT_S,
                                  delay_s=Config.NSE_EOD_DOWNLOAD_DELAY_S)
            return tick(conn, client, now, kite=kite_source, news=True)
        finally:
            conn.close()
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
    finally:
        _RUNNING.release()
