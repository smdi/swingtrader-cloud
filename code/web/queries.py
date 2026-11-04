"""What the pages show, read from the v2 database (plain sqlite3, no Flask).

Every function takes an open connection, so tests run them on a small
database without a web server.
"""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, time, timedelta

import numpy as np

from strategy import strength as st

MARKET_OPEN, MARKET_CLOSE = time(9, 15), time(15, 30)
RANK_WARMUP_DAYS = 420            # calendar days of prices (as strategy.live)

_cache: dict = {}
_cache_lock = threading.Lock()


# ------------------------------------------------------------------ calendar
def _calendar():
    from core import market_calendar
    return market_calendar


def month_end_session(day: date, cal=None) -> date:
    """The next month-end decision day: the last trading session of the month
    of the first session on or after ``day``."""
    cal = cal or _calendar()
    d = day if cal.is_trading_day(day) else cal.next_trading_day(day)
    while True:
        nxt = cal.next_trading_day(d)
        if nxt.month != d.month:
            return d
        d = nxt


def market_open(now: datetime, cal=None) -> bool:
    cal = cal or _calendar()
    return cal.is_trading_day(now.date()) and MARKET_OPEN <= now.time() <= MARKET_CLOSE


# ------------------------------------------------------------------ the book
def book(conn) -> dict | None:
    row = conn.execute("SELECT capital, started_on, rules FROM paper_book WHERE id = 1").fetchone()
    if not row:
        return None
    return {"capital": float(row[0]), "started_on": row[1], "rules": json.loads(row[2])}


def values(conn) -> list:
    return [dict(zip(("date", "value", "cash", "holdings", "nifty", "all_screened"), r)) for r in conn.execute(
        "SELECT date, value, cash, holdings, nifty, all_screened FROM paper_values ORDER BY date")]


def overview(conn, today: date, cal=None) -> dict:
    """The Today page: value, returns against NIFTY and all screened stocks,
    the next decision and what is waiting to be done."""
    cal = cal or _calendar()
    out = {"book": book(conn), "today": today.isoformat()}
    vals = values(conn)
    out["values"] = vals
    if vals:
        first, last = vals[0], vals[-1]
        capital = out["book"]["capital"]
        out["last"] = last
        out["return_pct"] = (last["value"] / capital - 1) * 100
        out["nifty_pct"] = _change(first["nifty"], last["nifty"])
        out["all_screened_pct"] = _change(first["all_screened"], last["all_screened"])
    nxt = month_end_session(today, cal)
    out["next_decision"] = nxt.isoformat()
    out["next_trades"] = cal.next_trading_day(nxt).isoformat()
    out["pending"] = pending_plan(conn)
    out["orders"] = orders(conn)
    out["last_jobs"] = jobs(conn, 6)
    return out


def _change(a, b):
    return (b / a - 1) * 100 if a and b else None


def pending_plan(conn) -> dict | None:
    """The decision waiting for the next open, as the trades it will make: a
    dry run of the same execution (strategy.strength.execute, as the morning or
    evening run) at the last closes - after the sale orders waiting for that
    open, with the sector limit and the cash rule (22 Oct 2026: the old list
    ignored both). The open's prices can still change a buy at the margin."""
    from strategy import live
    row = conn.execute("SELECT id, decided_on, keep, buy_order, scores FROM paper_decisions "
                       "WHERE status = 'pending' ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    rules = live.book_rules(conn) or st.Rules()
    costs = st.Costs.from_config()
    keep, order, scores = json.loads(row[2]), json.loads(row[3]), json.loads(row[4] or "{}")
    held = dict(conn.execute("SELECT symbol, shares FROM paper_positions"))
    # sold at that open before the decision's trades (live.fill_orders 'open' runs first, with overdue close orders)
    book_day = conn.execute("SELECT max(date) FROM paper_values").fetchone()[0]
    nxt = _calendar().next_trading_day(date.fromisoformat(book_day or row[1])).isoformat()
    ordered_out = {r[1] for r in conn.execute(live._DUE_AT_OPEN, (nxt, nxt))}
    symbols = sorted(set(held) | set(keep) | set(order))
    # on the basis of today's share counts (a split already applied to a held stock: 29 Oct 2026 review)
    last = live.adjusted_last_close(conn, date.fromisoformat(book_day or row[1]), symbols) if symbols else {}
    money = live.cash(conn)
    for s in ordered_out & set(held):             # sold first at that open by their orders
        if last.get(s):
            gross = held[s] * last[s] * (1 - costs.slippage)
            money += gross - gross * costs.sell_rate - costs.dp_per_sell
        held.pop(s)
    index = {s: i for i, s in enumerate(symbols)}
    price = np.array([last.get(s, np.nan) for s in symbols], dtype=float)
    book_ = st.Book(money, np.array([held.get(s, 0.0) for s in symbols], dtype=float))
    target = st.Target(frozenset(index[s] for s in keep), tuple(index[s] for s in order))
    groups = None
    if rules.sector_cap:
        from store import db as store
        groups = store.sector_groups(conn, symbols)
    report = st.execute(book_, target, price, price, rules, costs, groups)
    sells = sorted(symbols[f.stock] for f in report.fills if f.side == "SELL")
    buys = [symbols[f.stock] for f in report.fills if f.side == "BUY"]
    return {"id": row[0], "decided_on": row[1], "sells": sells, "buys": buys,
            "scores": {s: scores.get(s) for s in buys}}


REASONS = {"manual": "Sell now (you)", "stop": "stop-loss", "news": "red-flag announcement"}


def orders(conn, *, pending_only: bool = True) -> list:
    sql = "SELECT id, created_at, symbol, day, at, status, done_on, reason, signal_day, detail FROM paper_orders"
    sql += " WHERE status = 'pending' ORDER BY id" if pending_only else " ORDER BY id DESC LIMIT 50"
    out = []
    for r in conn.execute(sql):
        o = dict(zip(("id", "created_at", "symbol", "day", "at", "status", "done_on", "reason", "signal_day",
                      "detail"), r))
        o["why"] = REASONS.get(o["reason"], o["reason"])
        try:
            o["info"] = json.loads(o["detail"]) if o["detail"] and o["detail"].startswith("{") else \
                {"text": o["detail"]} if o["detail"] else {}
        except ValueError:
            o["info"] = {"text": o["detail"]}
        out.append(o)
    return out


def rule_sales(conn, limit: int = 20) -> list:
    """Sales made by a rule (stop-loss, red-flag announcement), newest first. One still
    waiting is listed under the waiting orders only (28 Oct 2026 review)."""
    return [o for o in orders(conn, pending_only=False)
            if o["reason"] in ("stop", "news") and o["status"] != "pending"][:limit]


def media_news(conn, symbols, days: int = 7, today: date | None = None, per_stock: int = 6) -> list:
    """Headlines per holding, newest first, with the good/bad/neutral count."""
    symbols = list(symbols)
    since = ((today or date.today()) - timedelta(days=days)).isoformat()
    out = []
    for s in symbols:
        rows = [dict(zip(("at", "title", "source", "link", "sentiment", "words"), r)) for r in conn.execute(
            "SELECT at, title, source, link, sentiment, words FROM media_news WHERE symbol = ? AND at >= ? "
            "ORDER BY at DESC", (s, since))]
        count = {k: sum(1 for r in rows if r["sentiment"] == k) for k in ("good", "bad", "neutral")}
        out.append({"symbol": s, "count": count, "rows": rows[:per_stock], "total": len(rows)})
    out.sort(key=lambda x: (-x["count"]["bad"], -x["total"]))
    return out


def media_news_time(conn):
    row = conn.execute("SELECT value FROM meta WHERE key = 'media_news_at'").fetchone()
    return row[0] if row else None


def announcements(conn, symbols, days: int = 7, today: date | None = None) -> list:
    symbols = list(symbols)
    if not symbols:
        return []
    since = ((today or date.today()) - timedelta(days=days)).isoformat()
    marks = ",".join("?" * len(symbols))
    return [dict(zip(("symbol", "at", "subject", "text", "link", "flag"), r)) for r in conn.execute(
        f"SELECT symbol, at, subject, text, link, flag FROM announcements WHERE symbol IN ({marks}) AND at >= ? "
        f"ORDER BY flag IS NULL, at DESC LIMIT 60", [*symbols, since])]


def names(conn, symbols) -> dict:
    symbols = list(symbols)
    if not symbols:
        return {}
    marks = ",".join("?" * len(symbols))
    return dict(conn.execute(f"SELECT symbol, name FROM instruments WHERE symbol IN ({marks})", symbols))


def _action_factor(conn, symbol: str, after: str, upto: str) -> float:
    """Product of NSE's split/bonus factors with an ex-date in (after, upto]."""
    from strategy import live as lv
    f = 1.0
    for (factor,) in conn.execute(
            f"SELECT factor FROM corporate_actions WHERE symbol = ? AND ex_date > ? AND ex_date <= ? AND factor > 0 "
            f"AND kind IN ({','.join('?' * len(lv.ADJUSTABLE))})", (symbol, after, upto, *lv.ADJUSTABLE)):
        f *= float(factor)
    return f


def positions(conn, live: dict | None = None, today: date | None = None) -> dict:
    """Holdings valued at Kite's live price when given, else the last close.
    ``day_pct``: today's move - the live price against the last close before
    ``today``, or the last close against the one before it."""
    live = live or {}
    rows = conn.execute("SELECT symbol, shares, cost, opened_on FROM paper_positions ORDER BY symbol").fetchall()
    last = {}
    for symbol, *_ in rows:
        last[symbol] = [(float(c), d) for c, d in conn.execute(
            "SELECT close, date FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 2", (symbol,))]
    label = names(conn, [r[0] for r in rows])
    book_day = conn.execute("SELECT max(date) FROM paper_values").fetchone()[0] or ""
    out, total, day_change = [], 0.0, 0.0
    for symbol, shares, cost, opened in rows:
        closes = last.get(symbol) or []
        close, close_day = closes[0] if closes else (None, None)
        price = live.get(symbol) or close
        if live.get(symbol) and close is not None and (today is None or close_day < today.isoformat()):
            ref, ref_day = close, close_day               # during the day: against yesterday's close
            price_day = today.isoformat() if today else close_day
        else:
            ref, ref_day = (closes[1][0], closes[1][1]) if len(closes) > 1 else (None, None)
            price_day = close_day
        # (22 Oct 2026) a split or bonus between the reference close and the price:
        # the reference is put on the new basis, and shares the evening run has not
        # scaled yet (an ex-date today, before the evening) are counted as they will be
        pending = _action_factor(conn, symbol, max(book_day, opened or ""), price_day) if price_day else 1.0
        if pending != 1.0 and live.get(symbol) and close:
            # before the open Kite still shows yesterday's price: is it on the new basis yet?
            ratio = live[symbol] / close
            if abs(np.log(ratio * pending)) >= abs(np.log(ratio)):
                pending, price_day = 1.0, close_day
        if ref is not None and ref_day and price_day:
            ref = ref / _action_factor(conn, symbol, ref_day, price_day)
        if close_day and not live.get(symbol):
            # (28 Oct 2026 review) the evening already multiplied the shares for a split or bonus on a day
            # the stock did not trade: its last close (and the one before) go on the same new basis
            from strategy import live as lv
            applied = lv.applied_after(conn, symbol, close_day)       # (29 Oct 2026) the book's own rule
            if applied != 1.0:
                price = price / applied if price else price
                ref = ref / applied if ref else ref
        shares = shares * pending
        value = shares * price if price else None
        total += value or 0.0
        day = (price / ref - 1) * 100 if price and ref else None
        if price and ref:
            day_change += shares * (price - ref)
        out.append({"symbol": symbol, "name": label.get(symbol), "shares": shares, "cost": cost,
                    "pending_factor": pending, "price_day": price_day,
                    "opened_on": opened, "last_close": close, "close_day": close_day,
                    "live": live.get(symbol), "price": price, "value": value, "day_ref": ref, "day_pct": day,
                    "pnl": value - cost if value is not None else None,
                    "pnl_pct": (value / cost - 1) * 100 if value and cost else None})
    money = float((conn.execute("SELECT balance FROM paper_cash ORDER BY id DESC LIMIT 1").fetchone() or [0.0])[0])
    for p in out:
        p["weight_pct"] = p["value"] / (total + money) * 100 if p["value"] and total + money else None
    value = total + money
    cost = sum(p["cost"] for p in out if p["value"] is not None)
    gain = sum(p["pnl"] for p in out if p["pnl"] is not None)
    days = {p["price_day"] for p in out if p.get("price_day")}
    newest = max(days) if days else None
    # (22 Oct 2026) without live prices during the day the move shown is the last session's, not today's
    day_label = "Today" if newest is None or today is None or newest == today.isoformat() \
        else f"Last session ({date.fromisoformat(newest):%d %b})"
    return {"positions": out, "cash": money, "invested": total, "value": value, "live": bool(live),
            "day_label": day_label,
            "gain": gain, "gain_pct": gain / cost * 100 if cost else None,
            "day_change": day_change, "day_pct": day_change / (value - day_change) * 100 if value - day_change else None}


LOSS_ALERT_PCT = -15.0           # a holding down this much from its buy price is flagged
MARKET_AVERAGE_SESSIONS = 200    # NIFTY below its 200-day average: the market is falling


def _inr(value, digits=0):
    from web.routes import inr                    # the pages' own rupee format (Indian grouping)
    return inr(value, digits)


def risk_alerts(conn, pos: dict) -> dict:
    """Flags on the holdings, each saying what the rules will do: a holding down
    15% or more (the 25% stop's price, or the day a triggered stop sells it),
    a holding below the keep rank (sold at the month-end decision if still
    below), and NIFTY below its 200-day average (shown only)."""
    from strategy import live
    rules = st.Rules(**((book(conn) or {}).get("rules") or {}))
    flags = {}
    stops = dict(conn.execute("SELECT symbol, day FROM paper_orders WHERE status = 'pending' AND reason = 'stop'"))
    for p in pos["positions"]:
        s = p["symbol"]
        if s in stops:
            flags.setdefault(s, []).append(f"stop hit: sold at the open on {date.fromisoformat(stops[s]):%d %b}")
        elif p["pnl_pct"] is not None and p["pnl_pct"] <= LOSS_ALERT_PCT:
            entry = live.entry_price(conn, s)
            if entry and p.get("pending_factor"):
                entry /= p["pending_factor"]              # an ex-date today: the stop price on the new basis
            entry = entry or (p["cost"] / p["shares"] if p.get("shares") else None)
            if rules.stop_loss and entry:
                flags.setdefault(s, []).append(
                    f"down {abs(p['pnl_pct']):.0f}%: the {rules.stop_loss:.0%} stop sells it at the next open "
                    f"if it closes at or below {_inr(entry * (1 - rules.stop_loss), 2)}")
            else:
                flags.setdefault(s, []).append("down 15% or more from its buy price")
    if pos["positions"]:
        rank = ranking(conn)
        decision = month_end_session(date.fromisoformat(rank["day"])) if rank["day"] else None
        if decision and conn.execute("SELECT 1 FROM paper_decisions WHERE decided_on = ?",
                                     (decision.isoformat(),)).fetchone():
            # that month end's decision is already made (28 Oct 2026 review): the next one is meant
            decision = month_end_session(_calendar().next_trading_day(decision))
        for r in rank["rows"]:
            if r["held"] and r["status"] == "below keep zone":
                when = f"the {decision:%d %b} decision" if decision else "the next month-end decision"
                flags.setdefault(r["symbol"], []).append(
                    f"below rank {rules.keep_rank}: sold at {when} if still below {rules.keep_rank}")
    closes = [r[0] for r in conn.execute(
        "SELECT close FROM index_prices WHERE index_key = 'NIFTY 50' ORDER BY date DESC LIMIT ?",
        (MARKET_AVERAGE_SESSIONS,))]
    market = {"falling": None, "nifty": None, "average_200": None}
    if len(closes) == MARKET_AVERAGE_SESSIONS:
        avg = sum(closes) / len(closes)
        market = {"falling": closes[0] < avg, "nifty": closes[0], "average_200": avg}
    return {"holdings": flags, "market": market}


def fills(conn, limit: int = 100, symbol: str | None = None) -> list:
    sql = ("SELECT f.date, f.symbol, f.side, f.shares, f.price, f.fill_price, f.value, f.charges, "
           "f.decision_id IS NULL, f.source, f.checked FROM paper_fills f")
    params = []
    if symbol:
        sql += " WHERE f.symbol = ?"
        params.append(symbol)
    sql += " ORDER BY f.id DESC LIMIT ?"
    params.append(limit)
    rows = [dict(zip(("date", "symbol", "side", "shares", "price", "fill_price", "value", "charges", "manual",
                      "source", "checked"), r)) for r in conn.execute(sql, params)]
    for r in rows:     # where the price came from, for the Trades table
        if r["source"] == "kite_open":
            r["price_note"] = ("Kite open, NSE check tonight" if r["checked"] is None else
                               "Kite open = NSE open" if r["checked"] == "same" else r["checked"])
        else:
            r["price_note"] = "live price" if r["source"] == "live" else None
    return rows


# ------------------------------------------------------------------ ranking
def latest_screen_day(conn) -> date | None:
    d = conn.execute("SELECT max(date) FROM screen").fetchone()[0]
    return date.fromisoformat(d) if d else None


def ranking(conn, day: date | None = None) -> dict:
    """Every screened stock ranked by 6-month return on ``day`` (default: the
    latest screen), as the monthly decision ranks them, plus the holdings.
    ``status``: buy zone (top ``holdings``), keep zone (to ``keep_rank``),
    or, for a holding, sell at the next decision."""
    rules = st.Rules(**((book(conn) or {}).get("rules") or {}))
    day = day or latest_screen_day(conn)
    if day is None:
        return {"day": None, "rows": [], "screened": 0, "rules": rules}
    stamp = conn.execute("SELECT max(date) FROM prices").fetchone()[0]
    held = {r[0]: r[1] for r in conn.execute("SELECT symbol, opened_on FROM paper_positions")}
    key = ("ranking", day, stamp, tuple(sorted(held)), rules)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    from pipeline import panel as pn
    screened = [r[0] for r in conn.execute("SELECT symbol FROM screen WHERE date = ?", (day.isoformat(),))]
    symbols = sorted(set(screened) | set(held))
    p = pn.load(conn, day - timedelta(days=RANK_WARMUP_DAYS), day, symbols=symbols, light=True)
    t = len(p.sessions) - 1
    rows = []
    if t >= 0 and p.sessions[t] == day:
        score = st.scores(p.close_ff, rules.lookback, p.listed)[t]
        six = st.scores(p.close_ff, "6m")[t]
        month = p.close_ff[t] / p.close_ff[max(0, t - 21)] - 1
        eligible = p.listed[t] & np.isfinite(score) & np.isfinite(p.close[t])
        order = sorted(np.flatnonzero(eligible), key=lambda j: -score[j])
        rank = {int(j): i + 1 for i, j in enumerate(order)}
        label = names(conn, symbols)
        for j, symbol in enumerate(p.symbols):
            r = rank.get(j)
            if r is None and symbol not in held:
                continue
            if r is not None and r <= rules.holdings:
                status = "buy zone"
            elif r is not None and r <= rules.keep_rank:
                status = "keep zone"
            else:
                status = "below keep zone"
            rows.append({"rank": r, "symbol": symbol, "name": label.get(symbol),
                         "close": float(p.raw_close[t, j]) if np.isfinite(p.raw_close[t, j]) else None,
                         "score": float(score[j]) if np.isfinite(score[j]) else None,
                         "ret_6m": float(six[j]) * 100 if np.isfinite(six[j]) else None,
                         "ret_1m": float(month[j]) * 100 if np.isfinite(month[j]) else None,
                         "held": symbol in held, "on_screen": bool(p.listed[t, j]), "status": status})
        rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0))
        sector = dict(conn.execute("SELECT symbol, sector FROM sectors"))
        count = {}
        for r in rows:          # holdings that stay (the others are sold at the next decision)
            r["sector"] = sector.get(r["symbol"])
            if r["held"] and r["status"] != "below keep zone" and r["sector"]:
                count[r["sector"]] = count.get(r["sector"], 0) + 1
        for r in rows:          # then the buys, best first, as execute() fills the places
            if not rules.sector_cap or r["held"] or r["status"] != "buy zone" or not r["sector"]:
                continue
            if count.get(r["sector"], 0) >= rules.sector_cap:
                r["status"] = "sector full"
            else:
                count[r["sector"]] = count.get(r["sector"], 0) + 1
    result = {"day": day.isoformat(), "rows": rows, "screened": len(screened), "rules": rules}
    with _cache_lock:
        _cache.clear()
        _cache[key] = result
    return result


def stock(conn, symbol: str, today: date) -> dict | None:
    """One stock: a year of adjusted closes beside NIFTY 50, its place in the
    ranking, and the book's trades in it."""
    if not conn.execute("SELECT 1 FROM prices WHERE symbol = ? LIMIT 1", (symbol,)).fetchone():
        return None
    from pipeline import panel as pn
    p = pn.load(conn, today - timedelta(days=400), today, symbols=[symbol], light=True)
    keep = np.isfinite(p.close[:, 0])
    series = [{"date": d.isoformat(), "close": float(c), "nifty": float(n) if np.isfinite(n) else None}
              for d, c, n, k in zip(p.sessions, p.close[:, 0], p.index_close, keep) if k]
    info = conn.execute("SELECT name, isin FROM instruments WHERE symbol = ?", (symbol,)).fetchone()
    rank = next((r for r in ranking(conn)["rows"] if r["symbol"] == symbol), None)
    held = conn.execute("SELECT shares, cost, opened_on FROM paper_positions WHERE symbol = ?", (symbol,)).fetchone()
    actions = [dict(zip(("ex_date", "kind", "factor", "subject"), r)) for r in conn.execute(
        "SELECT ex_date, kind, factor, subject FROM corporate_actions WHERE symbol = ? "
        "ORDER BY ex_date DESC LIMIT 10", (symbol,))]
    news = media_news(conn, [symbol], days=14, today=today, per_stock=15)[0]
    filings = announcements(conn, [symbol], days=30, today=today)
    return {"symbol": symbol, "name": info[0] if info else None, "isin": info[1] if info else None,
            "media": news, "filings": filings,
            "series": series, "rank": rank, "fills": fills(conn, 50, symbol), "actions": actions,
            "held": dict(zip(("shares", "cost", "opened_on"), held)) if held else None}


# ------------------------------------------------------------------ research
def runs(conn) -> list:
    out = []
    for run_id, created, rules, start, end, summary in conn.execute(
            "SELECT id, created_at, rules, start_date, end_date, summary FROM bt_runs ORDER BY id DESC"):
        out.append({"id": run_id, "created_at": created, "rules": json.loads(rules), "start": start,
                    "end": end, "summary": json.loads(summary or "{}")})
    return out


def run_values(conn, run_id: int, every: int = 5) -> list:
    """The run's value, NIFTY and all screened stocks, each starting at 100
    (every ``every``-th session, for a light chart)."""
    rows = conn.execute("SELECT date, value, nifty, all_screened FROM bt_values WHERE run_id = ? ORDER BY date",
                        (run_id,)).fetchall()
    if not rows:
        return []
    base = [next((r[i] for r in rows if r[i]), None) for i in range(4)]
    picked = rows[::every] + ([rows[-1]] if (len(rows) - 1) % every else [])
    return [{"date": r[0], **{k: (r[i] / base[i] * 100 if r[i] and base[i] else None)
                              for i, k in ((1, "value"), (2, "nifty"), (3, "all_screened"))}} for r in picked]


# ------------------------------------------------------------------ settings / status
def jobs(conn, limit: int = 30) -> list:
    out = []
    for job, started, finished, status, detail in conn.execute(
            "SELECT job, started_at, finished_at, status, detail FROM job_log ORDER BY id DESC LIMIT ?", (limit,)):
        try:
            detail = json.loads(detail or "{}")
        except ValueError:
            detail = {"text": detail}
        out.append({"job": job, "started_at": started, "finished_at": finished, "status": status,
                    "detail": detail})
    return out


def data_status(conn) -> dict:
    one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]
    last_day = one("SELECT max(date) FROM prices")
    recent = (date.fromisoformat(last_day) - timedelta(days=30)).isoformat() if last_day else "0000"
    first = _cached_first_day(conn)
    return {
        "last_price_day": last_day,
        "last_nse_day": one("SELECT max(date) FROM prices WHERE source = 'nse'"),
        "kite_days": [r[0] for r in conn.execute(      # Kite rows are replaced within days: recent only
            "SELECT DISTINCT date FROM prices WHERE date >= ? AND source = 'kite' ORDER BY date DESC LIMIT 5",
            (recent,))],
        "stocks_last_day": one("SELECT count(*) FROM prices WHERE date = ?", last_day) if last_day else 0,
        "first_day": first,
        "nifty_last_day": one("SELECT max(date) FROM index_prices WHERE index_key = 'NIFTY 50'"),
        "screen_last_day": one("SELECT max(date) FROM screen"),
        "screened_last_day": one("SELECT count(*) FROM screen WHERE date = (SELECT max(date) FROM screen)"),
        "actions_last_ex_date": one("SELECT max(ex_date) FROM corporate_actions"),
    }


_first_day = {}


def _cached_first_day(conn):
    """The first price day never changes; reading it scans millions of rows (34 s on the PC), so once."""
    key = conn.execute("PRAGMA database_list").fetchone()[2]
    if key not in _first_day:
        _first_day[key] = (conn.execute("SELECT min(date) FROM legacy_prices").fetchone()[0]
                           or conn.execute("SELECT min(date) FROM prices").fetchone()[0])
    return _first_day[key]


def kite_status() -> dict:
    """Configured, and when the last login was saved (not the token itself)."""
    from config import Config
    out = {"configured": bool(Config.KITE_API_KEY), "login_saved_at": None}
    try:
        if Config.KITE_TOKEN_FILE.exists():
            out["login_saved_at"] = json.loads(Config.KITE_TOKEN_FILE.read_text(encoding="utf-8")).get("created_at")
    except (OSError, ValueError):
        pass
    return out


# ------------------------------------------------------------------ experiments
def week_end_session(day: date, cal=None) -> date:
    """The next week-end decision day: the last trading session of the week of
    the first session on or after ``day``."""
    cal = cal or _calendar()
    d = day if cal.is_trading_day(day) else cal.next_trading_day(day)
    while True:
        nxt = cal.next_trading_day(d)
        if nxt.isocalendar()[:2] != d.isocalendar()[:2]:
            return d
        d = nxt


def _worst_fall(series) -> float | None:
    """Biggest drop from a peak, in %; None before there are two values."""
    peak, worst, n = None, 0.0, 0
    for v in series:
        if v is None:
            continue
        n += 1
        peak = v if peak is None else max(peak, v)
        worst = min(worst, (v / peak - 1) * 100)
    return worst if n > 1 else None


def experiments(conn, today: date, cal=None) -> dict:
    """The Experiments page: each pretend portfolio beside the real one, all
    from the experiments' start day (strategy/experiments.py)."""
    from store import db as store
    from strategy import experiments as ex
    cal = cal or _calendar()
    start = store.get_meta(conn, "experiments_started_on")
    out = {"start": start, "done_for": store.get_meta(conn, "experiments_done_for"),
           "note": store.get_meta(conn, "experiments_note"), "capital": ex.CAPITAL, "items": [], "chart": []}
    if not start:
        return out
    real = {r[0]: (r[1], r[2]) for r in conn.execute(
        "SELECT date, value, nifty FROM paper_values WHERE date >= ? ORDER BY date", (start,))}
    real_held = {r[0] for r in conn.execute("SELECT symbol FROM paper_positions")}
    held_by = {e.key: {r[0] for r in conn.execute("SELECT symbol FROM exp_positions WHERE exp = ?", (e.key,))}
               for e in ex.EXPERIMENTS}
    curves = {}
    for e in ex.EXPERIMENTS:
        vals = conn.execute("SELECT date, value, cash, holdings FROM exp_values WHERE exp = ? ORDER BY date",
                            (e.key,)).fetchall()
        curves[e.key] = {d: v for d, v, _, _ in vals}
        last = vals[-1] if vals else None
        pos = [dict(zip(("symbol", "shares", "buy_price", "bought_on", "close"), r)) for r in conn.execute(
            "SELECT symbol, shares, buy_price, bought_on, close FROM exp_positions WHERE exp = ? "
            "ORDER BY shares * close DESC", (e.key,))]
        for p in pos:
            p["value"] = p["shares"] * p["close"] if p["close"] else None
            p["gain_pct"] = (p["close"] / p["buy_price"] - 1) * 100 if p["close"] and p["buy_price"] else None
            p["in_real"] = p["symbol"] in real_held
        fills = [dict(zip(("date", "symbol", "side", "shares", "price", "value", "charges", "reason"), r))
                 for r in conn.execute("SELECT date, symbol, side, shares, price, value, charges, reason FROM exp_fills "
                                       "WHERE exp = ? ORDER BY date DESC, side DESC, symbol LIMIT 60", (e.key,))]
        planned = [dict(zip(("symbol", "side", "decided_on", "reason"), r)) for r in conn.execute(
            "SELECT symbol, side, decided_on, reason FROM exp_planned WHERE exp = ? ORDER BY side DESC, symbol",
            (e.key,))]
        for p in planned:          # made at the open after the decision; shown with prices after that evening
            p["trade_on"] = cal.next_trading_day(date.fromisoformat(p["decided_on"])).isoformat()
        trades = conn.execute("SELECT count(*), coalesce(sum(charges), 0) FROM exp_fills WHERE exp = ?",
                              (e.key,)).fetchone()
        nxt = (week_end_session if e.rules.rebalance == "week_end" else month_end_session)(today, cal)
        b = e.backtest
        out["items"].append({
            "key": e.key, "name": e.name, "about": e.about, "rebalance": e.rules.rebalance, "periods": e.periods,
            "backtest": dict(zip(("a_ret", "a_fall", "b_ret", "b_fall", "costs"), b)) if b else None,
            "value": last[1] if last else None, "cash": last[2] if last else None,
            "return_pct": (last[1] / ex.CAPITAL - 1) * 100 if last else None,
            "worst_fall": _worst_fall(v for _, v, _, _ in vals),
            "positions": pos, "fills": fills, "planned": planned, "trades": trades[0], "charges": trades[1],
            "same_as_real": len(held_by[e.key] & real_held), "real_count": len(real_held),
            "same_as_copy": len(held_by[e.key] & held_by["copy"]) if e.key != "copy" else None,
            "next_decision": nxt.isoformat()})
    # the chart: everything at 100 on the start day's close
    days = sorted(set(real) | {d for c in curves.values() for d in c})
    first_real = real.get(start) or (real[min(real)] if real else None)
    for d in days:
        row = {"date": d}
        for k, c in curves.items():
            row[k] = c[d] / ex.CAPITAL * 100 if d in c else None
        r = real.get(d)
        row["real"] = r[0] / first_real[0] * 100 if r and first_real else None
        row["nifty"] = r[1] / first_real[1] * 100 if r and r[1] and first_real and first_real[1] else None
        out["chart"].append(row)
    real_vals = [real[d][0] for d in sorted(real)]
    out["real"] = {"value": real_vals[-1] if real_vals else None,
                   "return_pct": (real_vals[-1] / real_vals[0] - 1) * 100 if len(real_vals) > 1 else None,
                   "worst_fall": _worst_fall(real_vals)}
    return out
