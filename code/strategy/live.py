"""The live paper portfolio (Redesign v2, Phase 4).

Run once each evening after the day's NSE file is in (pipeline/daily.py):

  1. splits/bonuses on held stocks (NSE's corporate actions, ex-date today):
     shares are multiplied, as a broker credits them;
  2. a decision made at an earlier close is executed at TODAY's official open
     with strategy.strength.execute - the backtest's own function;
  3. the book is valued at today's close, beside NIFTY 50 and all screened
     stocks in equal amounts;
  4. on the last session of a month (or the first evening), a new decision is
     made with strategy.strength.decide on 6-month returns of adjusted prices.

"Sell now" (the Portfolio page): during market hours a holding is sold at
Kite's live price at once (sell_now); outside market hours, or on a day the
monthly trades are being made, the click is kept as an order that this
evening run fills at the official open (or close) - see place_sell.

Morning fills (16 Oct 2026): at about 09:20 the trades waiting for today's
open (the decision, stop/news/Sell-now orders) are made at Kite's opening
price (morning); the evening run then checks each against NSE's official open
and corrects any difference (check_kite_fills). Without Kite, or when anything
looks unusual (a missing open, a split on the day), the morning leaves them to
the evening as before.

Running it twice for the same day changes nothing.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from pipeline import panel as pn
from strategy import strength as st

ADJUSTABLE = ("split", "bonus", "consolidation")
LATE_ACTION_DAYS = 30            # an ex-date this many days back is still applied if it was missed
SCORE_WARMUP_DAYS = 420          # calendar days of prices: the NSE-style score needs a year and more
# One writer of the book at a time in this process: the evening run, "Sell
# now" and order cancels (each also takes SQLite's write lock first, with
# BEGIN IMMEDIATE, so what it reads cannot change before it writes).
BOOK_LOCK = threading.RLock()


def begin(conn) -> None:
    """Start a write transaction before reading (no-op if one is open)."""
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")


def _next_session_default(day: date) -> date:
    from core.market_calendar import next_trading_day
    return next_trading_day(day)


def start_book(conn, capital: float, rules: st.Rules, day: date) -> None:
    conn.execute("INSERT INTO paper_book (id, capital, started_on, rules) VALUES (1, ?, ?, ?)",
                 (float(capital), day.isoformat(), json.dumps(asdict(rules))))
    conn.execute("INSERT INTO paper_cash (date, kind, amount, balance) VALUES (?, 'deposit', ?, ?)",
                 (day.isoformat(), float(capital), float(capital)))


def book_rules(conn) -> st.Rules | None:
    row = conn.execute("SELECT rules FROM paper_book WHERE id = 1").fetchone()
    return st.Rules(**json.loads(row[0])) if row else None


LIVE_RULES = st.Rules(sector_cap=4, stop_loss=0.25)   # NSE-style ranking, 4 per sector, 25% stop at the close
# (the stop during the day - Rules.stop_intraday, check_stops_live - was tested on 16 Oct and not adopted:
# about 1-1.5 points a year lower, no smaller worst fall; the code stays, switched off)

RULE_CHANGES = (
    # (meta key, field, from, to, why) - applied once to a running book; the
    # next decision uses the new rule, nothing already bought is touched.
    ("rule_change_nse_ranking", "lookback", "6m", "nse",
     "15 Oct 2026: NSE-style ranking passed Phase 6 on 2010-18 and 2019-26 (higher return, smaller falls)"),
    ("rule_change_sector_cap_4", "sector_cap", None, 4,
     "15 Oct 2026: at most 4 stocks per NSE sector passed Phase 6 round 2 (2010-18 +28.5%/yr, worst -29%; "
     "2019-26 +36.1%/yr, worst -35%)"),
    ("rule_change_stop_25", "stop_loss", None, 0.25,
     "15 Oct 2026: 25% disaster stop (user's choice; with the sector limit: 2010-18 +27.2%/yr, worst -31%; "
     "2019-26 +37.0%/yr, worst -33%; 2018 -14%)"),
    ("rule_change_stop_at_close", "stop_intraday", True, False,
     "16 Oct 2026: back to the 25% stop at the close (the stop during the day tested 1-1.5 points a year lower)"),
)


def apply_rule_changes(conn) -> list:
    """Agreed rule changes for a book started under older rules. Commits."""
    from store import db as store
    done = []
    row = conn.execute("SELECT rules FROM paper_book WHERE id = 1").fetchone()
    for key, field_name, old, new, why in RULE_CHANGES:
        if store.get_meta(conn, key) or not row:
            continue
        rules = json.loads(row[0])
        if rules.get(field_name) == old:
            rules[field_name] = new
            conn.execute("UPDATE paper_book SET rules = ? WHERE id = 1", (json.dumps(rules),))
            conn.execute("INSERT INTO job_log (job, started_at, finished_at, status, detail) VALUES "
                         "('rule_change', datetime('now', 'localtime'), datetime('now', 'localtime'), 'ok', ?)",
                         (json.dumps({"field": field_name, "from": old, "to": new, "why": why}),))
            done.append((field_name, old, new))
            row = (json.dumps(rules),)
        store.set_meta(conn, key, "done")
    conn.commit()
    return done


def cash(conn) -> float:
    row = conn.execute("SELECT balance FROM paper_cash ORDER BY id DESC LIMIT 1").fetchone()
    return float(row[0]) if row else 0.0


def _prices(conn, day: date, symbols) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "open", "close"]).set_index("symbol")
    marks = ",".join("?" * len(symbols))
    return pd.read_sql_query(f"SELECT symbol, open, close FROM prices WHERE date = ? AND symbol IN ({marks})",
                             conn, params=[day.isoformat(), *symbols]).set_index("symbol")


def _last_close(conn, day: date, symbols, *, before: bool) -> dict:
    """Last raw close on or before ``day`` (strictly before if ``before``)."""
    op = "<" if before else "<="
    out = {}
    for s in symbols:
        row = conn.execute(f"SELECT close FROM prices WHERE symbol = ? AND date {op} ? ORDER BY date DESC LIMIT 1",
                           (s, day.isoformat())).fetchone()
        if row:
            out[s] = float(row[0])
    return out


def apply_corporate_actions(conn, day: date) -> list:
    """Multiply held shares by NSE's split/bonus factor on the ex-date. A held
    stock whose price jumped outside 0.55-1.8 with no NSE record is adjusted
    the same way the backtest does (value kept) and logged as 'detected'."""
    done = []
    # a position bought today (a morning fill) was bought at today's prices: nothing to adjust
    rows = conn.execute("SELECT symbol, shares, opened_on FROM paper_positions WHERE opened_on != ?",
                        (day.isoformat(),)).fetchall()
    held = {r[0]: r[1] for r in rows}
    opened = {r[0]: r[2] for r in rows}
    if not held:
        return _credit_sold_before_listing(conn, day, set())     # an all-cash book can still be owed shares
    # today's ex-dates, and (22 Oct 2026) earlier ones NSE listed late or that a
    # failed refresh missed: shares the holder was owed then, not adjusted since
    acts = conn.execute(
        f"SELECT symbol, kind, factor, ex_date FROM corporate_actions WHERE ex_date BETWEEN ? AND ? AND kind IN "
        f"({','.join('?' * len(ADJUSTABLE))}) AND factor > 0 AND symbol IN ({','.join('?' * len(held))}) "
        f"ORDER BY ex_date",
        [(day - timedelta(days=LATE_ACTION_DAYS)).isoformat(), day.isoformat(), *ADJUSTABLE, *held]).fetchall()
    recorded = set()
    for symbol, kind, factor, ex_date in _combined(acts):
        if ex_date <= opened[symbol]:
            continue                                   # bought on or after the ex-date: at the new price
        # what was applied for this ex-date already (on the day, late, "detected" from the price
        # jump, or a correction): (29 Oct 2026 review) NSE may list a bonus and a split for one
        # ex-date on different days - only the part not applied yet is applied now
        since = conn.execute("SELECT kind, factor FROM paper_actions WHERE symbol = ? AND date >= ? AND date < ? "
                             "AND kind NOT LIKE 'credited after the sale%' AND ((kind = 'detected' AND date <= ?) "
                             "OR date = ? OR kind LIKE ?)",
                             (symbol, ex_date, day.isoformat(),
                              (date.fromisoformat(ex_date) + timedelta(days=7)).isoformat(),   # that move, not a later one
                              ex_date, f"%ex {ex_date}%")
                             ).fetchall() if ex_date < day.isoformat() else []
        if since:
            applied = float(np.prod([float(f) for _, f in since if f]))
            fix = float(factor) / applied
            if abs(fix - 1) > 1e-6:
                how = "detected" if all(k == "detected" for k, _ in since) else "applied"
                _scale(conn, day, symbol, f"correction to {kind} {float(factor):g} (ex {ex_date}, was {how} "
                                          f"{applied:.4g})", fix, held)
                recorded.add(symbol)
                done.append((symbol, "correction", fix))
            continue
        _scale(conn, day, symbol, kind if ex_date == day.isoformat() else f"{kind} (late, ex {ex_date})",
               float(factor), held)
        recorded.add(symbol)
        done.append((symbol, kind, float(factor)))
    done += _credit_sold_before_listing(conn, day, set(held))
    today = _prices(conn, day, [s for s in held if s not in recorded])
    # the last close before today on the shares' basis: a split applied on a day the stock did not
    # trade must not be "detected" again the next session (28 Oct 2026 review)
    before = adjusted_last_close(conn, day, list(today.index), before=True)
    for symbol, row in today.iterrows():
        prev = before.get(symbol)
        if not prev or not np.isfinite(row["close"]):
            continue
        ratio = float(row["close"]) / prev
        if not (st.JUMP_LOW < ratio < st.JUMP_HIGH):
            _scale(conn, day, symbol, "detected", 1.0 / ratio, held)
            done.append((symbol, "detected", 1.0 / ratio))
    return done


def _combined(acts) -> list:
    """(29 Oct 2026 review) NSE's rows for one stock on one ex-date (a bonus and a split
    together) as one action: kinds joined, factors multiplied - so a "detected" jump is
    corrected once, to the whole factor, and a credit covers every share owed."""
    out = {}
    for symbol, kind, factor, ex_date in acts:
        k = (symbol, ex_date)
        if k in out:
            out[k] = (symbol, f"{out[k][1]} + {kind}", out[k][2] * float(factor), ex_date)
        else:
            out[k] = (symbol, kind, float(factor), ex_date)
    return sorted(out.values(), key=lambda a: (a[3], a[0]))


def _credit_sold_before_listing(conn, day: date, held: set) -> list:
    """A split or bonus NSE listed only after the book had sold the stock (29 Oct 2026
    review): the sale sold the old share count at the new, lower price. The shares the
    book was owed are credited as cash at the sale's net price a share (after its charges)."""
    out = []
    for symbol, kind, factor, ex_date in _combined(conn.execute(
            f"SELECT symbol, kind, factor, ex_date FROM corporate_actions WHERE ex_date BETWEEN ? AND ? AND kind IN "
            f"({','.join('?' * len(ADJUSTABLE))}) AND factor > 0", [(day - timedelta(days=LATE_ACTION_DAYS)).isoformat(),
                                                                    day.isoformat(), *ADJUSTABLE]).fetchall()):
        if symbol in held or abs(factor - 1) < 1e-9:   # a consolidation (factor < 1) is a debit
            continue
        tag = f"credited after the sale ({kind}, ex {ex_date})"
        mark = f"credited after the sale (%, ex {ex_date})"
        if conn.execute("SELECT 1 FROM paper_actions WHERE symbol = ? AND date >= ? AND kind NOT LIKE "
                        "'credited after the sale%' LIMIT 1", (symbol, ex_date)).fetchone():
            continue                                   # adjusted while held: the sale had the right shares
        paid = conn.execute("SELECT factor FROM paper_actions WHERE symbol = ? AND kind LIKE ? ORDER BY id DESC "
                            "LIMIT 1", (symbol, mark)).fetchone()
        paid = float(paid[0]) if paid else 1.0         # a part NSE listed earlier (a bonus, then the split)
        if abs(float(factor) - paid) < 1e-9:
            continue                                   # credited already
        sale = conn.execute("SELECT date, shares, value FROM paper_fills WHERE symbol = ? AND side = 'SELL' "
                            "AND date >= ? AND date <= ? ORDER BY id LIMIT 1", (symbol, ex_date, day.isoformat())).fetchone()
        bought = conn.execute("SELECT max(date) FROM paper_fills WHERE symbol = ? AND side = 'BUY' AND date < ?",
                              (symbol, ex_date)).fetchone()[0]
        if not sale or not bought or conn.execute(
                "SELECT 1 FROM paper_fills WHERE symbol = ? AND side = 'SELL' AND date >= ? AND date < ?",
                (symbol, bought, ex_date)).fetchone():
            continue                                   # not held over the ex-date
        owed = float(sale[1]) * (float(factor) - paid)          # negative for a consolidation: a debit
        amount = owed * float(sale[2]) / float(sale[1])        # at the sale's own net price per share
        conn.execute("INSERT INTO paper_actions (date, symbol, kind, factor, shares_before, shares_after) "
                     "VALUES (?, ?, ?, ?, ?, ?)", (day.isoformat(), symbol, tag, float(factor), 0.0, 0.0))
        conn.execute("INSERT INTO paper_cash (date, kind, symbol, amount, balance) VALUES (?, 'late_action', ?, ?, ?)",
                     (day.isoformat(), symbol, amount, cash(conn) + amount))
        out.append((symbol, "credited", amount))
    return out


def _scale(conn, day, symbol, kind, factor, held):
    before = held[symbol]
    after = before * factor
    conn.execute("UPDATE paper_positions SET shares = ? WHERE symbol = ?", (after, symbol))
    conn.execute("INSERT INTO paper_actions (date, symbol, kind, factor, shares_before, shares_after) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (day.isoformat(), symbol, kind, factor, before, after))
    held[symbol] = after


def execute_pending(conn, day: date, rules: st.Rules, costs: st.Costs, opens: dict | None = None) -> list:
    """Execute the decision made at an earlier close at today's open (NSE's
    official open, or ``opens``: Kite's opening prices in the morning)."""
    row = conn.execute("SELECT id, keep, buy_order FROM paper_decisions WHERE status = 'pending' "
                       "AND decided_on < ? ORDER BY id DESC LIMIT 1", (day.isoformat(),)).fetchone()
    if not row:
        return []
    decision_id, keep, buy_order = row[0], json.loads(row[1]), json.loads(row[2])
    held = dict(conn.execute("SELECT symbol, shares FROM paper_positions"))
    symbols = sorted(set(held) | set(keep) | set(buy_order))
    index = {s: i for i, s in enumerate(symbols)}
    if opens is None:
        today = _prices(conn, day, symbols)
        open_row = np.array([today["open"].get(s, np.nan) for s in symbols], dtype=float)
    else:
        open_row = np.array([opens.get(s, np.nan) for s in symbols], dtype=float)
    source = None if opens is None else MORNING_SOURCE
    # a stock with no open that day is sold at its last close on the basis of today's shares
    # (actions applied this evening included: apply_corporate_actions has scaled them already)
    last = adjusted_last_close(conn, day, symbols, before=True, through_today=True)
    last_row = np.array([last.get(s, np.nan) for s in symbols], dtype=float)
    if opens is None:      # (29 Oct 2026) traded today but no open recorded: sold at today's close
        close_today = np.array([today["close"].get(s, np.nan) for s in symbols], dtype=float)
        last_row = np.where(np.isfinite(close_today) & (close_today > 0), close_today, last_row)
    book = st.Book(cash(conn), np.array([held.get(s, 0.0) for s in symbols], dtype=float))
    target = st.Target(frozenset(index[s] for s in keep), tuple(index[s] for s in buy_order))
    groups = None
    if rules.sector_cap:
        from store import db as store
        groups = store.sector_groups(conn, symbols)
    report = st.execute(book, target, open_row, last_row, rules, costs, groups)
    balance = cash(conn)
    for f in report.fills:
        symbol = symbols[f.stock]
        conn.execute("INSERT INTO paper_fills (decision_id, date, symbol, side, shares, price, fill_price, value, "
                     "charges, slippage, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (decision_id, day.isoformat(), symbol, f.side, f.shares, f.price, f.fill_price, f.value,
                      f.charges, f.slippage_cost, source))
        if f.side == "SELL":
            balance += f.value
            conn.execute("DELETE FROM paper_positions WHERE symbol = ?", (symbol,))
        else:
            balance -= f.value
            conn.execute("INSERT INTO paper_positions (symbol, shares, cost, opened_on) VALUES (?, ?, ?, ?)",
                         (symbol, f.shares, f.value, day.isoformat()))
        conn.execute("INSERT INTO paper_cash (date, kind, symbol, amount, balance) VALUES (?, ?, ?, ?, ?)",
                     (day.isoformat(), f.side.lower(), symbol, f.value if f.side == "SELL" else -f.value, balance))
    assert abs(balance - book.cash) < 1e-6 * max(1.0, abs(balance)), "cash ledger and book disagree"
    conn.execute("UPDATE paper_decisions SET status = 'executed', executed_on = ? WHERE id = ?",
                 (day.isoformat(), decision_id))
    return report.fills


def sell_amounts(shares: float, price: float, costs: st.Costs) -> tuple[float, float, float]:
    """(rupees received, charges, slippage) for selling ``shares`` at ``price`` -
    the same arithmetic as strategy.strength.execute."""
    gross = shares * price * (1 - costs.slippage)
    fee = gross * costs.sell_rate + costs.dp_per_sell
    return gross - fee, fee, shares * price * costs.slippage


def _drop_from_pending(conn, symbol: str) -> None:
    """A stock sold by hand is not bought back by a decision still waiting to
    be executed (it can return at the next month end if still strong). Its
    place in ``keep`` is left alone: whether the waiting decision sells it at
    the open is the decision's business, not the click's."""
    for decision_id, buy_order in conn.execute(
            "SELECT id, buy_order FROM paper_decisions WHERE status = 'pending'").fetchall():
        buy_order = json.loads(buy_order)
        if symbol in buy_order:
            conn.execute("UPDATE paper_decisions SET buy_order = ? WHERE id = ?",
                         (json.dumps([s for s in buy_order if s != symbol]), decision_id))


def unsafe_live_sale(conn, symbol: str, price: float, day: date) -> str | None:
    """Why a sale at Kite's live price now would be wrong, or None. On a
    split/bonus ex-date the price is already the new one while the book still
    holds the old number of shares (they are scaled in the evening run); a
    price far from the last close may be an unlisted action too (22 Oct 2026)."""
    if conn.execute(f"SELECT 1 FROM corporate_actions WHERE symbol = ? AND ex_date = ? AND kind IN "
                    f"({','.join('?' * len(ADJUSTABLE))})", (symbol, day.isoformat(), *ADJUSTABLE)).fetchone():
        return "today is its split/bonus ex-date"
    last = _last_close(conn, day, [symbol], before=True).get(symbol)
    if last and not (MORNING_BAND[0] <= price / last <= MORNING_BAND[1]):
        return f"Kite's price {price:.2f} is far from the last close {last:.2f} (a split or bonus?)"
    return None


def sell_now(conn, symbol: str, price: float, day: date, costs: st.Costs | None = None,
             source: str | None = "live") -> dict:
    """Sell a whole holding at ``price`` (Kite's live price, or an official
    open/close for an order: ``source`` None; Kite's open: kite_open). Does not commit."""
    costs = costs or st.Costs.from_config()
    row = conn.execute("SELECT shares FROM paper_positions WHERE symbol = ?", (symbol,)).fetchone()
    if not row:
        raise ValueError(f"{symbol} is not held")
    if not (price and np.isfinite(price) and price > 0):
        raise ValueError(f"no price to sell {symbol} at")
    shares = float(row[0])
    received, fee, slip = sell_amounts(shares, float(price), costs)
    balance = cash(conn) + received
    conn.execute("INSERT INTO paper_fills (decision_id, date, symbol, side, shares, price, fill_price, value, "
                 "charges, slippage, source) VALUES (NULL, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?)",
                 (day.isoformat(), symbol, shares, float(price), float(price) * (1 - costs.slippage), received,
                  fee, slip, source))
    conn.execute("DELETE FROM paper_positions WHERE symbol = ?", (symbol,))
    conn.execute("INSERT INTO paper_cash (date, kind, symbol, amount, balance) VALUES (?, 'sell', ?, ?, ?)",
                 (day.isoformat(), symbol, received, balance))
    _drop_from_pending(conn, symbol)
    return {"symbol": symbol, "shares": shares, "price": float(price), "received": received, "charges": fee}


def place_sell(conn, symbol: str, day: date, at: str, *, reason: str = "manual", signal_day: date | None = None,
               detail: str | None = None) -> int:
    """Keep a sale as an order for the evening run: a "Sell now" click (manual),
    the disaster stop (stop) or a red-flag announcement (news). Does not commit."""
    if at not in ("open", "close"):
        raise ValueError("at must be 'open' or 'close'")
    if not conn.execute("SELECT 1 FROM paper_positions WHERE symbol = ?", (symbol,)).fetchone():
        raise ValueError(f"{symbol} is not held")
    if conn.execute("SELECT 1 FROM paper_orders WHERE symbol = ? AND status = 'pending'", (symbol,)).fetchone():
        raise ValueError(f"a sell order for {symbol} is already waiting")
    from datetime import datetime
    cur = conn.execute("INSERT INTO paper_orders (created_at, symbol, day, at, reason, signal_day, detail) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, day.isoformat(), at, reason,
                        signal_day.isoformat() if signal_day else None, detail))
    _drop_from_pending(conn, symbol)
    return cur.lastrowid


_DUE_AT_OPEN = ("SELECT id, symbol FROM paper_orders WHERE status = 'pending' AND "
                "((at = 'open' AND day <= ?) OR (at = 'close' AND day < ?)) ORDER BY id")


def fill_orders(conn, day: date, at: str, costs: st.Costs, opens: dict | None = None) -> list:
    """Fill waiting sell orders due by ``day`` at its official open (or close),
    or at ``opens`` (Kite's opening prices, in the morning); a stock with no
    price that day goes at its last close."""
    done = []
    if at == "open":     # due at this open, or a close order whose day passed without a run (market closed)
        due = (_DUE_AT_OPEN, (day.isoformat(),) * 2)
    else:
        due = ("SELECT id, symbol FROM paper_orders WHERE status = 'pending' AND at = 'close' AND day = ? "
               "ORDER BY id", (day.isoformat(),))
    for order_id, symbol in conn.execute(*due).fetchall():
        if not conn.execute("SELECT 1 FROM paper_positions WHERE symbol = ?", (symbol,)).fetchone():
            conn.execute("UPDATE paper_orders SET status = 'not held', done_on = ? WHERE id = ?",
                         (day.isoformat(), order_id))
            continue
        if opens is not None:
            price, source = opens[symbol], MORNING_SOURCE
        else:
            row = _prices(conn, day, [symbol])
            v = row[at].iloc[0] if len(row) else None
            c = row["close"].iloc[0] if len(row) else None
            if v is not None and pd.notna(v) and float(v) > 0:
                price = float(v)
            elif c is not None and pd.notna(c) and float(c) > 0:
                price = float(c)   # (29 Oct 2026) traded today but no open recorded: today's close
            else:          # no price today: the last close, on the basis of today's (scaled) shares
                price = adjusted_last_close(conn, day, [symbol], before=True, through_today=True).get(symbol)
            source = None
        done.append(sell_now(conn, symbol, price, day, costs, source=source))
        conn.execute("UPDATE paper_orders SET status = 'done', done_on = ? WHERE id = ?", (day.isoformat(), order_id))
    return done


def entry_price(conn, symbol: str) -> float | None:
    """The price paid per share now held (the buy's fill price, carried through
    splits and bonuses: x shares bought / shares held) - the backtest's entry."""
    buy = conn.execute("SELECT fill_price, shares FROM paper_fills WHERE symbol = ? AND side = 'BUY' "
                       "ORDER BY id DESC LIMIT 1", (symbol,)).fetchone()
    held = conn.execute("SELECT shares FROM paper_positions WHERE symbol = ?", (symbol,)).fetchone()
    if not buy or not held or not held[0]:
        return None
    return float(buy[0]) * float(buy[1]) / float(held[0])


def check_stops(conn, day: date, rules: st.Rules, next_session) -> list:
    """Disaster stop: a holding whose close is ``stop_loss`` or more below its
    buy price is sold at the next open (as strategy.strength's overlay)."""
    if not rules.stop_loss:
        return []
    from store import db as store
    held = [r[0] for r in conn.execute("SELECT symbol FROM paper_positions")]
    # (29 Oct 2026 review) only stocks that traded today, and every close on the basis of today's share
    # count: a split or consolidation on a day without trades must not look like a crash
    traded = {r[0] for r in conn.execute(f"SELECT symbol FROM prices WHERE date = ? AND symbol IN "
                                         f"({','.join('?' * len(held))})", (day.isoformat(), *held))} if held else set()
    close = {s: v for s, v in adjusted_last_close(conn, day, held).items() if s in traded}
    prev = adjusted_last_close(conn, day, held, before=True, through_today=True)
    waiting = {r[0] for r in conn.execute("SELECT symbol FROM paper_orders WHERE status = 'pending'")}
    # (22 Oct 2026) without today's corporate-actions list, a one-day fall of a
    # fifth or more may be a bonus not yet known: that stop waits a day (the
    # list is read again tomorrow and a late ex-date is applied first)
    actions_known = (store.get_meta(conn, "actions_refreshed_on") or "") >= day.isoformat()
    out = []
    for symbol in held:
        entry = entry_price(conn, symbol)
        if symbol in waiting or entry is None or symbol not in close:
            continue
        ratio = close[symbol] / prev[symbol] if prev.get(symbol) else None
        stop_price = entry * (1 - rules.stop_loss)
        if prev.get(symbol) and prev[symbol] <= stop_price:
            ratio = None       # already below the stop before today: today's gap is not what triggers it - no wait
        if ratio is not None and ratio <= MORNING_BAND[0] and not actions_known:
            continue           # a one-day fall of a fifth or more without today's actions list: may be a bonus
        if ratio is not None and BONUS_LIKE[0] <= ratio <= BONUS_LIKE[1]:
            # even with the list, a 1:2-1:4 bonus missing from it looks exactly like this (28 Oct 2026
            # review) - the stop waits one day, once per holding: a stock at its lower circuit day after
            # day must not dodge the stop
            key = f"bonus_wait:{symbol}"
            opened = (conn.execute("SELECT opened_on FROM paper_positions WHERE symbol = ?", (symbol,)).fetchone()
                      or [""])[0] or ""
            waited = store.get_meta(conn, key) or ""
            if not (waited and waited >= opened):
                store.set_meta(conn, key, day.isoformat())
                continue
        if close[symbol] <= stop_price:
            place_sell(conn, symbol, next_session(day), "open", reason="stop", signal_day=day,
                       detail=f"closed at {close[symbol]:.2f}, {(close[symbol] / entry - 1) * 100:.1f}% "
                              f"from the buy price {entry:.2f}")
            out.append(symbol)
    return out


# ------------------------------------------------------------------ morning fills
MORNING_SOURCE = "kite_open"
# A held stock opening outside this band of its last close leaves the trades to
# the evening: a split or bonus NSE has not listed yet (1:4 bonus = 0.8) must
# not sell the old share count at the new price. Normal days rarely move this much.
MORNING_BAND = (0.8, 1.25)
BONUS_LIKE = (st.JUMP_LOW, 0.8)  # close / previous close of a 1:2 .. 1:4 bonus, a bad day included (29 Oct 2026)


def morning_symbols(conn, day: date, prev_day: date) -> list | None:
    """The stocks whose opening price today's waiting trades need, or None when
    nothing waits for this open - or the book's evening runs are not up to the
    previous session (the evening then catches up in order)."""
    last_run = conn.execute("SELECT max(date) FROM paper_values").fetchone()[0]
    if last_run != prev_day.isoformat():
        return None
    held = {r[0] for r in conn.execute("SELECT symbol FROM paper_positions")}
    symbols = {s for _, s in conn.execute(_DUE_AT_OPEN, (day.isoformat(),) * 2) if s in held}
    decision = conn.execute("SELECT buy_order FROM paper_decisions WHERE status = 'pending' AND decided_on < ? "
                            "ORDER BY id DESC LIMIT 1", (day.isoformat(),)).fetchone()
    if decision:
        symbols |= held | set(json.loads(decision[0]))
    return sorted(symbols) or None


def _morning_blockers(conn, day: date, symbols: list, opens: dict) -> str | None:
    """Why the morning should leave the trades to the evening, if it should."""
    missing = [s for s in symbols if not (opens.get(s) and np.isfinite(opens[s]) and opens[s] > 0)]
    if missing:
        return "no Kite opening price for " + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
    held = [s for s in symbols if conn.execute("SELECT 1 FROM paper_positions WHERE symbol = ?", (s,)).fetchone()]
    if not held:
        return None
    from store import db as store
    refreshed = store.get_meta(conn, "actions_refreshed_on")
    if not refreshed or refreshed < _prev_day_of(conn, day):
        return (f"NSE's corporate actions were last read on {refreshed or 'never'}: "
                "a split or bonus today might be missing")
    marks = ",".join("?" * len(held))
    acted = [r[0] for r in conn.execute(f"SELECT DISTINCT symbol FROM corporate_actions WHERE ex_date = ? AND "
                                        f"kind IN ({','.join('?' * len(ADJUSTABLE))}) AND symbol IN ({marks})",
                                        [day.isoformat(), *ADJUSTABLE, *held])]
    if acted:
        return "a split or bonus today on " + ", ".join(sorted(acted))
    late = []
    for s in held:                                 # (29 Oct 2026) listed late, not applied by an evening yet
        opened = conn.execute("SELECT opened_on FROM paper_positions WHERE symbol = ?", (s,)).fetchone()[0] or ""
        for (ex,) in conn.execute(f"SELECT ex_date FROM corporate_actions WHERE symbol = ? AND ex_date < ? AND "
                                  f"ex_date > ? AND ex_date >= ? AND factor > 0 AND kind IN ({','.join('?' * len(ADJUSTABLE))})",
                                  (s, day.isoformat(), opened,
                                   (day - timedelta(days=LATE_ACTION_DAYS)).isoformat(), *ADJUSTABLE)):
            if not conn.execute("SELECT 1 FROM paper_actions WHERE symbol = ? AND date >= ?", (s, ex)).fetchone():
                late.append(s)
    if late:
        return "a split or bonus listed late, not applied yet: " + ", ".join(sorted(set(late)))
    prev = _last_close(conn, day, held, before=True)
    low, high = MORNING_BAND
    jumped = [s for s in held if s in prev and not (low < opens[s] / prev[s] < high)]
    if jumped:
        return "opened far from its last close (a split or bonus?): " + ", ".join(jumped)
    return None


def _prev_day_of(conn, day: date) -> str:
    """The book's last evening run before ``day`` (morning_symbols has checked
    it is the previous session)."""
    return conn.execute("SELECT max(date) FROM paper_values WHERE date < ?", (day.isoformat(),)).fetchone()[0] or ""


def morning(conn, day: date, opens: dict, prev_day: date, *, costs: st.Costs | None = None) -> dict:
    """Make the trades waiting for ``day``'s open at Kite's opening prices
    (``opens``) with the evening's own steps - the same result as the evening
    when Kite's and NSE's opens agree (check_kite_fills corrects each price
    otherwise; a sale's difference is not re-spent on that day's buys). Commits."""
    with BOOK_LOCK:
        conn.commit()
        begin(conn)
        try:
            symbols = morning_symbols(conn, day, prev_day)
            if not symbols:
                conn.rollback()
                return {"day": day.isoformat(), "status": "nothing to do"}
            why = _morning_blockers(conn, day, symbols, opens)
            if why:
                conn.rollback()
                return {"day": day.isoformat(), "status": "left for the evening", "why": why}
            costs = costs or st.Costs.from_config()
            orders = fill_orders(conn, day, "open", costs, opens=opens)
            fills = execute_pending(conn, day, book_rules(conn), costs, opens=opens)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return {"day": day.isoformat(), "status": "ok", "orders": len(orders), "fills": len(fills)}


def check_kite_fills(conn, costs: st.Costs) -> list:
    """Each morning fill against NSE's official open, once NSE's file has the
    day. A different price is corrected: a buy keeps the money spent and gets
    the shares that money buys at NSE's open; a sale's proceeds are recomputed
    and the difference is entered on the cash record. Does not commit."""
    out = []
    # (29 Oct 2026 review) only a shortfall these corrections cause is covered: cash owed already
    # (a late consolidation's debit) stays owed and is repaid by the next sale
    floor = min(cash(conn), 0.0)
    for fid, day, symbol, side, shares, price, fill_price, value in conn.execute(
            "SELECT f.id, f.date, f.symbol, f.side, f.shares, f.price, f.fill_price, f.value FROM paper_fills f "
            "WHERE f.source = ? AND f.checked IS NULL AND EXISTS (SELECT 1 FROM prices p WHERE p.date = f.date "
            "AND p.source = 'nse') ORDER BY f.id", (MORNING_SOURCE,)).fetchall():
        row = conn.execute("SELECT open FROM prices WHERE symbol = ? AND date = ? AND source = 'nse'",
                           (symbol, day)).fetchone()
        nse = float(row[0]) if row and row[0] and row[0] > 0 else None
        if nse is None:
            note = "NSE has no open for this day: Kite's price kept"
        elif abs(nse - price) <= 1e-9 * nse:            # (29 Oct 2026) exact: a paisa-fraction difference is corrected too
            note = "same"
        elif side == "BUY":
            ratio = price / nse          # the same money buys ratio x the shares at the lower price
            if not conn.execute("SELECT 1 FROM paper_positions WHERE symbol = ? AND opened_on = ?",
                                (symbol, day)).fetchone():
                note = f"not corrected (sold before NSE's check): NSE open {nse:.2f}"
            else:
                conn.execute("UPDATE paper_positions SET shares = shares * ? WHERE symbol = ? AND opened_on = ?",
                             (ratio, symbol, day))
                # multiplied on what is stored now: a buy trimmed earlier in this loop keeps its trim
                conn.execute("UPDATE paper_fills SET shares = shares * ?, price = ?, fill_price = fill_price / ? "
                             "WHERE id = ?", (ratio, nse, ratio, fid))
                note = f"corrected: Kite {price:.2f}, NSE {nse:.2f}"
        else:
            received, fee, slip = sell_amounts(shares, nse, costs)
            conn.execute("UPDATE paper_fills SET price = ?, fill_price = ?, value = ?, charges = ?, slippage = ? "
                         "WHERE id = ?", (nse, nse * (1 - costs.slippage), received, fee, slip, fid))
            diff = received - value
            conn.execute("INSERT INTO paper_cash (date, kind, symbol, amount, balance) VALUES (?, 'price_check', "
                         "?, ?, ?)", (day, symbol, diff, cash(conn) + diff))
            note = f"corrected: Kite {price:.2f}, NSE {nse:.2f}"
        conn.execute("UPDATE paper_fills SET checked = ? WHERE id = ?", (note, fid))
        out.append((symbol, side, note))
        if cash(conn) < floor - 0.005:
            out += _cover_shortfall(conn, day, floor)
    return out


def _cover_shortfall(conn, day: str, floor: float = 0.0) -> list:
    """A morning sale corrected to NSE's lower open can leave the cash below zero
    when that morning's buys spent everything (28 Oct 2026 review): the newest
    morning buys of that day are made smaller by the shortfall, as if that much
    less had been spent - the cash is never negative."""
    out = []
    for fid, symbol, shares, value, charges, slip in conn.execute(
            "SELECT id, symbol, shares, value, charges, slippage FROM paper_fills WHERE date = ? AND source = ? "
            "AND side = 'BUY' ORDER BY id DESC", (day, MORNING_SOURCE)).fetchall():
        short = floor - cash(conn)
        if short <= 0.005:
            break
        held = conn.execute("SELECT shares FROM paper_positions WHERE symbol = ? AND opened_on = ?",
                            (symbol, day)).fetchone()
        if not held or value <= 0:
            continue
        cut = min(short, value * 0.999)
        f = (value - cut) / value
        conn.execute("UPDATE paper_positions SET shares = shares * ?, cost = cost * ? WHERE symbol = ? AND opened_on = ?",
                     (f, f, symbol, day))
        conn.execute("UPDATE paper_fills SET shares = ?, value = ?, charges = ?, slippage = ? WHERE id = ?",
                     (shares * f, value * f, charges * f, slip * f, fid))
        conn.execute("INSERT INTO paper_cash (date, kind, symbol, amount, balance) VALUES (?, 'price_check', ?, ?, ?)",
                     (day, symbol, cut, cash(conn) + cut))
        out.append((symbol, "BUY", f"made smaller by Rs {cut:,.2f}: a sale's NSE price left the cash short"))
    return out


def check_stops_live(conn, day: date, prices: dict, prev_day: date, *, costs: st.Costs | None = None) -> list:
    """The stop during the day: a holding whose live price is ``stop_loss`` or
    more below its buy price is sold at once at that price. Only when the
    book's rules say so, its evening runs are up to date and no trades wait
    for today's open (the evening's close check is the backup). Commits."""
    out = []
    with BOOK_LOCK:
        conn.commit()
        begin(conn)
        try:
            rules = book_rules(conn)
            last_run = conn.execute("SELECT max(date) FROM paper_values").fetchone()[0]
            waiting = conn.execute("SELECT 1 FROM paper_decisions WHERE status = 'pending' AND decided_on < ?",
                                   (day.isoformat(),)).fetchone()
            from store import db as store
            refreshed = store.get_meta(conn, "actions_refreshed_on") or ""
            if not rules or not rules.stop_loss or not rules.stop_intraday or waiting or \
                    not last_run or last_run < prev_day.isoformat() or refreshed < prev_day.isoformat():
                # (the corporate-action list not read last evening: a split today could be missing)
                conn.rollback()
                return out
            costs = costs or st.Costs.from_config()
            from datetime import datetime
            # a split/bonus today: the price is already split, the shares are scaled only in the evening
            acted = {r[0] for r in conn.execute(
                f"SELECT symbol FROM corporate_actions WHERE ex_date = ? AND kind IN "
                f"({','.join('?' * len(ADJUSTABLE))})", [day.isoformat(), *ADJUSTABLE])}
            due = {r[1] for r in conn.execute(_DUE_AT_OPEN, (day.isoformat(),) * 2)}   # sold by their order
            held = [r[0] for r in conn.execute("SELECT symbol FROM paper_positions ORDER BY symbol")]
            prev = _last_close(conn, day, held, before=True)
            for symbol in held:
                price, entry = prices.get(symbol), entry_price(conn, symbol)
                if not price or not entry or not np.isfinite(price) or price > entry * (1 - rules.stop_loss):
                    continue
                if symbol in acted or symbol in due or (symbol in prev and price / prev[symbol] <= st.JUMP_LOW):
                    continue           # (halved or worse since the last close: an unlisted split - the evening checks)
                sale = sell_now(conn, symbol, float(price), day, costs, source="live")
                conn.execute("INSERT INTO paper_orders (created_at, symbol, day, at, status, done_on, reason, "
                             "signal_day, detail) VALUES (?, ?, ?, 'live', 'done', ?, 'stop', ?, ?)",
                             (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, day.isoformat(), day.isoformat(),
                              day.isoformat(), f"live price {price:.2f}, {(price / entry - 1) * 100:.1f}% from the "
                                               f"buy price {entry:.2f}"))
                out.append(sale)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return out


def value(conn, day: date) -> tuple[float, float, int]:
    held = dict(conn.execute("SELECT symbol, shares FROM paper_positions"))
    last = adjusted_last_close(conn, day, list(held))
    money = cash(conn)
    return money + sum(sh * last.get(s, 0.0) for s, sh in held.items()), money, len(held)


_EX_DATE = re.compile(r"\bex (\d{4}-\d{2}-\d{2})")


def adjusted_last_close(conn, day: date, symbols, *, before: bool = False, through_today: bool = False) -> dict:
    """Last close on or before ``day``, on the basis of today's share count: a stock
    that did not trade on its split/bonus ex-date still shows the old price while
    its shares are already multiplied - that close is divided by the factors
    applied since (28 Oct 2026 review)."""
    out = {}
    op, upto = ("<", "<=" if through_today else "<") if before else ("<=", "<=")
    for s in symbols:
        row = conn.execute(f"SELECT close, date FROM prices WHERE symbol = ? AND date {op} ? ORDER BY date DESC "
                           f"LIMIT 1", (s, day.isoformat())).fetchone()
        if not row:
            continue
        close, on = float(row[0]), row[1]
        out[s] = close / applied_after(conn, s, on, day.isoformat(), include_day=upto == "<=")
    return out


def applied_after(conn, symbol: str, close_day: str, day: str | None = None, *, include_day: bool = True) -> float:
    """The share factor the book applied after ``close_day`` (up to ``day``) that a close
    of ``close_day`` does not hold yet: divide that close by it to put it on today's basis.
    (29 Oct 2026) An action applied late names its ex-date: a close from on or after that
    date is already on the new basis and is not divided again."""
    sql = "SELECT shares_before, shares_after, kind, date FROM paper_actions WHERE symbol = ? AND date > ?"
    args = [symbol, close_day]
    if day is not None:
        sql += " AND date <= ?" if include_day else " AND date < ?"
        args.append(day)
    f = 1.0
    for b, a, kind, applied_on in conn.execute(sql, args):
        ex = _EX_DATE.search(kind or "")
        if b and a and (ex.group(1) if ex else applied_on) > close_day:
            f *= float(a) / float(b)
    return f


def all_screened_index(conn, day: date) -> float | None:
    """All stocks on the previous session's screen, equal amounts, compounded
    from the book's start (no costs) - the same yardstick as the backtest."""
    prev_row = conn.execute("SELECT date, all_screened FROM paper_values WHERE date < ? ORDER BY date DESC LIMIT 1",
                            (day.isoformat(),)).fetchone()
    if not prev_row:
        return 1.0
    prev_day, prev_level = prev_row
    returns = pd.read_sql_query(
        "SELECT t.close / y.close AS r FROM screen s JOIN prices y ON y.symbol = s.symbol AND y.date = s.date "
        "JOIN prices t ON t.symbol = s.symbol AND t.date = ? WHERE s.date = ?", conn,
        params=(day.isoformat(), prev_day))["r"]
    returns = returns[(returns > st.JUMP_LOW) & (returns < st.JUMP_HIGH)]
    if prev_level is None or returns.empty:
        return prev_level
    return float(prev_level) * (1.0 + float((returns - 1).mean()))


def decide_today(conn, day: date, rules: st.Rules) -> int:
    held = [r[0] for r in conn.execute("SELECT symbol FROM paper_positions")]
    screened = [r[0] for r in conn.execute("SELECT symbol FROM screen WHERE date = ?", (day.isoformat(),))]
    symbols = sorted(set(held) | set(screened))
    p = pn.load(conn, day - timedelta(days=SCORE_WARMUP_DAYS), day, symbols=symbols, light=True)
    t = len(p.sessions) - 1
    if t < 0 or p.sessions[t] != day:
        raise RuntimeError(f"no prices for {day} in the v2 database")
    score = st.scores(p.close_ff, rules.lookback, p.listed)[t]
    target = st.decide(score, p.listed[t], p.close[t], rules)
    # not bought: a stock waiting to be sold, or sold by the stop or for bad news
    # within the last COOLDOWN sessions (as the backtest)
    waiting = {r[0] for r in conn.execute("SELECT symbol FROM paper_orders WHERE status = 'pending'")}
    index = {d.isoformat(): i for i, d in enumerate(p.sessions)}
    for symbol, signal in conn.execute("SELECT symbol, signal_day FROM paper_orders WHERE reason IN ('stop', 'news') "
                                       "AND status != 'cancelled' AND signal_day IS NOT NULL").fetchall():
        i = index.get(signal)
        if i is not None and t - i <= st.COOLDOWN:
            waiting.add(symbol)
    order = [p.symbols[i] for i in target.buy_order if p.symbols[i] not in waiting]
    scores = {p.symbols[i]: round(float(score[i]), 6) for i in target.buy_order}
    cur = conn.execute("INSERT INTO paper_decisions (decided_on, keep, buy_order, scores) VALUES (?, ?, ?, ?)",
                       (day.isoformat(), json.dumps(sorted(p.symbols[i] for i in target.keep)), json.dumps(order),
                        json.dumps(scores)))
    return cur.lastrowid


def evening(conn, day: date | None = None, *, capital: float | None = None, rules: st.Rules | None = None,
            costs: st.Costs | None = None, next_session=_next_session_default) -> dict:
    with BOOK_LOCK:
        return _evening(conn, day, capital=capital, rules=rules, costs=costs, next_session=next_session)


def _evening(conn, day, *, capital, rules, costs, next_session) -> dict:
    if day is None:
        day = date.fromisoformat(conn.execute(
            "SELECT max(date) FROM prices WHERE source = 'nse'").fetchone()[0])
    conn.commit()
    begin(conn)
    if conn.execute("SELECT 1 FROM paper_values WHERE date = ?", (day.isoformat(),)).fetchone():
        conn.rollback()
        return {"day": day.isoformat(), "status": "already done"}
    if not conn.execute("SELECT 1 FROM prices WHERE date = ? LIMIT 1", (day.isoformat(),)).fetchone():
        conn.rollback()
        return {"day": day.isoformat(), "status": "no prices for this day yet"}
    if not conn.execute("SELECT 1 FROM screen WHERE date = ? LIMIT 1", (day.isoformat(),)).fetchone():
        # (22 Oct 2026) without the day's screen a decision would keep nothing and sell everything
        conn.rollback()
        return {"day": day.isoformat(), "status": "no screen for this day yet"}
    costs = costs or st.Costs.from_config()
    if book_rules(conn) is None:
        from config import Config
        start_book(conn, float(capital or Config.INITIAL_CAPITAL), rules or LIVE_RULES, day)
    rules = book_rules(conn)
    summary = {"day": day.isoformat(), "status": "ok"}
    prev_done = conn.execute("SELECT max(date) FROM paper_values WHERE date < ?", (day.isoformat(),)).fetchone()[0]
    try:
        summary["price_checks"] = check_kite_fills(conn, costs)
        summary["corporate_actions"] = apply_corporate_actions(conn, day)
        summary["orders"] = len(fill_orders(conn, day, "open", costs))
        summary["fills"] = len(execute_pending(conn, day, rules, costs))
        summary["orders"] += len(fill_orders(conn, day, "close", costs))
        v, money, n = value(conn, day)
        nifty = conn.execute("SELECT close FROM index_prices WHERE index_key = 'NIFTY 50' AND date = ?",
                             (day.isoformat(),)).fetchone()
        conn.execute("INSERT INTO paper_values (date, value, cash, holdings, nifty, all_screened) "
                     "VALUES (?, ?, ?, ?, ?, ?)",
                     (day.isoformat(), v, money, n, nifty[0] if nifty else None, all_screened_index(conn, day)))
        summary.update(value=v, cash=money, holdings=n)
        summary["stops"] = check_stops(conn, day, rules, next_session)
        first_evening = conn.execute("SELECT count(*) FROM paper_decisions").fetchone()[0] == 0
        month_end = next_session(day).month != day.month
        # (22 Oct 2026) a month's last session the calendar did not know was its last
        # (the next weekday an unlisted holiday): decide at the first session after it
        last_decided = conn.execute("SELECT max(decided_on) FROM paper_decisions").fetchone()[0]
        missed = (prev_done is not None and date.fromisoformat(prev_done).month != day.month
                  and (last_decided or "") < prev_done)
        if first_evening or month_end or missed:
            summary["decision"] = decide_today(conn, day, rules)
            if missed:
                summary["decision_late"] = f"the {prev_done} month-end decision, a session late"
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return summary
