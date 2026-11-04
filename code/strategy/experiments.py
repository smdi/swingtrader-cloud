"""Experiments (21 Oct 2026): pretend portfolios that run other rules beside
the paper book, to see live whether they behave as their backtests said.

Each experiment starts with CAPITAL on the same day (meta experiments_started_on,
the first evening they run), decides at that close and then by its own rules.
Every evening, once NSE's file for a session is in, each one is replayed from
its start with the backtest's own function (strategy.strength.simulate) on
NSE's prices: decide at a close, buy and sell at the next official open
(Kite's open matched NSE's on 763 of 763 checks), the 25% stop at the close,
all charges and slippage. The replay is deterministic, so a crash or a missed
evening loses nothing; the results are stored in the exp_ tables, which the
paper book never reads. A replay that changes a trade already shown (NSE
corrected an old price) is logged and noted on the page.

What an experiment does not do: news sales and "Sell now" (those are the real
book's), and Kite's morning fill (the official open is used).

Judging rule, fixed before any result: an experiment replaces the live rules
only after at least 6-12 months and when it is no worse on both return and
worst fall than "Current rules (copy)" - and it passed the backtest. A few
months ahead or behind is mostly luck.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from strategy import signals, strength as st

CAPITAL = 100_000.0
WARMUP_DAYS = 800        # calendar days of prices before the start: residual momentum needs about 2 years


@dataclass(frozen=True)
class Experiment:
    key: str
    name: str
    about: str
    rules: st.Rules
    score: str = "nse"                # "nse" (the live score), "residual" or "nse+delivery"
    near_high: float | None = None    # buy only within this fraction of the 52-week high
    backtest: tuple = ()              # (period 1 %/yr, worst fall %, period 2 %/yr, worst fall %, charges %/yr)
    periods: tuple = ("2010–18", "2019–26")    # the backtest's two periods


EXPERIMENTS = (
    Experiment("copy", "Current rules (copy)",
               "The live rules, started on the same day as the experiments - the fair yardstick. "
               "It differs from the real portfolio only by news sales, Sell now and start date.",
               st.Rules(sector_cap=4, stop_loss=0.25), backtest=(27.2, -31, 36.6, -33, 1.4)),
    Experiment("residual_weekly", "Residual momentum, weekly",
               "The 12-month rise after taking out what NIFTY explains, skipping the last month, divided by "
               "its own volatility (Blitz, Huij & Martens 2011); decided at the end of every week.",
               st.Rules(rebalance="week_end", sector_cap=4, stop_loss=0.25), score="residual",
               backtest=(26.2, -30, 35.9, -30, 2.1)),
    Experiment("near_high", "Near the 1-year high",
               "The live rules, but a stock is bought only within 15% of its highest close of the last year "
               "(George & Hwang 2004); holdings are kept as now.",
               st.Rules(sector_cap=4, stop_loss=0.25), near_high=0.15, backtest=(25.4, -30, 36.3, -33, 1.4)),
    Experiment("delivery", "Score + delivery %",
               "The NSE-style score averaged with the 60-day average delivery % (shares taken home, not "
               "day-traded), each as a z-score across the screen. NSE's delivery data starts in 2019, so its "
               "backtest covers only 2021-26 (added 22 Oct 2026).",
               st.Rules(sector_cap=4, stop_loss=0.25), score="nse+delivery",
               backtest=(35.4, -22, 35.7, -26, 1.4), periods=("2021–23", "2023–26")),
)
DELIVERY_DAYS, DELIVERY_MIN = 60, 40   # the 60-session average delivery %, from 40 sessions with data
BY_KEY = {e.key: e for e in EXPERIMENTS}


def _latest_nse(conn) -> date | None:
    d = conn.execute("SELECT max(date) FROM prices WHERE source = 'nse'").fetchone()[0]
    return date.fromisoformat(d) if d else None


def _research_folder():
    from config import BASE_DIR
    from pathlib import Path
    return Path(BASE_DIR) / "data" / "nse_research"


def _score_matrices(p, research=None):
    nse = st.scores(p.close_ff, "nse", p.listed)
    cache = {}
    out = {}
    for e in EXPERIMENTS:
        if e.score == "nse":
            out[e.key] = nse
        elif e.score == "residual":
            if "residual" not in cache:
                cache["residual"] = signals.zscore(signals.residual_momentum(p.close_ff, p.index_close), p.listed)
            out[e.key] = cache["residual"]
        elif e.score == "nse+delivery":
            folder = research or _research_folder()
            # NSE's daily files, kept by the evening update (pipeline.nse_daily.read_bhav) in nse_research\bhav
            deliv = signals.delivery_matrix(p.sessions, p.symbols, folder / "bhav",
                                            cache=folder / "delivery_cache_experiments.npz")
            avg = signals.rolling_mean(deliv, DELIVERY_DAYS, DELIVERY_MIN)
            out[e.key] = (signals.zscore(nse, p.listed) + signals.zscore(avg, p.listed)) / 2
        else:
            raise ValueError(f"unknown score {e.score!r}")
    return out


def _no_buy(p, e: Experiment):
    if e.near_high is None:
        return None
    near = signals.nearness_52w(p.close_ff)
    with np.errstate(invalid="ignore"):
        return ~(near >= 1 - e.near_high)


def replay(conn, start: date, end: date, next_session, *, costs: st.Costs | None = None, research=None) -> dict:
    """Each experiment from ``start`` to ``end`` (sessions with NSE's prices).
    Returns {key: result rows} - nothing is written."""
    from pipeline import panel as pn
    from store import db as store
    from strategy import backtest as bt
    costs = costs or st.Costs.from_config()
    symbols = bt.screened_symbols(conn, start, end)
    p = pn.load(conn, start - timedelta(days=WARMUP_DAYS), end, symbols=symbols, light=True)
    first = int(np.searchsorted(p.days, start.toordinal()))
    if first >= len(p.sessions) or p.sessions[first] != start or p.sessions[-1] != end:
        raise RuntimeError(f"no prices for the experiments' start {start} or for {end}")
    groups = store.sector_groups(conn, p.symbols)
    sessions = list(p.sessions) + [next_session(end)]
    score = _score_matrices(p, research)
    out = {}
    for e in EXPERIMENTS:
        res = st.simulate(sessions, p.open, p.close, p.close_ff, p.listed, rules=e.rules, costs=costs,
                          capital=CAPITAL, start_index=first, score_matrix=score[e.key], groups=groups,
                          no_buy=_no_buy(p, e), decide_first=True, open_end=True)
        out[e.key] = _rows(p, first, res, e, costs, groups)
    return out


def _as_traded(p):
    """Per (session, stock): the factor from the panel's adjusted prices back
    to the prices traded that day. The panel divides earlier prices by every
    split or bonus it knows of - also one announced for a later date - so a
    fill or a holding is shown as it was (or is) on the exchange."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f = p.raw_close / p.close
    f = np.where(np.isfinite(f) & (f > 0), f, np.nan)
    import pandas as pd
    return pd.DataFrame(f).ffill().fillna(1.0).to_numpy()


def _rows(p, first, res, e: Experiment, costs, groups) -> dict:
    T = len(p.sessions)
    factor = _as_traded(p)
    cash, held, reasons = _cash_path(p, first, res, e.rules.stop_loss)
    values = [(p.sessions[t].isoformat(), round(float(res.values[t]), 2), cash[t], held[t]) for t in range(first, T)]
    fills, bought_on = [], {}
    for i, (t, f) in enumerate(res.fills):
        sym = p.symbols[f.stock]
        k = factor[t, f.stock]
        fills.append((p.sessions[t].isoformat(), sym, f.side, round(f.shares / k, 6), round(f.price * k, 4),
                      round(f.value, 2), round(f.charges, 2), reasons[i]))
        if f.side == "BUY":
            bought_on[sym] = p.sessions[t].isoformat()
    book = res.book
    positions = []
    for s in np.flatnonzero(book.shares > 0):
        sym = p.symbols[s]
        k = factor[-1, s]
        buy = float(res.entry[s]) * k if res.entry is not None and np.isfinite(res.entry[s]) else None
        positions.append((sym, round(float(book.shares[s]) / k, 6), buy, bought_on.get(sym),
                          float(p.close_ff[-1, s]) * k))
    return {"values": values, "fills": fills, "positions": positions,
            "planned": _planned(p, res, e, costs, groups)}


class _Shares:
    """The part of a Book that carry_corporate_action uses."""
    def __init__(self, shares):
        self.shares = shares


def _cash_path(p, first, res, stop_loss) -> tuple[dict, dict, list]:
    """Cash and the number of holdings after each session, rebuilt from the
    fills and the split adjustments (simulate keeps only the final book), and
    why each fill was made: a sale of a holding whose previous close was at or
    below its stop price is the stop (also on a decision day); the rest are
    the decision's."""
    T, S = p.close.shape
    book = _Shares(np.zeros(S))
    entry = np.full(S, np.nan)
    money = CAPITAL
    by_t = {}
    for i, (t, f) in enumerate(res.fills):
        by_t.setdefault(t, []).append((i, f))
    cash, held, reasons = {}, {}, [""] * len(res.fills)
    for t in range(first, T):
        if t > first:
            ratio = p.close_ff[t] / p.close_ff[t - 1]
            with np.errstate(invalid="ignore"):
                bad = (book.shares > 0) & np.isfinite(ratio) & ~((ratio > st.JUMP_LOW) & (ratio < st.JUMP_HIGH))
            entry[bad] *= ratio[bad]
            st.carry_corporate_action(book, ratio)
        for i, f in by_t.get(t, []):
            if f.side == "BUY":
                book.shares[f.stock] = f.shares
                entry[f.stock] = f.fill_price
                money -= f.value
                reasons[i] = "decision"
            else:
                hit = (stop_loss and t > first and np.isfinite(entry[f.stock])
                       and p.close_ff[t - 1, f.stock] <= entry[f.stock] * (1 - stop_loss))
                reasons[i] = "stop" if hit else "decision"
                book.shares[f.stock] = 0.0
                entry[f.stock] = np.nan
                money += f.value
        cash[t] = round(money, 2)
        held[t] = int(np.count_nonzero(book.shares > 0))
    return cash, held, reasons


def _planned(p, res, e: Experiment, costs, groups) -> list:
    """The trades a decision (or a stop) at the last close makes at the next
    open, estimated at that close (the open decides the shares): only stocks
    that traded today are bought, as at a real open."""
    if res.pending is None:
        return []
    book = copy.deepcopy(res.book)
    report = st.execute(book, res.pending, p.close[-1], p.close_ff[-1], e.rules, costs, groups)
    stop = e.rules.stop_loss
    out = []
    for f in report.fills:
        hit = (f.side == "SELL" and stop and res.entry is not None and np.isfinite(res.entry[f.stock])
               and p.close_ff[-1, f.stock] <= res.entry[f.stock] * (1 - stop))
        out.append((p.symbols[f.stock], f.side, p.sessions[-1].isoformat(), "stop" if hit else "decision"))
    return out


def _inputs_missing(conn, end: date) -> str | None:
    """NIFTY's close and the day's screen arrive with NSE's prices; without
    them the replay waits (a missing screen would sell everything)."""
    d = end.isoformat()
    if not conn.execute("SELECT 1 FROM screen WHERE date = ? LIMIT 1", (d,)).fetchone():
        return "the screen"
    if not conn.execute("SELECT 1 FROM index_prices WHERE index_key = 'NIFTY 50' AND date = ?", (d,)).fetchone():
        return "NIFTY 50's close"
    return None


def update(conn, *, next_session, log=print, force: bool = False, today_start: date | None = None) -> dict:
    """The evening step: replay every experiment up to the newest session with
    NSE's prices and store it. Cheap when already done for that session."""
    from store import db as store
    from strategy import live
    end = _latest_nse(conn)
    if end is None:
        return {"status": "no data"}
    started = store.get_meta(conn, "experiments_started_on")
    start = date.fromisoformat(started) if started else (today_start or end)
    shown = {r[0] for r in conn.execute("SELECT DISTINCT exp FROM exp_values")}
    if (not force and store.get_meta(conn, "experiments_done_for") == end.isoformat()
            and all(e.key in shown for e in EXPERIMENTS)):        # (an experiment added since runs at once)
        return {"status": "already done", "for": end.isoformat()}
    if start > end:
        return {"status": "not started", "start": start.isoformat()}
    missing = _inputs_missing(conn, end)
    if missing:
        return {"status": f"waiting for {missing}", "for": end.isoformat()}
    if not force and store.get_meta(conn, "experiments_failed_for") == end.isoformat():
        return {"status": "failed for this session; tried again with the next one", "for": end.isoformat()}
    try:
        out = replay(conn, start, end, next_session)
    except Exception:          # the replay only reads; the next try is with the next session, not in 15 minutes
        store.set_meta(conn, "experiments_failed_for", end.isoformat())
        conn.commit()
        raise
    live.begin(conn)
    try:
        if not started:
            store.set_meta(conn, "experiments_started_on", start.isoformat())
        changed = _changed_past_fills(conn, out, store.get_meta(conn, "experiments_done_for"))
        for key, rows in out.items():
            for table in ("exp_values", "exp_positions", "exp_fills", "exp_planned"):
                conn.execute(f"DELETE FROM {table} WHERE exp = ?", (key,))
            conn.executemany("INSERT INTO exp_values VALUES (?, ?, ?, ?, ?)", [(key, *r) for r in rows["values"]])
            conn.executemany("INSERT INTO exp_positions VALUES (?, ?, ?, ?, ?, ?)",
                             [(key, *r) for r in rows["positions"]])
            conn.executemany("INSERT INTO exp_fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             [(key, *r) for r in rows["fills"]])
            conn.executemany("INSERT OR REPLACE INTO exp_planned VALUES (?, ?, ?, ?, ?)",
                             [(key, *r) for r in rows["planned"]])
        if changed:            # shown until the next replay
            note = (f"{end:%d %b %Y}: the replay changed {changed} trade(s) shown before "
                    f"(an earlier day's data changed: NSE's prices, or a sector added since)")
            store.set_meta(conn, "experiments_note", note)
            log("experiments: " + note)
        else:
            conn.execute("DELETE FROM meta WHERE key = 'experiments_note'")
        store.set_meta(conn, "experiments_done_for", end.isoformat())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    summary = {"status": "ok", "for": end.isoformat(), "start": start.isoformat(),
               "values": {k: round(v["values"][-1][1]) for k, v in out.items()},
               "trades": {k: len(v["fills"]) for k, v in out.items()}}
    if changed:
        summary["changed_past_trades"] = changed
    return summary


def _changed_past_fills(conn, out, done_for: str | None) -> int:
    """Trades on days already shown (up to the last replay) that differ now:
    one missing, new, or of other shares or rupees (both as traded that day, so
    a split since changes neither)."""
    if not done_for:
        return 0
    changed = 0
    for key, rows in out.items():
        if not conn.execute("SELECT 1 FROM exp_values WHERE exp = ? LIMIT 1", (key,)).fetchone():
            continue                                   # an experiment added since: nothing was shown before
        old = {(r[0], r[1], r[2]): (round(r[3], 2), round(r[4])) for r in conn.execute(
            "SELECT date, symbol, side, shares, value FROM exp_fills WHERE exp = ? AND date <= ?", (key, done_for))}
        new = {(r[0], r[1], r[2]): (round(r[3], 2), round(r[5])) for r in rows["fills"] if r[0] <= done_for}
        changed += sum(1 for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    return changed
