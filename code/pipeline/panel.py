"""Read prices from the v2 database as adjusted matrices (Redesign v2, Phase 2).

Adjustment, in one basis for every day:
  * NSE days: raw price / (product of split, bonus and consolidation factors
    with an ex-date after that day), from NSE's corporate actions;
  * days before NSE's files (legacy): the old database's split-adjusted price
    times the stock's join factor (lines it up with NSE's raw price on the
    first NSE days), then the same division for actions after the join.
So a stock's series is continuous across splits and across the join.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

ADJUSTABLE = ("split", "bonus", "consolidation")
SPLICE_SESSIONS = 20          # first NSE sessions used to line up the old history
MIN_STOCKS_A_DAY = 50         # fewer rows on a date: not a market day (stray rows)


@dataclass
class Panel:
    sessions: list            # date objects, oldest first
    symbols: list
    open: np.ndarray          # adjusted, NaN when not traded
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    close_ff: np.ndarray      # adjusted close carried forward over gaps
    raw_close: np.ndarray     # as published (NSE days) or as stored (legacy)
    volume: np.ndarray        # adjusted volume (shares x factor)
    turnover: np.ndarray      # rupees (NSE days); close x volume on legacy days
    listed: np.ndarray        # on that day's screen
    index_close: np.ndarray   # NIFTY 50, carried forward
    index_open: np.ndarray

    @property
    def days(self) -> np.ndarray:
        return np.array([d.toordinal() for d in self.sessions])


def splice_factors(conn) -> dict:
    """Per stock: the factor that puts the old (adjusted) history on NSE's raw
    basis at the join.

    Exact method: NSE's first file for the stock carries the previous day's
    close (PREV_CLOSE, on NSE's basis); divided by the old history's close on
    that previous day it gives the factor - no overlap needed (14 Oct: the old
    history of HDFC Bank and Aarti was on one basis before 23 Aug 2019 and on
    another after, so lining up on the days after left a x2 jump).
    Fallback (no previous close, or no old price within a week before): the
    median ratio over the first SPLICE_SESSIONS shared sessions, counting only
    days before the stock's first split/bonus there."""
    first = conn.execute("SELECT min(date) FROM prices WHERE source = 'nse'").fetchone()[0]
    if first is None:
        return {}
    starts = pd.read_sql_query(
        "SELECT p.symbol, p.date, p.prev_close FROM prices p JOIN (SELECT symbol, min(date) AS d FROM prices "
        "WHERE source = 'nse' GROUP BY symbol) f ON f.symbol = p.symbol AND f.d = p.date "
        "WHERE p.symbol IN (SELECT DISTINCT symbol FROM legacy_prices WHERE date < ?)", conn, params=(first,))
    week_before = date.fromordinal(date.fromisoformat(first).toordinal() - 10).isoformat()
    old = pd.read_sql_query("SELECT symbol, date, close FROM legacy_prices WHERE date < ? AND date >= ?",
                            conn, params=(first, week_before))
    last_old = old.sort_values("date").groupby("symbol").last()
    exact = {}
    for row in starts.itertuples(index=False):
        if row.symbol not in last_old.index or not row.prev_close or row.prev_close <= 0:
            continue
        if row.date != first:          # first traded later than NSE's first file: the day before is not old history
            continue
        ref = last_old.loc[row.symbol]
        if ref["close"] > 0:
            exact[row.symbol] = float(row.prev_close) / float(ref["close"])
    out = _median_factors(conn, first)
    out.update(exact)
    return out


def _median_factors(conn, first) -> dict:
    days = [r[0] for r in conn.execute(
        "SELECT date FROM prices WHERE date >= ? GROUP BY date HAVING count(*) >= ? ORDER BY date LIMIT ?",
        (first, MIN_STOCKS_A_DAY, SPLICE_SESSIONS))]
    if not days:
        return {}
    marks = ",".join("?" * len(days))
    both = pd.read_sql_query(
        f"SELECT p.symbol, p.date, p.close AS nse, l.close AS old FROM prices p JOIN legacy_prices l "
        f"ON l.symbol = p.symbol AND l.date = p.date WHERE p.date IN ({marks})", conn, params=days)
    both = both[(both["old"] > 0) & (both["nse"] > 0)]
    acts = pd.read_sql_query(
        "SELECT symbol, min(ex_date) AS ex FROM corporate_actions WHERE kind IN "
        "('split', 'bonus', 'consolidation') AND ex_date > ? AND ex_date <= ? GROUP BY symbol",
        conn, params=(days[0], days[-1]))
    cut = both["symbol"].map(dict(zip(acts["symbol"], acts["ex"])))
    both = both[cut.isna() | (both["date"] < cut)]
    return (both["nse"] / both["old"]).groupby(both["symbol"]).median().to_dict()


def refresh_splice(conn) -> int:
    """Recompute the join factors in place (no rebuild needed)."""
    factors = splice_factors(conn)
    conn.execute("DELETE FROM legacy_splice")
    conn.executemany("INSERT INTO legacy_splice (symbol, factor, days) VALUES (?, ?, NULL)",
                     [(k, float(v)) for k, v in factors.items()])
    conn.commit()
    return len(factors)


def drop_stray_yahoo(old: pd.DataFrame, window_days: int = 10) -> pd.DataFrame:
    """The old database mixed sources: Kite's history, with Yahoo prices on
    scattered days, sometimes on another basis (14 Oct: SUVEN x20 on 23 Oct
    2017 and 9 Nov 2018). A Yahoo row is dropped when the same stock has a
    Kite row within ``window_days``; stocks known only from Yahoo keep theirs."""
    if old.empty or "source" not in old:
        return old
    yahoo = old["source"].fillna("").str.startswith("yfinance")
    if not yahoo.any():
        return old.drop(columns="source")
    day = pd.to_datetime(old["date"]).map(pd.Timestamp.toordinal).to_numpy()
    drop = np.zeros(len(old), dtype=bool)
    kite = ~yahoo.to_numpy()
    for symbol, idx in old.groupby("symbol").indices.items():
        k_days = np.sort(day[idx][kite[idx]])
        y_idx = idx[~kite[idx]]
        if not len(k_days) or not len(y_idx):
            continue
        y_days = day[y_idx]
        pos = np.searchsorted(k_days, y_days)
        before = np.abs(y_days - k_days[np.clip(pos - 1, 0, len(k_days) - 1)])
        after = np.abs(k_days[np.clip(pos, 0, len(k_days) - 1)] - y_days)
        drop[y_idx] = np.minimum(before, after) <= window_days
    return old[~drop].drop(columns="source").reset_index(drop=True)


def divisors(actions: pd.DataFrame, sessions: list, symbols: list) -> np.ndarray:
    """(session, stock) divisor: product of adjustable factors whose ex-date
    is after that session."""
    div = np.ones((len(sessions), len(symbols)))
    position = {s: i for i, s in enumerate(symbols)}
    iso = np.array([d.isoformat() for d in sessions])
    for row in actions.itertuples(index=False):
        j = position.get(row.symbol)
        if j is None or row.kind not in ADJUSTABLE or not row.factor or not np.isfinite(row.factor) \
                or row.factor <= 0:
            continue
        cut = int(np.searchsorted(iso, row.ex_date))       # first session on/after the ex-date
        div[:cut, j] *= float(row.factor)
    return div


def load(conn, start: date, end: date, *, symbols=None, equities_only: bool = True,
         light: bool = False) -> Panel:
    """``light``: skip high, low, volume and turnover (all NaN) - the backtest needs only open and close."""
    s, e = start.isoformat(), end.isoformat()
    nse_start = conn.execute("SELECT min(date) FROM prices WHERE source = 'nse'").fetchone()[0] or "9999"
    kinds = "AND symbol IN (SELECT symbol FROM instruments WHERE kind = 'equity')" if equities_only else ""
    few = list(symbols) if symbols is not None and len(symbols) <= 300 else None   # a few stocks: read only theirs
    only = f"AND symbol IN ({','.join('?' * len(few))})" if few else ""
    new = pd.read_sql_query(
        f"SELECT symbol, date, open, high, low, close, volume, turnover FROM prices WHERE date BETWEEN ? AND ? "
        f"{kinds} {only}", conn, params=(max(s, nse_start), e, *(few or [])))
    # (29 Oct 2026) the old prices fill every day NSE's files lack - before they start (23 Aug 2019)
    # and the 24 sessions of Aug-Sep 2019 that only the old prices have (they were dropped before)
    # (29 Oct 2026 review) a date with only a few stray NSE rows is not an NSE day: its old prices stay
    nse_days = {r[0] for r in conn.execute("SELECT date FROM prices WHERE date BETWEEN ? AND ? GROUP BY date "
                                           "HAVING count(*) >= ?", (s, e, MIN_STOCKS_A_DAY))}
    old = pd.read_sql_query(
        f"SELECT symbol, date, open, high, low, close, volume, source FROM legacy_prices "
        f"WHERE date BETWEEN ? AND ? {only}", conn, params=(s, e, *(few or [])))
    old = old[~old["date"].isin(nse_days)]
    old = drop_stray_yahoo(old)
    if len(old):
        factors = {r[0]: r[1] for r in conn.execute("SELECT symbol, factor FROM legacy_splice")}
        f = old["symbol"].map(factors).fillna(1.0)
        for col in ("open", "high", "low", "close"):
            old[col] *= f
        old["volume"] /= f
        old["turnover"] = old["close"] * old["volume"]
    frame = pd.concat([old, new], ignore_index=True)
    frame = frame.drop_duplicates(["symbol", "date"], keep="last")      # NSE's row over the old one
    if symbols is not None:
        frame = frame[frame["symbol"].isin(set(symbols))]
    # Market days are judged on every stock in the database, not on the
    # stocks asked for (14 Oct test: asking for two stocks dropped every day).
    market = {r[0] for r in conn.execute(
        "SELECT date FROM prices WHERE date BETWEEN ? AND ? GROUP BY date HAVING count(*) >= ?",
        (max(s, nse_start), e, MIN_STOCKS_A_DAY))}
    market |= {r[0] for r in conn.execute(
        "SELECT date FROM legacy_prices WHERE date BETWEEN ? AND ? GROUP BY date HAVING count(*) >= ?",
        (s, e, MIN_STOCKS_A_DAY)) if r[0] not in nse_days}
    days = sorted(market)              # market days: a stock missing that day gets NaN, the day stays
    frame = frame[frame["date"].isin(set(days))]
    sessions = [date.fromisoformat(d) for d in days]
    names = sorted(frame["symbol"].unique()) if symbols is None else list(symbols)
    # Placed by index arithmetic (a pivot table of millions of rows is slow and
    # memory-hungry on a PC); a later row for the same day and stock wins.
    ri = pd.Index(days).get_indexer(frame["date"])
    ci = pd.Index(names).get_indexer(frame["symbol"])
    ok = (ri >= 0) & (ci >= 0)
    ri, ci = ri[ok], ci[ok]

    def wide(col):
        out = np.full((len(days), len(names)), np.nan)
        if light and col in ("high", "low", "volume", "turnover"):
            return out
        out[ri, ci] = frame[col].to_numpy(float)[ok]
        return out

    raw_open, raw_high, raw_low, raw_close = wide("open"), wide("high"), wide("low"), wide("close")
    volume, turnover = wide("volume"), wide("turnover")
    actions = pd.read_sql_query("SELECT symbol, ex_date, kind, factor FROM corporate_actions", conn)
    # Legacy days are already in the join basis: only actions after the join apply.
    after_join = actions[actions["ex_date"] > nse_start] if len(old) else actions
    div = divisors(actions, sessions, names)
    if len(old):
        legacy_rows = np.array([d not in nse_days for d in days], dtype=bool)
        div[legacy_rows] = divisors(after_join, sessions, names)[legacy_rows]
    adj_open, adj_close = raw_open / div, raw_close / div
    close_ff = pd.DataFrame(adj_close).ffill().to_numpy()
    screen = pd.read_sql_query("SELECT date, symbol FROM screen WHERE date BETWEEN ? AND ?", conn, params=(s, e))
    listed = np.zeros_like(adj_close, dtype=bool)
    if len(screen):
        di = {d: i for i, d in enumerate(days)}
        si = {x: i for i, x in enumerate(names)}
        rows = screen["date"].map(di)
        cols = screen["symbol"].map(si)
        ok = rows.notna() & cols.notna()
        listed[rows[ok].astype(int).to_numpy(), cols[ok].astype(int).to_numpy()] = True
    idx = pd.read_sql_query("SELECT date, open, close FROM index_prices WHERE index_key = 'NIFTY 50' "
                            "AND date BETWEEN ? AND ?", conn, params=(s, e)).set_index("date")
    index_close = idx["close"].reindex(days).ffill().to_numpy(float)
    index_open = idx["open"].reindex(days).to_numpy(float)
    index_open = np.where(np.isfinite(index_open) & (index_open > 0), index_open,
                          np.r_[index_close[:1], index_close[:-1]])
    return Panel(sessions, names, adj_open, raw_high / div, raw_low / div, adj_close, close_ff, raw_close, volume * div,
                 turnover, listed, index_close, index_open)
