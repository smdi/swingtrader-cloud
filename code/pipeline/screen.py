"""The stocks considered each day (Redesign v2, Phase 2).

From LIVE_FROM on, the screen is the live rules (universe_mask below: Rs 5 crore median turnover, price >= Rs 20, a year listed,
daily range 1-6%, few gaps, the least liquid tenth dropped) applied to NSE's
files - the screen the strength checks used (+35.5%/yr on 2020-26). Before
that, the old app's daily lists (scanner_rankings) are imported unchanged -
the lists the 2010-2019 tests used.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

LIVE_FROM = date(2020, 1, 1)
WARMUP_DAYS = 420            # calendar days of history the live rules need


def live_screen(conn, start: date, end: date) -> pd.DataFrame:
    """(date, symbol) rows passing the live rules on NSE days in [start, end]."""
    from config import Config
    from pipeline import panel as pn
    first = max(start, LIVE_FROM)
    if first > end:
        return pd.DataFrame(columns=["date", "symbol"])
    p = pn.load(conn, first - timedelta(days=WARMUP_DAYS), end)
    nse_start = conn.execute("SELECT min(date) FROM prices WHERE source = 'nse'").fetchone()[0]
    keep = np.array([d.isoformat() >= nse_start for d in p.sessions])
    idx = [d for d, k in zip(p.sessions, keep) if k]
    frame = lambda m: pd.DataFrame(m[keep], index=idx, columns=p.symbols)
    mask = universe_mask(frame(p.raw_close), frame(p.open), frame(p.high), frame(p.low), frame(p.close),
                            frame(p.turnover), frame(p.volume),
                            floor=float(Config.SCANNER_EOD_MIN_MEDTV_RUPEES))
    mask = mask.loc[[d for d in mask.index if first <= d <= end]]
    stacked = mask.stack()
    rows = stacked[stacked].reset_index()
    rows.columns = ["date", "symbol", "_"]
    rows["date"] = rows["date"].map(lambda d: d.isoformat())
    return rows[["date", "symbol"]]


def wilder_atr_wide(high, low, close, period: int = 14) -> pd.DataFrame:
    """Wilder ATR per column (SMA seed); a missing session restarts the run."""
    h, l, c = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    with np.errstate(invalid="ignore"):
        tr = np.fmax(np.fmax(h - l, np.abs(h - prev)), np.abs(l - prev))
    tr[~np.isfinite(h - l)] = np.nan
    res = np.full(tr.shape, np.nan)
    for j in range(tr.shape[1]):
        col = tr[:, j]
        prev_atr, count = np.nan, 0
        for i in range(len(col)):
            value = col[i]
            if not np.isfinite(value):
                prev_atr, count = np.nan, 0
                continue
            count += 1
            if count < period:
                continue
            if count == period:
                prev_atr = col[i - period + 1:i + 1].mean()
            else:
                prev_atr = (prev_atr * (period - 1) + value) / period
            res[i, j] = prev_atr
    return pd.DataFrame(res, index=close.index, columns=close.columns)


def universe_mask(raw_close, adj_open, adj_high, adj_low, adj_close, turnover, volume, *,
                  floor: float = 5e7, min_price: float = 20.0, illiq_drop: float = 0.10,
                  min_listing_sessions: int = 250) -> pd.DataFrame:
    """The screen's rules per (day, stock): liquid (20-day median turnover >= floor,
    traded 19 of 20 days), price >= min_price, ATR 1-6% of price, at most 3
    opening gaps > 1.5 ATR in 60 days, listed a year; the least liquid tenth
    (price impact per rupee traded) is dropped each day."""
    medtv = turnover.fillna(0).rolling(20).median()
    active = (volume.fillna(0) > 0).rolling(20).sum()
    atr = wilder_atr_wide(adj_high, adj_low, adj_close, 14)
    atr_pct = atr / adj_close * 100.0
    gap = ((adj_open - adj_close.shift(1)).abs() > 1.5 * atr.shift(1)).astype(float)
    gaps60 = gap.rolling(60, min_periods=60).sum()
    ret = (adj_close / adj_close.shift(1) - 1).abs()
    illiq = (ret / (turnover / 1e7)).replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=15).mean()
    mask = (
        (raw_close >= min_price) & (medtv >= floor) & (active >= 19)
        & (atr_pct >= 1.0) & (atr_pct <= 6.0) & (gaps60 <= 3) & illiq.notna()
    )
    # Listed a year: past listing dates of delisted stocks are not available,
    # so the first appearance in the files counts; stocks there from the start count as listed.
    seen = raw_close.notna().cumsum()
    first_index = raw_close.notna().idxmax()
    position = pd.Series(np.arange(len(raw_close)), index=raw_close.index)
    first_pos = pd.Series({c: raw_close.index.get_loc(first_index[c]) for c in raw_close.columns})
    listed = pd.DataFrame(position.to_numpy()[:, None] - first_pos.to_numpy()[None, :] >= min_listing_sessions,
                          index=raw_close.index, columns=raw_close.columns)
    listed |= pd.DataFrame(np.broadcast_to((first_pos == 0).to_numpy(), raw_close.shape),
                           index=raw_close.index, columns=raw_close.columns)
    mask &= listed & (seen > 0)
    ranks = illiq.where(mask).rank(axis=1, ascending=False, method="first")
    counts = mask.sum(axis=1)
    drop = np.floor(counts * illiq_drop)
    return mask & ~(ranks.le(drop, axis=0))


def write_screen(conn, rows: pd.DataFrame, source: str, start: date, end: date) -> int:
    """Replace the screen for [start, end] with ``rows``."""
    conn.execute("DELETE FROM screen WHERE date BETWEEN ? AND ? AND source = ?",
                 (start.isoformat(), end.isoformat(), source))
    conn.executemany("INSERT OR REPLACE INTO screen (date, symbol, source) VALUES (?, ?, ?)",
                     [(d, s, source) for d, s in zip(rows["date"], rows["symbol"])])
    return len(rows)
