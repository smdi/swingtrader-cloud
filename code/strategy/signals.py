"""Ranking signals beyond the NSE-style score (research, 21 Oct 2026): used by
scripts/signals_check.py, scripts/frequency_check.py and the Experiments page
(strategy/experiments.py). Each takes forward-filled closes (sessions x stocks)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 252
DELIVERY_SERIES = {"EQ", "BE", "BZ"}
DELIVERY_CACHE_VERSION = 2       # 2 (22 Oct 2026): every stock of each file


def zscore(x, listed):
    """Cross-section z-score per session among the screened stocks (NaN elsewhere)."""
    x = np.where(listed & np.isfinite(x), x, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
    return (x - mean) / np.where(sd > 0, sd, np.nan)


def nearness_52w(close_ff):
    high = pd.DataFrame(close_ff).rolling(YEAR, min_periods=200).max().to_numpy()
    return close_ff / high


def smoothness(close_ff):
    """Frog in the pan: sign(12-month return) x (% up days - % down days) over a year; higher = steadier."""
    r = pd.DataFrame(close_ff).pct_change(fill_method=None)
    up = (r > 0).astype(float).where(r.notna()).rolling(YEAR, min_periods=200).mean()
    down = (r < 0).astype(float).where(r.notna()).rolling(YEAR, min_periods=200).mean()
    ret12 = pd.DataFrame(close_ff) / pd.DataFrame(close_ff).shift(YEAR) - 1
    return (np.sign(ret12) * (up - down)).to_numpy()


def residual_momentum(close_ff, index_close):
    """Sum of daily residuals (stock - beta x NIFTY) over months 2-12, / their volatility."""
    r = pd.DataFrame(close_ff).pct_change(fill_method=None)
    m = pd.Series(index_close).pct_change(fill_method=None)
    roll = lambda x: x.rolling(YEAR, min_periods=100).mean()      # ready by Jan 2010 (data from Nov 2008)
    mm = pd.DataFrame(np.repeat(m.to_numpy()[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns)
    mm = mm.where(r.notna())                           # the market over the same days as the stock
    beta = (roll(r * mm) - roll(r) * roll(mm)) / (roll(mm * mm) - roll(mm) ** 2)
    e = r - beta.mul(m, axis=0)
    total = e.rolling(YEAR - 21, min_periods=160).sum().shift(21)
    vol = e.rolling(YEAR - 21, min_periods=160).std().shift(21)
    return (total / vol).to_numpy()


def sma200_above(close_ff):
    avg = pd.DataFrame(close_ff).rolling(200, min_periods=200).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        return close_ff > avg


def _read_delivery_file(f: Path) -> dict | None:
    """``{symbol: delivery %}`` of one NSE full daily file (EQ/BE/BZ), or None if unreadable."""
    try:
        frame = pd.read_csv(f, skipinitialspace=True, dtype=str)
    except Exception:
        return None
    frame.columns = [c.strip() for c in frame.columns]
    if "DELIV_PER" not in frame or "SYMBOL" not in frame or "SERIES" not in frame:
        return None
    frame = frame[frame["SERIES"].str.strip().isin(DELIVERY_SERIES)]
    per = pd.to_numeric(frame["DELIV_PER"].str.strip(), errors="coerce")
    return {s: float(v) for s, v in zip(frame["SYMBOL"].str.strip(), per) if v == v}


def delivery_matrix(sessions, symbols, folder: Path, cache: Path | None = None) -> np.ndarray:
    """(session, stock) delivery % from NSE's full daily files; NaN where missing.
    ``cache``: an .npz holding, for every day already read, EVERY stock in that
    day's file (22 Oct 2026: it held only the stocks asked for then, so a stock
    new to the list had no history for the days already cached)."""
    days, names, values = [], [], np.zeros((0, 0))
    if cache is not None and cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                if int(z["version"]) != DELIVERY_CACHE_VERSION:
                    raise ValueError("an older cache")     # it held only the stocks asked for: read again
                days, names, values = list(z["days"]), list(z["symbols"]), z["deliv"]   # read once
        except Exception:
            days, names, values = [], [], np.zeros((0, 0))       # a damaged cache: read the files again
    row = {d: i for i, d in enumerate(days)}
    name_col = {s: j for j, s in enumerate(names)}
    new_days = []
    for d in sessions:
        iso = d.isoformat()
        if iso in row:
            continue
        f = folder / f"sec_bhavdata_full_{d:%d%m%Y}.csv"
        if not f.exists():
            continue
        got = _read_delivery_file(f)
        if got is not None:
            new_days.append((iso, got))
    if new_days:
        for _, got in new_days:
            for s in got:
                if s not in name_col:
                    name_col[s] = len(names)
                    names.append(s)
        grown = np.full((len(days) + len(new_days), len(names)), np.nan)
        if values.size:
            grown[:len(days), :values.shape[1]] = values
        for k, (iso, got) in enumerate(new_days):
            i = len(days) + k
            row[iso] = i
            idx = np.fromiter((name_col[s] for s in got), dtype=int, count=len(got))
            grown[i, idx] = np.fromiter(got.values(), dtype=float, count=len(got))
        days = days + [iso for iso, _ in new_days]
        values = grown
        if cache is not None:
            import os
            tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.part.npz")
            try:
                np.savez_compressed(tmp, days=np.array(days), symbols=np.array(names), deliv=values,
                                    version=np.array(DELIVERY_CACHE_VERSION))
                os.replace(tmp, cache)
            except OSError:          # (Windows: the cache open elsewhere) the result stands; saved next time
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    out = np.full((len(sessions), len(symbols)), np.nan)
    pairs = [(j, name_col[s]) for j, s in enumerate(symbols) if s in name_col]
    if pairs and values.size:
        dst = np.array([p[0] for p in pairs], dtype=int)
        src = np.array([p[1] for p in pairs], dtype=int)
        for t, d in enumerate(sessions):
            i = row.get(d.isoformat())
            if i is not None:
                out[t, dst] = values[i, src]
    return out


def rolling_mean(x: np.ndarray, n: int, min_periods: int) -> np.ndarray:
    return pd.DataFrame(x).rolling(n, min_periods=min_periods).mean().to_numpy()
