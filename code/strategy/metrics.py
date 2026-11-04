"""Performance measures for the strength portfolio (Redesign v2, Phase 3).

The same measures the 14 Oct checks used (scripts/strength_checks.py), so the
app's backtest page reads exactly like the study."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

JUMP_LOW, JUMP_HIGH = 0.55, 1.8


def equal_weight_curve(close_ff: np.ndarray, listed: np.ndarray, first: int) -> np.ndarray:
    """All screened stocks in equal amounts, rebalanced daily, no costs (a
    generous yardstick). A day-to-day ratio outside 0.55-1.8 (an unadjusted
    corporate action) counts as no change."""
    T = close_ff.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close_ff[1:] / close_ff[:-1]
    ok = listed[:-1] & np.isfinite(ratio) & (ratio > JUMP_LOW) & (ratio < JUMP_HIGH)
    daily = np.where(ok, ratio - 1, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
    curve = np.full(T, np.nan)
    curve[first] = 1.0
    curve[first + 1:] = np.cumprod(1 + daily[first:])
    return curve


def cagr_pct(values: np.ndarray, days: np.ndarray, mask: np.ndarray) -> float:
    v, d = values[mask], days[mask]
    keep = np.isfinite(v)
    v, d = v[keep], d[keep]
    if len(v) < 2 or v[0] <= 0:
        return float("nan")
    return float(((v[-1] / v[0]) ** (365.25 / (d[-1] - d[0])) - 1) * 100)


def worst_fall_pct(values: np.ndarray, mask: np.ndarray) -> float:
    v = values[mask]
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    return float((v / np.maximum.accumulate(v) - 1).min() * 100)


def monthly_t(values: np.ndarray, other: np.ndarray, days: np.ndarray, mask: np.ndarray) -> float:
    """t-value of the monthly return over ``other`` (2 or more: unlikely to be luck)."""
    m = mask & np.isfinite(values) & np.isfinite(other)
    month = [date.fromordinal(int(x)).strftime("%Y-%m") for x in days[m]]
    a = pd.Series(values[m]).groupby(month).last().pct_change()
    b = pd.Series(other[m]).groupby(month).last().pct_change()
    ex = (a - b).dropna()
    if len(ex) < 3 or ex.std(ddof=1) == 0:
        return float("nan")
    return float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))))


def yearly(values: np.ndarray, bench: np.ndarray, days: np.ndarray) -> list:
    """[(year, portfolio %, benchmark %)] from each year's last value."""
    years = np.array([date.fromordinal(int(d)).year for d in days])
    out, prev = [], None
    for y in sorted(set(years)):
        idx = np.flatnonzero((years == y) & np.isfinite(values))
        if not len(idx):
            continue
        start = prev if prev is not None else idx[0]
        out.append((int(y), float((values[idx[-1]] / values[start] - 1) * 100),
                    float((bench[idx[-1]] / bench[start] - 1) * 100)))
        prev = idx[-1]
    return out
