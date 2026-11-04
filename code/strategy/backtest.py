"""Strength backtest on the v2 database (Redesign v2, Phase 3).

Runs strategy.strength.simulate (the same code the live paper portfolio uses)
over stored history and measures it against NIFTY 50 and all screened stocks
in equal amounts, per period. Results are saved in bt_runs / bt_values /
bt_fills for the Research page.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np

from pipeline import panel as pn
from strategy import metrics, strength as st

WARMUP_DAYS = 420                       # calendar days before the start: 12-1m look-back
DEFAULT_PERIODS = ((date(2010, 1, 1), date(2018, 12, 31)), (date(2019, 1, 1), date(2100, 1, 1)))


def screened_symbols(conn, start: date, end: date) -> list:
    return [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM screen WHERE date BETWEEN ? AND ?",
                                       (start.isoformat(), end.isoformat()))]


def run(conn, rules: st.Rules, start: date, end: date, *, capital: float = 1_000_000.0,
        costs: st.Costs | None = None, periods=DEFAULT_PERIODS, save: bool = True, log=print, panel=None) -> dict:
    """``panel``: an already loaded panel (several runs on the same data)."""
    t0 = time.time()
    costs = costs or st.Costs.from_config()
    if panel is None:
        symbols = screened_symbols(conn, start, end)
        panel = pn.load(conn, start - timedelta(days=WARMUP_DAYS), end, symbols=symbols, light=True)
    p = panel
    days = p.days
    first = int(np.searchsorted(days, start.toordinal()))
    log(f"  {len(p.sessions)} sessions x {len(p.symbols)} stocks loaded ({time.time() - t0:.0f}s)")
    groups = None
    if rules.sector_cap:
        from store import db as store
        groups = store.sector_groups(conn, p.symbols)
    result = st.simulate(p.sessions, p.open, p.close, p.close_ff, p.listed, rules=rules, costs=costs,
                         capital=capital, start_index=first, groups=groups)
    values = result.values
    ew = metrics.equal_weight_curve(p.close_ff, p.listed, first)
    nifty = p.index_close
    summary = {"periods": [], "costs_per_year_pct": None, "turnover_per_year_pct": None,
               "corporate_actions": result.corporate_actions, "rebalances": len(result.rebalances),
               "fills": len(result.fills)}
    years = (days[-1] - days[first]) / 365.25
    avg = float(np.nanmean(values))
    summary["costs_per_year_pct"] = result.costs / years / avg * 100
    summary["turnover_per_year_pct"] = result.turnover / years / avg * 100
    for a, b in periods:
        mask = (days >= max(a, start).toordinal()) & (days <= min(b, end).toordinal())
        if mask.sum() < 60:
            continue
        summary["periods"].append({
            "from": p.sessions[int(np.argmax(mask))].isoformat(),
            "to": p.sessions[len(mask) - 1 - int(np.argmax(mask[::-1]))].isoformat(),
            "return_pct": metrics.cagr_pct(values, days, mask),
            "nifty_pct": metrics.cagr_pct(nifty, days, mask),
            "all_screened_pct": metrics.cagr_pct(ew, days, mask),
            "t_vs_nifty": metrics.monthly_t(values, nifty, days, mask),
            "t_vs_all_screened": metrics.monthly_t(values, ew, days, mask),
            "worst_fall_pct": metrics.worst_fall_pct(values, mask),
            "nifty_worst_fall_pct": metrics.worst_fall_pct(nifty, mask),
        })
    span = days >= days[first]
    summary["years"] = metrics.yearly(values[span], nifty[span], days[span])
    summary["seconds"] = round(time.time() - t0)
    if save:
        summary["run_id"] = _save(conn, rules, start, end, p, first, values, nifty, ew, result, summary)
    return summary


def _save(conn, rules, start, end, p, first, values, nifty, ew, result, summary) -> int:
    cur = conn.execute("INSERT INTO bt_runs (created_at, rules, start_date, end_date, summary) VALUES (?, ?, ?, ?, ?)",
                       (time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(asdict(rules)), start.isoformat(),
                        end.isoformat(), json.dumps(summary)))
    run_id = cur.lastrowid
    conn.executemany("INSERT INTO bt_values (run_id, date, value, nifty, all_screened) VALUES (?, ?, ?, ?, ?)",
                     [(run_id, p.sessions[t].isoformat(), _n(values[t]), _n(nifty[t]), _n(ew[t]))
                      for t in range(first, len(p.sessions))])
    conn.executemany("INSERT INTO bt_fills (run_id, date, symbol, side, shares, price, value, charges) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     [(run_id, p.sessions[t].isoformat(), p.symbols[f.stock], f.side, f.shares, f.price,
                       f.value, f.charges) for t, f in result.fills])
    conn.commit()
    return run_id


def _n(x):
    return None if x is None or not np.isfinite(x) else float(x)
