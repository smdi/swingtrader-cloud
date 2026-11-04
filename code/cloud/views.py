"""The page's data (data/view/*.json), made with the app's own queries (web/queries.py),
so every number is the one the PC app would show for the same database."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

STOCK_PAGES_TOP = 60          # stock pages made for the top of the ranking, plus every holding


def _plain(v):
    """JSON-safe: dates as ISO text, numpy numbers as Python ones, NaN and infinity as null."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_plain(x) for x in v]
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "_asdict"):
        return _plain(v._asdict())
    if hasattr(v, "__dataclass_fields__"):
        from dataclasses import asdict
        return _plain(asdict(v))
    return v


def _stock(conn, q, symbol: str, today: date, ranked: dict) -> dict | None:
    """web.queries.stock without re-ranking every stock for each page."""
    from pipeline import panel as pn
    if not conn.execute("SELECT 1 FROM prices WHERE symbol = ? LIMIT 1", (symbol,)).fetchone():
        return None
    p = pn.load(conn, today - timedelta(days=400), today, symbols=[symbol], light=True)
    if not len(p.symbols):
        return None
    keep = np.isfinite(p.close[:, 0])
    series = [{"date": d.isoformat(), "close": float(c), "nifty": float(n) if np.isfinite(n) else None}
              for d, c, n, k in zip(p.sessions, p.close[:, 0], p.index_close, keep) if k]
    info = conn.execute("SELECT name, isin FROM instruments WHERE symbol = ?", (symbol,)).fetchone()
    held = conn.execute("SELECT shares, cost, opened_on FROM paper_positions WHERE symbol = ?", (symbol,)).fetchone()
    actions = [dict(zip(("ex_date", "kind", "factor", "subject"), r)) for r in conn.execute(
        "SELECT ex_date, kind, factor, subject FROM corporate_actions WHERE symbol = ? "
        "ORDER BY ex_date DESC LIMIT 10", (symbol,))]
    return {"symbol": symbol, "name": info[0] if info else None,
            "media": q.media_news(conn, [symbol], days=14, today=today, per_stock=15)[0],
            "filings": q.announcements(conn, [symbol], days=30, today=today),
            "series": series, "rank": ranked.get(symbol), "fills": q.fills(conn, 50, symbol), "actions": actions,
            "held": dict(zip(("shares", "cost", "opened_on"), held)) if held else None}


def build(conn, today: date, ran_at: datetime, run_log: list) -> dict:
    """{file name: data} for data/view/."""
    from web import queries as q
    out = {}
    out["today.json"] = q.overview(conn, today)
    rank = q.ranking(conn)
    out["stocks.json"] = rank
    pos = q.positions(conn, today=today)
    held = [p["symbol"] for p in pos["positions"]]
    out["portfolio.json"] = {
        "d": pos, "alerts": q.risk_alerts(conn, pos), "orders": q.orders(conn), "fills": q.fills(conn),
        "rule_sales": q.rule_sales(conn), "news": q.announcements(conn, held, today=today),
        "media": q.media_news(conn, held, today=today), "media_at": q.media_news_time(conn), "book": q.book(conn)}
    out["experiments.json"] = q.experiments(conn, today)
    runs = q.runs(conn)
    out["research.json"] = {"runs": runs, "series": {str(r["id"]): q.run_values(conn, r["id"]) for r in runs},
                            "costs": _costs()}
    ranked = {r["symbol"]: r for r in rank.get("rows", [])}
    want = list(dict.fromkeys(held + [r["symbol"] for r in rank.get("rows", []) if r.get("rank")][:STOCK_PAGES_TOP]
                              + [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM exp_positions")]))
    out["stocks_detail.json"] = {s: v for s in want if (v := _stock(conn, q, s, today, ranked))}
    last_nse = conn.execute("SELECT max(date) FROM prices WHERE source = 'nse'").fetchone()[0]
    out["status.json"] = {"ran_at": ran_at.strftime("%Y-%m-%d %H:%M"), "today": today.isoformat(),
                          "prices_to": last_nse, "log": run_log[-40:],
                          "jobs": q.jobs(conn, 15),
                          "window_from": conn.execute("SELECT min(date) FROM prices").fetchone()[0]}
    return {k: _plain(v) for k, v in out.items()}


def _costs():
    from strategy import strength as st
    c = st.Costs.from_config()
    return {"slippage": c.slippage, "dp_per_sell": c.dp_per_sell}


def write(views: dict, outdir: Path) -> list:
    folder = Path(outdir) / "data/view"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in views.items():
        (folder / name).write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
                                   encoding="utf-8")
        paths.append(f"data/view/{name}")
    return paths
