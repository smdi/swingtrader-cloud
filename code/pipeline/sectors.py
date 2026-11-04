"""NSE sectors for stocks not on the list yet (16 Oct 2026).

The sector list (data\\nse_research\\industries.csv, loaded into the sectors
table) was fetched once on 6 Oct from EQ-series stocks only, so stocks in
NSE's BE/BZ series and stocks listed since had no sector - and the 4-per-sector
limit cannot count a stock without one. Each evening the update looks up the
held and screened stocks still missing (holdings first, at most ``limit``)
on NSE's quote page and adds them to the file and the table.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from pipeline.nse_eod import NseFileUnavailable, NseSchemaError

FIELDS = ["symbol", "macro", "sector", "industry", "basic_industry", "fetched_at"]
GIVE_UP_AFTER = 5            # NSE not answering this many times in a row: try again tomorrow


def missing(conn, day: date) -> list:
    """Held, then screened (latest screen on or before ``day``) stocks without a sector."""
    known = {r[0] for r in conn.execute("SELECT symbol FROM sectors WHERE sector IS NOT NULL")}
    held = [r[0] for r in conn.execute("SELECT symbol FROM paper_positions ORDER BY symbol")]
    screen_day = conn.execute("SELECT max(date) FROM screen WHERE date <= ?", (day.isoformat(),)).fetchone()[0]
    screened = [r[0] for r in conn.execute("SELECT symbol FROM screen WHERE date = ? ORDER BY symbol",
                                           (screen_day,))] if screen_day else []
    out = []
    for s in held + screened:
        if s not in known and s not in out:
            out.append(s)
    return out


def fill_missing(conn, client, file: Path, day: date, *, limit: int = 40, log=print) -> dict:
    """Look up missing sectors on NSE and add them to ``file`` and the table. Commits."""
    file = Path(file)
    todo = missing(conn, day)[:limit]
    added, failed, unavailable_run, found = [], 0, 0, []
    for symbol in todo:
        try:
            info = client.stock_industry(symbol)
        except NseFileUnavailable as exc:
            failed += 1
            unavailable_run += 1
            if unavailable_run >= GIVE_UP_AFTER:
                log(f"sectors: NSE not answering ({exc}); the rest tomorrow")
                break
            continue
        except NseSchemaError:
            failed += 1
            unavailable_run = 0
            continue
        unavailable_run = 0
        row = {"symbol": symbol, **info, "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        _append(file, row)
        found.append(row)
    for row in found:                  # written after the lookups: no write lock held across NSE's pacing
        conn.execute("INSERT OR REPLACE INTO sectors VALUES (?, ?, ?, ?)",
                     (row["symbol"], (row.get("sector") or "").strip().title() or None,
                      (row.get("industry") or "").strip().title() or None, row["fetched_at"]))
        added.append(row["symbol"])
    if added:
        from store import db as store
        store.set_meta(conn, "sectors_file", str(int(file.stat().st_mtime)))   # the app need not reload it
    conn.commit()
    if added or failed:
        log(f"sectors: added {len(added)} ({', '.join(added[:10])}), not found {failed}, "
            f"still missing {len(todo) - len(added)}")
    return {"added": len(added), "failed": failed, "symbols": added}


def _append(file: Path, row: dict) -> None:
    new = not file.exists() or file.stat().st_size == 0
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
