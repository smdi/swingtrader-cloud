"""The database as published text files, and back (cloud/__init__.py).

Each table is one file ``data/db/<table>.jz``: JSON {"columns", "rows"}, gzipped
(mtime 0, so the same rows give the same bytes) and base64-encoded - published
files must be text. The two big tables (prices, screen) get a file per month
(``data/db/prices-2026-11.jz``), so an evening changes only the current month's.
``data/manifest.json`` holds each file's sha256: an export writes only files whose
content changed and lists the ones retention removed.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

MONTHLY = ("prices", "screen")                    # a file per calendar month (their `date` column)
SKIP = ("legacy_prices", "legacy_splice", "sqlite_sequence")   # history before Aug 2019: never needed here
EXTRA_FILES = {                                   # files outside the database, under the app's data/ folder
    "data/db/nse_holidays.json": "data/nse_holidays.json",
}
DELIVERY_CACHE = "data/nse_research/delivery_cache_experiments.npz"
DELIVERY_FILE = "data/db/delivery_cache.npz.b64"

# ------------------------------------------------------------------ retention (what the rules need, no more)
RANK_DAYS = 430              # the NSE-style score: a year and more of prices (strategy.live.SCORE_WARMUP_DAYS 420)
EXPERIMENT_WARMUP = 810      # experiments replay from their start with 800 days before it (strategy.experiments)
DELIVERY_DAYS_BEFORE = 150   # the delivery experiment's 60-session average needs ~90 sessions before its start
SCREEN_DAYS = 40             # the screen: from the experiments' start (they replay over it), else 40 days
ACTIONS_DAYS = 400           # corporate actions: the stock page shows the last year
JOB_LOG_DAYS = 90


def _encode(obj) -> tuple:
    """(file text, fingerprint of the content). The fingerprint is taken before compression:
    two zlib versions compress the same rows to different bytes (4 Nov 2026)."""
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return (base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii"),
            hashlib.sha256(raw).hexdigest())


def _plain_file(text: str) -> tuple:
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode(text: str):
    return json.loads(gzip.decompress(base64.b64decode(text.strip())).decode("utf-8"))


def _clean(v):
    if isinstance(v, float) and v != v:            # NaN is not JSON: stored as NULL (as SQLite reads it back)
        return None
    if isinstance(v, bytes):
        raise TypeError("binary columns are not exported")
    return v


def tables(conn) -> list:
    return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            if r[0] not in SKIP]


def experiments_start(conn) -> date | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'experiments_started_on'").fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None


def keep_from(conn, today: date) -> dict:
    """The first date kept per kind of data."""
    start = experiments_start(conn)
    prices = today - timedelta(days=RANK_DAYS)
    screen = today - timedelta(days=SCREEN_DAYS)
    if start:
        prices = min(prices, start - timedelta(days=EXPERIMENT_WARMUP))
        screen = min(screen, start)
    book = conn.execute("SELECT min(started_on) FROM paper_book").fetchone()[0]
    if book:                                       # the Today chart and the book's NIFTY line from its start
        prices = min(prices, date.fromisoformat(book) - timedelta(days=RANK_DAYS))
    return {"prices": prices, "screen": screen, "index_prices": prices,
            "corporate_actions": min(today - timedelta(days=ACTIONS_DAYS), prices),
            "job_log": today - timedelta(days=JOB_LOG_DAYS),
            "delivery": (start - timedelta(days=DELIVERY_DAYS_BEFORE)) if start else today - timedelta(days=200)}


def apply_retention(conn, today: date) -> dict:
    """Deletes what the rules no longer need. Commits. Returns rows deleted per table."""
    k = keep_from(conn, today)
    out = {}
    for table, col in (("prices", "date"), ("screen", "date"), ("index_prices", "date"),
                       ("corporate_actions", "ex_date")):
        out[table] = conn.execute(f"DELETE FROM {table} WHERE {col} < ?", (k[table].isoformat(),)).rowcount
    out["job_log"] = conn.execute("DELETE FROM job_log WHERE started_at < ?", (k["job_log"].isoformat(),)).rowcount
    for t in ("legacy_prices", "legacy_splice"):
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (t,)).fetchone():
            out[t] = conn.execute(f"DELETE FROM {t}").rowcount
    conn.commit()
    return out


def trim_delivery_cache(path: Path, first: date) -> None:
    """Keeps the delivery cache's days from ``first`` (strategy.signals.delivery_matrix's format)."""
    import numpy as np
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    days = [str(d) for d in data["days"]]
    keep = [i for i, d in enumerate(days) if d >= first.isoformat()]
    if len(keep) == len(days):
        return
    data["days"] = data["days"][keep]
    data["deliv"] = data["deliv"][keep]
    tmp = path.with_name(path.stem + ".trim.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)


# ------------------------------------------------------------------ database -> files
def dump(conn, base: Path, app_dir: Path) -> dict:
    """Every file's (text, fingerprint), keyed by its published path (data/db/...)."""
    out = {}
    for t in tables(conn):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{t}')")]
        order = ", ".join(str(i + 1) for i in range(len(cols)))
        if t in MONTHLY:
            months = [r[0] for r in conn.execute(f"SELECT DISTINCT substr(date, 1, 7) FROM {t} ORDER BY 1")]
            for m in months:
                rows = [[_clean(v) for v in r] for r in conn.execute(
                    f"SELECT * FROM {t} WHERE date >= ? AND date < ? ORDER BY {order}", (f"{m}-01", f"{m}-32"))]
                out[f"data/db/{t}-{m}.jz"] = _encode({"table": t, "columns": cols, "rows": rows})
        else:
            rows = [[_clean(v) for v in r] for r in conn.execute(f"SELECT * FROM {t} ORDER BY {order}")]
            out[f"data/db/{t}.jz"] = _encode({"table": t, "columns": cols, "rows": rows})
    for published, local in EXTRA_FILES.items():
        p = app_dir / local
        if p.exists():
            out[published] = _plain_file(p.read_text(encoding="utf-8"))
    cache = app_dir / DELIVERY_CACHE
    if cache.exists():
        out[DELIVERY_FILE] = _delivery_file(cache)
    return out


def _delivery_file(path: Path) -> tuple:
    """The delivery cache as base64 text; its fingerprint is of the arrays inside (the .npz
    zip carries timestamps, so the same arrays give different bytes)."""
    import numpy as np
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(z.files):
            a = z[k]
            h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
            h.update(a.tobytes() if a.dtype.kind != "U" else "\n".join(a.tolist()).encode())
    return base64.b64encode(path.read_bytes()).decode("ascii"), h.hexdigest()


def export(conn, outdir: Path, app_dir: Path, previous: dict) -> dict:
    """Writes the data files that differ from ``previous`` (the last manifest) into
    ``outdir`` at their published paths, and the new manifest. Returns
    {"changed": [paths], "removed": [paths], "manifest": {...}}."""
    files = dump(conn, outdir, app_dir)
    manifest = {p: h for p, (_, h) in files.items()}
    changed = sorted(p for p, h in manifest.items() if previous.get(p) != h)
    removed = sorted(p for p in previous if p not in manifest)
    for p in changed:
        f = outdir / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(files[p][0], encoding="utf-8")
    m = outdir / "data/manifest.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(manifest, indent=0, sort_keys=True), encoding="utf-8")
    return {"changed": changed, "removed": removed, "manifest": manifest}


# ------------------------------------------------------------------ files -> database
def restore(datadir: Path, db_path: Path, app_dir: Path) -> dict:
    """Builds the database at ``db_path`` (new) from the files under ``datadir``
    (the folder holding data/db/...). Returns rows per table and the manifest read."""
    from store import db as store
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    conn = store.connect(db_path)                  # the app's own schema
    conn.execute("PRAGMA foreign_keys = OFF")      # tables load in name order; links are checked at the end
    counts = {}
    for f in sorted((Path(datadir) / "data/db").glob("*.jz")):
        payload = _decode(f.read_text(encoding="ascii"))
        t, cols, rows = payload["table"], payload["columns"], payload["rows"]
        have = [r[1] for r in conn.execute(f"PRAGMA table_info('{t}')")]
        if not have:
            raise RuntimeError(f"{f.name}: table {t} is not in the app's schema")
        idx = [cols.index(c) for c in have if c in cols]
        names = [c for c in have if c in cols]
        conn.executemany(f"INSERT OR REPLACE INTO {t} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                         [[r[i] for i in idx] for r in rows])
        counts[t] = counts.get(t, 0) + len(rows)
    conn.commit()
    broken = conn.execute("PRAGMA foreign_key_check").fetchall()
    if broken:
        raise RuntimeError(f"the data files do not fit together: {broken[:5]}")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.close()
    for published, local in EXTRA_FILES.items():
        src = Path(datadir) / published
        if src.exists():
            dst = app_dir / local
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    src = Path(datadir) / DELIVERY_FILE
    if src.exists():
        dst = app_dir / DELIVERY_CACHE
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(src.read_text(encoding="ascii").strip()))
    mf = Path(datadir) / "data/manifest.json"
    manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    return {"rows": counts, "manifest": manifest}
