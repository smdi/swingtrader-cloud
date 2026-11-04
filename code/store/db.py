"""v2 database: a slim SQLite file (data/v2/trader.db) - Redesign v2, Phase 2.

Prices are kept exactly as NSE publishes them (raw); split/bonus adjustment
is computed from NSE's corporate actions when prices are read, so the stored
data never needs rewriting. History before NSE's daily files (2008 - Aug 2019)
comes from the old database, which held prices already adjusted for splits:
it lives in its own table and is lined up with NSE's at the join.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    symbol      TEXT PRIMARY KEY,
    name        TEXT,
    isin        TEXT,
    kind        TEXT NOT NULL DEFAULT 'equity',   -- equity | other (ETF, bond, ...)
    first_seen  TEXT,
    last_seen   TEXT
);
-- NSE's official daily prices, unchanged (series EQ, BE, BZ).
CREATE TABLE IF NOT EXISTS prices (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL, high REAL, low REAL, close REAL NOT NULL,
    prev_close  REAL,
    volume      REAL,
    turnover    REAL,                              -- rupees
    series      TEXT,
    source      TEXT NOT NULL DEFAULT 'nse',      -- nse | kite (backup, replaced when NSE's file arrives)
    PRIMARY KEY (symbol, date)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_prices_date ON prices(date);
-- Before NSE's files: the old database's split-adjusted history.
CREATE TABLE IF NOT EXISTS legacy_prices (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL, high REAL, low REAL, close REAL NOT NULL,
    volume      REAL,
    source      TEXT,
    PRIMARY KEY (symbol, date)
) WITHOUT ROWID;
-- Per stock: NSE raw close / old adjusted close on the first NSE sessions.
CREATE TABLE IF NOT EXISTS legacy_splice (
    symbol      TEXT PRIMARY KEY,
    factor      REAL NOT NULL,
    days        INTEGER
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol      TEXT NOT NULL,
    ex_date     TEXT NOT NULL,
    kind        TEXT NOT NULL,                     -- split | bonus | consolidation | other
    factor      REAL,                              -- price before / price after
    subject     TEXT,
    PRIMARY KEY (symbol, ex_date, kind)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS index_prices (
    index_key   TEXT NOT NULL,                     -- 'NIFTY 50'
    date        TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL NOT NULL,
    source      TEXT,
    PRIMARY KEY (index_key, date)
) WITHOUT ROWID;
-- The stocks considered each day (the screen).
CREATE TABLE IF NOT EXISTS screen (
    date        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    source      TEXT NOT NULL,                     -- legacy (old app's daily list) | live (today's screen rules)
    PRIMARY KEY (date, symbol)
) WITHOUT ROWID;
-- Strength backtests (Phase 3).
CREATE TABLE IF NOT EXISTS bt_runs (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,
    rules       TEXT NOT NULL,                     -- JSON of strategy.strength.Rules
    start_date  TEXT NOT NULL,
    end_date    TEXT NOT NULL,
    summary     TEXT                               -- JSON: per period return, vs NIFTY, vs all screened, t, worst fall
);
CREATE TABLE IF NOT EXISTS bt_values (
    run_id      INTEGER NOT NULL REFERENCES bt_runs(id) ON DELETE CASCADE,
    date        TEXT NOT NULL,
    value       REAL, nifty REAL, all_screened REAL,
    PRIMARY KEY (run_id, date)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS bt_fills (
    run_id      INTEGER NOT NULL REFERENCES bt_runs(id) ON DELETE CASCADE,
    date        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    shares      REAL, price REAL, value REAL, charges REAL
);
CREATE INDEX IF NOT EXISTS ix_bt_fills_run ON bt_fills(run_id, date);
-- The live paper portfolio (Phase 4).
CREATE TABLE IF NOT EXISTS paper_book (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    capital     REAL NOT NULL,
    started_on  TEXT NOT NULL,
    rules       TEXT NOT NULL                      -- JSON of strategy.strength.Rules
);
CREATE TABLE IF NOT EXISTS paper_positions (
    symbol      TEXT PRIMARY KEY,
    shares      REAL NOT NULL,
    cost        REAL NOT NULL,                     -- rupees paid, charges included
    opened_on   TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS paper_cash (
    id          INTEGER PRIMARY KEY,
    date        TEXT NOT NULL,
    kind        TEXT NOT NULL,                     -- deposit | buy | sell
    symbol      TEXT,
    amount      REAL NOT NULL,                     -- + in, - out
    balance     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_decisions (
    id          INTEGER PRIMARY KEY,
    decided_on  TEXT NOT NULL,
    keep        TEXT NOT NULL,                     -- JSON list: stocks that may stay (top keep_rank)
    buy_order   TEXT NOT NULL,                     -- JSON list: best first
    scores      TEXT,                              -- JSON {symbol: 6-month return}
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | executed
    executed_on TEXT
);
CREATE TABLE IF NOT EXISTS paper_fills (
    id          INTEGER PRIMARY KEY,
    decision_id INTEGER REFERENCES paper_decisions(id),
    date        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    shares      REAL NOT NULL,
    price       REAL NOT NULL,                     -- NSE's official open
    fill_price  REAL NOT NULL,                     -- after slippage
    value       REAL NOT NULL,                     -- paid (buy) or received (sell, net)
    charges     REAL NOT NULL,
    slippage    REAL NOT NULL,
    source      TEXT,                              -- NULL: NSE's open/close | kite_open (morning) | live (Sell now)
    checked     TEXT                               -- a kite_open fill against NSE's open: same | corrected ...
);
CREATE TABLE IF NOT EXISTS paper_actions (
    id          INTEGER PRIMARY KEY,
    date        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    kind        TEXT NOT NULL,                     -- split | bonus | consolidation | detected (no NSE record)
    factor      REAL NOT NULL,
    shares_before REAL NOT NULL,
    shares_after  REAL NOT NULL
);
-- "Sell now" clicks outside market hours: executed by the evening run at
-- the official open (at='open') or close (at='close') of ``day``.
CREATE TABLE IF NOT EXISTS paper_orders (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    day         TEXT NOT NULL,
    at          TEXT NOT NULL,                     -- open | close
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | done | cancelled | not held
    done_on     TEXT,
    reason      TEXT NOT NULL DEFAULT 'manual',    -- manual (Sell now) | stop | news
    signal_day  TEXT,                              -- the close (stop) or day (news) that triggered it
    detail      TEXT
);
CREATE TABLE IF NOT EXISTS paper_values (
    date        TEXT PRIMARY KEY,
    value       REAL NOT NULL,
    cash        REAL NOT NULL,
    holdings    INTEGER NOT NULL,
    nifty       REAL,
    all_screened REAL                              -- all screened stocks, equal amounts, from the start = 1.0
) WITHOUT ROWID;
-- Experiments (21 Oct 2026): pretend portfolios replayed from their start each
-- evening by strategy/experiments.py. Never read or written by the paper book.
CREATE TABLE IF NOT EXISTS exp_values (
    exp         TEXT NOT NULL,
    date        TEXT NOT NULL,
    value       REAL NOT NULL,
    cash        REAL NOT NULL,
    holdings    INTEGER NOT NULL,
    PRIMARY KEY (exp, date)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS exp_positions (
    exp         TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    shares      REAL NOT NULL,
    buy_price   REAL,                              -- after slippage, adjusted for splits since
    bought_on   TEXT,
    close       REAL,                              -- the last close
    PRIMARY KEY (exp, symbol)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS exp_fills (
    exp         TEXT NOT NULL,
    date        TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    shares      REAL, price REAL, value REAL, charges REAL,
    reason      TEXT                               -- decision | stop
);
CREATE INDEX IF NOT EXISTS ix_exp_fills ON exp_fills(exp, date);
CREATE TABLE IF NOT EXISTS exp_planned (           -- trades waiting for the next open
    exp         TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    decided_on  TEXT NOT NULL,
    reason      TEXT,
    PRIMARY KEY (exp, symbol, side)
) WITHOUT ROWID;
-- NSE's classification (from its quote page; the file industries.csv in the NSE folder).
CREATE TABLE IF NOT EXISTS sectors (
    symbol      TEXT PRIMARY KEY,
    sector      TEXT,
    industry    TEXT,
    updated     TEXT
) WITHOUT ROWID;
-- NSE's corporate announcements (pipeline/news.py); flag = the red flag raised, if any.
CREATE TABLE IF NOT EXISTS announcements (
    id          TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    at          TEXT NOT NULL,
    subject     TEXT,
    text        TEXT,
    link        TEXT,
    flag        TEXT,
    seen_at     TEXT,
    acted       TEXT                               -- 'order <id>' once sold; 'before purchase'
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_announcements_symbol ON announcements(symbol, at);
CREATE INDEX IF NOT EXISTS ix_announcements_flag ON announcements(flag, acted, at);
-- Google News headlines about the holdings, with a word-based label (pipeline/media_news.py).
CREATE TABLE IF NOT EXISTS media_news (
    id          TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    at          TEXT NOT NULL,
    title       TEXT NOT NULL,
    source      TEXT,
    link        TEXT,
    sentiment   TEXT NOT NULL,                     -- good | bad | neutral
    words       TEXT
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_media_news_symbol ON media_news(symbol, at);
CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT
);
CREATE TABLE IF NOT EXISTS job_log (
    id          INTEGER PRIMARY KEY,
    job         TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT,
    detail      TEXT
);
"""


def default_path() -> Path:
    from config import BASE_DIR
    return Path(BASE_DIR) / "data" / "v2" / "trader.db"


def connect(path=None, *, readonly: bool = False) -> sqlite3.Connection:
    path = Path(path or default_path())
    if readonly:
        conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=60)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
        _add_columns(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


ADDED_COLUMNS = (   # columns added after a table was first created (CREATE IF NOT EXISTS skips them)
    ("paper_orders", "reason", "TEXT NOT NULL DEFAULT 'manual'"),
    ("paper_orders", "signal_day", "TEXT"),
    ("paper_orders", "detail", "TEXT"),
    ("announcements", "acted", "TEXT"),
    ("paper_fills", "source", "TEXT"),
    ("paper_fills", "checked", "TEXT"),
)


def _add_columns(conn) -> None:
    for table, column, kind in ADDED_COLUMNS:
        if column not in [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    conn.commit()


def load_sectors(conn, path) -> int:
    """Replace the sector table from NSE's classification file (symbol, sector,
    industry, ...). Names are title-cased (NSE writes a few in capitals)."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(r["symbol"].strip(), (r.get("sector") or "").strip().title() or None,
                 (r.get("industry") or "").strip().title() or None, r.get("fetched_at"))
                for r in csv.DictReader(f) if r.get("symbol")]
    conn.execute("DELETE FROM sectors")
    conn.executemany("INSERT OR REPLACE INTO sectors VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


def sector_groups(conn, symbols, column: str = "sector"):
    """Integer group per symbol for strategy.strength's sector limit; -1 when
    the sector is not on file (such a stock is never limited)."""
    import numpy as np
    names = dict(conn.execute(f"SELECT symbol, {column} FROM sectors WHERE {column} IS NOT NULL"))
    index = {n: i for i, n in enumerate(sorted(set(names.values())))}
    return np.array([index.get(names.get(s), -1) for s in symbols], dtype=int)


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn, key, value) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))
