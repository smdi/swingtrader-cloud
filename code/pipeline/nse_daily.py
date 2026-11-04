"""NSE's official daily files into the v2 database (Redesign v2, Phase 2).

NSE is the main source; Kite is the backup (pipeline/kite_source.py). A day
written from Kite is replaced as soon as NSE's file for it is read.
"""
from __future__ import annotations

import io
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger("trader.pipeline")

SERIES = ("EQ", "BE", "BZ")           # BE/BZ: trade-to-trade; a stock moved there keeps its real price
PRICE_KINDS = ("bonus", "split", "consolidation", "demerger", "rights", "scheme", "capital_reduction")
INDEX_KEY = "NIFTY 50"
INDEX_FILE = "ind_close_all_{:%d%m%Y}.csv"


# ---------------------------------------------------------------- client
def offline_client(cache_dir):
    """An NSE client that only reads the cache folder (no request is sent)."""
    from pipeline.nse_eod import NseEodClient, NseFileNotPublished, NseFileUnavailable

    class Offline(NseEodClient):
        def _download_text(self, url, *args, **kwargs):
            raise NseFileNotPublished(f"offline: {url} not in the cache")

        def _request_json(self, url, *args, **kwargs):
            raise NseFileUnavailable(f"offline: {url} not in the cache")

        def equity_list(self, as_of, *, max_age_days: int = 30):
            from pipeline.nse_eod import parse_equity_list
            cached = self._latest_cached("ref", "EQUITY_L_", ".csv", as_of, max_age_days)
            if cached is None:
                raise NseFileUnavailable("offline: no cached EQUITY_L")
            return parse_equity_list(cached[0].read_text(encoding="utf-8"), min_rows=self.min_equity_rows), cached[1]

    return Offline(cache_dir, delay_s=0)


# ---------------------------------------------------------------- stock prices
def read_bhav(client, day: date):
    """NSE's daily file for ``day`` (downloaded once, then read from the cache),
    rows of series EQ, BE and BZ. Raises the client's errors when unavailable."""
    from pipeline.nse_eod import parse_bhavcopy
    client.bhavcopy(day)                       # downloads + validates + caches (EQ check)
    text = client.bhav_path(day).read_text(encoding="utf-8", errors="replace")
    return parse_bhavcopy(text, series=SERIES, min_rows=100)


def write_prices(conn, frame: pd.DataFrame, *, source: str = "nse") -> int:
    """Upsert one day's rows. An NSE row replaces anything; a Kite row never
    replaces an NSE row."""
    if frame.empty:
        return 0
    frame = frame.copy()
    frame["eq"] = frame["series"].eq("EQ")
    frame = frame.sort_values("eq", ascending=False).drop_duplicates(["symbol", "date"])
    rows = [(r.symbol, _iso(r.date), _f(r.open), _f(r.high), _f(r.low), float(r.close),
             _f(getattr(r, "prev_close", None)), _f(r.volume), _f(getattr(r, "turnover", None)),
             getattr(r, "series", None), source)
            for r in frame.itertuples(index=False)]
    guard = "" if source == "nse" else " WHERE prices.source != 'nse'"
    conn.executemany(
        "INSERT INTO prices (symbol, date, open, high, low, close, prev_close, volume, turnover, series, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(symbol, date) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, "
        "close=excluded.close, prev_close=excluded.prev_close, volume=excluded.volume, "
        "turnover=excluded.turnover, series=excluded.series, source=excluded.source" + guard, rows)
    days = sorted({row[1] for row in rows})
    conn.executemany(
        "INSERT INTO instruments (symbol, first_seen, last_seen) VALUES (?, ?, ?) "
        "ON CONFLICT(symbol) DO UPDATE SET first_seen = min(coalesce(first_seen, excluded.first_seen), "
        "excluded.first_seen), last_seen = max(coalesce(last_seen, excluded.last_seen), excluded.last_seen)",
        [(s, days[0], days[-1]) for s in frame["symbol"].unique()])
    return len(rows)


def write_equity_list(conn, equity: dict) -> int:
    """Names and ISINs from NSE's EQUITY_L; a symbol never in it while still
    trading is not an ordinary share (an ETF, bond, ...)."""
    rows = [(s, (v or {}).get("name"), (v or {}).get("isin")) for s, v in equity.items()]
    conn.executemany(
        "INSERT INTO instruments (symbol, name, isin, kind) VALUES (?, ?, ?, 'equity') "
        "ON CONFLICT(symbol) DO UPDATE SET name = excluded.name, isin = excluded.isin, kind = 'equity'", rows)
    return len(rows)


def mark_non_equities(conn, equity_symbols: set, last_day: str) -> int:
    """Instruments still trading on ``last_day`` but absent from EQUITY_L are
    not ordinary shares. Delisted stocks (not trading any more) stay equities."""
    still = {r[0] for r in conn.execute("SELECT symbol FROM instruments WHERE last_seen = ?", (last_day,))}
    others = sorted(still - set(equity_symbols))
    conn.executemany("UPDATE instruments SET kind = 'other' WHERE symbol = ?", [(s,) for s in others])
    return len(others)


# ---------------------------------------------------------------- corporate actions
def write_actions(conn, events) -> int:
    rows = [(e.symbol, _iso(e.ex_date), e.kind, e.factor, e.subject)
            for e in events if e.kind in PRICE_KINDS]
    conn.executemany(
        "INSERT INTO corporate_actions (symbol, ex_date, kind, factor, subject) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(symbol, ex_date, kind) DO UPDATE SET factor = excluded.factor, subject = excluded.subject",
        rows)
    return len(rows)


def cached_actions(cache_dir) -> list:
    """Every corporate action in the cached NSE action files (no download)."""
    from pipeline.nse_eod import parse_corporate_actions
    events, seen = [], set()
    for path in sorted((Path(cache_dir) / "actions").glob("actions_*.json")):
        try:
            parsed, _ = parse_corporate_actions(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:          # one bad file must not stop the rest
            logger.warning("Skipping corporate-action file %s: %s", path.name, exc)
            continue
        for e in parsed:
            key = (e.symbol, e.ex_date, e.kind, e.factor)
            if key not in seen:
                seen.add(key)
                events.append(e)
    return events


# ---------------------------------------------------------------- NIFTY 50
def parse_index_close(text: str, day: date, index_key: str = INDEX_KEY) -> dict:
    """One index's row from NSE's ``ind_close_all_DDMMYYYY.csv``."""
    from pipeline.nse_eod import NseSchemaError, parse_nse_date
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [str(c).strip() for c in frame.columns]
    need = ["Index Name", "Index Date", "Open Index Value", "High Index Value",
            "Low Index Value", "Closing Index Value"]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise NseSchemaError(f"index file lacks {missing}")
    names = frame["Index Name"].astype(str).str.strip().str.upper()
    row = frame[names.str.replace(" ", "") == index_key.upper().replace(" ", "")]
    if row.empty:
        raise NseSchemaError(f"index file has no {index_key}")
    row = row.iloc[0]
    when = parse_nse_date(str(row["Index Date"]).strip())
    if when != day:
        raise NseSchemaError(f"index file for {day} is dated {when}")
    value = lambda c: float(str(row[c]).replace(",", "")) if str(row[c]).strip() not in ("", "-") else None
    close = value("Closing Index Value")
    if not close or close <= 0:
        raise NseSchemaError(f"index file has no close for {index_key}")
    return {"open": value("Open Index Value"), "high": value("High Index Value"),
            "low": value("Low Index Value"), "close": close}


def read_index_close(client, day: date, index_key: str = INDEX_KEY) -> dict:
    from pipeline.nse_eod import ARCHIVE_URL
    path = Path(client.cache_dir) / "indices" / INDEX_FILE.format(day)
    if path.exists():
        return parse_index_close(path.read_text(encoding="utf-8", errors="replace"), day, index_key)
    text = client._download_text(f"{ARCHIVE_URL}/content/indices/{INDEX_FILE.format(day)}")
    parsed = parse_index_close(text, day, index_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    client._write_atomic(path, text)
    return parsed


def write_index(conn, day, values: dict, *, index_key: str = INDEX_KEY, source: str = "nse") -> None:
    guard = "" if source == "nse" else " WHERE index_prices.source != 'nse'"
    conn.execute(
        "INSERT INTO index_prices (index_key, date, open, high, low, close, source) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(index_key, date) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, "
        "close=excluded.close, source=excluded.source" + guard,
        (index_key, _iso(day), values.get("open"), values.get("high"), values.get("low"),
         float(values["close"]), source))


# ---------------------------------------------------------------- helpers
def _iso(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _f(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None      # NaN -> None
