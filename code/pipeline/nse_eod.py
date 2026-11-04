"""NSE's official end-of-day files: download once, validate, keep on disk.

* ``sec_bhavdata_full_DDMMYYYY.csv``: the daily prices (bhavcopy)
* ``EQUITY_L.csv``: listed equities
* ``/api/corporates-corporateActions``: splits, bonuses and other actions

Every parser checks its columns and fails loudly (``NseSchemaError``) rather
than guessing. A file NSE has not published yet raises ``NseFileNotPublished``.
Files are written to the cache only after they parse.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from core.logging_config import get_logger
ADJUSTABLE_KINDS = frozenset({"split", "bonus", "consolidation"})
UNADJUSTABLE_PRICE_KINDS = frozenset({"demerger", "rights", "scheme", "capital_reduction"})


@dataclass(frozen=True)
class CorporateActionEvent:
    symbol: str
    ex_date: date
    kind: str
    subject: str
    factor: Optional[float] = None       # price before / price after (adjustable kinds)

logger = get_logger("nse_eod")

ARCHIVE_URL = "https://nsearchives.nseindia.com"
API_URL = "https://www.nseindia.com/api"
EQUITY_LIST_URL = f"{ARCHIVE_URL}/content/equities/EQUITY_L.csv"
ACTIONS_URL = f"{API_URL}/corporates-corporateActions"
HOLIDAYS_URL = f"{API_URL}/holiday-master"


def bhavcopy_url(session: date) -> str:
    return f"{ARCHIVE_URL}/products/content/sec_bhavdata_full_{session:%d%m%Y}.csv"


BHAV_COLUMNS = (
    "SYMBOL", "SERIES", "DATE1", "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE",
    "LOW_PRICE", "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY",
    "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER",
)
EQUITY_LIST_REQUIRED = ("SYMBOL", "NAME OF COMPANY", "SERIES", "DATE OF LISTING")
BAND_REQUIRED = ("SYMBOL", "SERIES", "BAND")
NO_BAND_VALUES = frozenset({"NO BAND", "NOBAND", "NO-BAND"})
_MISSING = frozenset({"", "-", "NA", "N/A", "NAN", "NONE", "NULL"})
_DATE_FORMATS = ("%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y")


class NseFileUnavailable(RuntimeError):
    """NSE has not published the file (yet), or NSE could not be reached."""


class NseFileNotPublished(NseFileUnavailable):
    """NSE answered that the file does not exist (404 or its "not found" page).

    Only this answer can mark a past calendar session as an unlisted holiday;
    timeouts, 403 and 5xx are outages and never do.
    """


class NseSchemaError(ValueError):
    """A file was received but does not have the expected shape."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def parse_nse_date(value) -> Optional[date]:
    """Parse the date spellings NSE uses (``06-Sep-2024``, ``06-SEP-2024``)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text.upper() in _MISSING:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _read_csv_text(text: str) -> pd.DataFrame:
    if text is None or not str(text).strip():
        raise NseSchemaError("empty file")
    if str(text).lstrip().lower().startswith(("<!doctype", "<html")):
        raise NseSchemaError("received an HTML page instead of a CSV file")
    try:
        frame = pd.read_csv(io.StringIO(str(text)), dtype=str, keep_default_na=False,
                            skipinitialspace=True)
    except Exception as exc:  # pandas raises several parser errors
        raise NseSchemaError(f"unreadable CSV: {exc}") from exc
    frame.columns = [str(col).strip().upper() for col in frame.columns]
    for col in frame.columns:
        frame[col] = frame[col].astype(str).str.strip()
    return frame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise NseSchemaError(f"{label}: missing columns {missing}; got {list(frame.columns)}")


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.where(~cleaned.str.upper().isin(_MISSING), None)
    return pd.to_numeric(cleaned, errors="coerce")


# ---------------------------------------------------------------------------
# Parsers (pure: text in, data out)
# ---------------------------------------------------------------------------

def parse_bhavcopy(text: str, *, series: Optional[Iterable[str]] = ("EQ",),
                   min_rows: int = 500) -> pd.DataFrame:
    """Parse ``sec_bhavdata_full``.

    Returns one row per symbol with columns ``symbol, series, date,
    prev_close, open, high, low, close, volume, turnover, trades, deliv_qty,
    deliv_pct``. ``turnover`` is in rupees (the file is in lakhs).
    ``frame.attrs`` carries ``session_date``, ``total_rows`` and
    ``dropped_rows`` (rows with unusable prices).
    """
    frame = _read_csv_text(text)
    _require_columns(frame, BHAV_COLUMNS, "bhavcopy")
    if frame.empty:
        raise NseSchemaError("bhavcopy has no rows")
    if len(frame) < min_rows:
        raise NseSchemaError(f"bhavcopy has only {len(frame)} rows (expected at least {min_rows})")

    dates = {parse_nse_date(value) for value in frame["DATE1"].unique()}
    if None in dates:
        raise NseSchemaError("bhavcopy has an unreadable DATE1 value")
    if len(dates) != 1:
        raise NseSchemaError(f"bhavcopy mixes several dates: {sorted(dates)}")
    session_date = dates.pop()

    total_rows = len(frame)
    frame["SYMBOL"] = frame["SYMBOL"].str.upper()
    frame["SERIES"] = frame["SERIES"].str.upper()
    if series is not None:
        wanted = {str(item).upper() for item in series}
        frame = frame[frame["SERIES"].isin(wanted)]
    if frame.duplicated(["SYMBOL", "SERIES"]).any():
        dupes = sorted(frame.loc[frame.duplicated(["SYMBOL", "SERIES"]), "SYMBOL"].unique())[:5]
        raise NseSchemaError(f"bhavcopy repeats symbol/series rows, e.g. {dupes}")

    out = pd.DataFrame({
        "symbol": frame["SYMBOL"].to_numpy(),
        "series": frame["SERIES"].to_numpy(),
        "date": session_date,
        "prev_close": _numeric(frame["PREV_CLOSE"]).to_numpy(),
        "open": _numeric(frame["OPEN_PRICE"]).to_numpy(),
        "high": _numeric(frame["HIGH_PRICE"]).to_numpy(),
        "low": _numeric(frame["LOW_PRICE"]).to_numpy(),
        "close": _numeric(frame["CLOSE_PRICE"]).to_numpy(),
        "volume": _numeric(frame["TTL_TRD_QNTY"]).to_numpy(),
        "turnover": (_numeric(frame["TURNOVER_LACS"]) * 1e5).round(2).to_numpy(),
        "trades": _numeric(frame["NO_OF_TRADES"]).to_numpy(),
        "deliv_qty": _numeric(frame["DELIV_QTY"]).to_numpy(),
        "deliv_pct": _numeric(frame["DELIV_PER"]).to_numpy(),
    })
    prices = out[["open", "high", "low", "close"]]
    tolerance = 1e-6
    valid = (
        prices.notna().all(axis=1)
        & (prices > 0).all(axis=1)
        & out["volume"].notna() & (out["volume"] >= 0)
        & out["turnover"].notna() & (out["turnover"] >= 0)
        & (out["high"] + tolerance >= out["low"])
        & (out["high"] + tolerance >= out[["open", "close"]].max(axis=1))
        & (out["low"] - tolerance <= out[["open", "close"]].min(axis=1))
    )
    dropped = int((~valid).sum())
    out = out[valid].reset_index(drop=True)
    out.attrs.update({"session_date": session_date, "total_rows": total_rows,
                      "dropped_rows": dropped})
    return out


def parse_equity_list(text: str, *, min_rows: int = 1000) -> dict:
    """Parse ``EQUITY_L.csv`` into ``symbol -> {name, series, listing_date, isin, face_value}``."""
    frame = _read_csv_text(text)
    _require_columns(frame, EQUITY_LIST_REQUIRED, "EQUITY_L")
    if len(frame) < min_rows:
        raise NseSchemaError(f"EQUITY_L has only {len(frame)} rows (expected at least {min_rows})")
    result = {}
    unreadable = 0
    for record in frame.to_dict("records"):
        symbol = str(record.get("SYMBOL") or "").strip().upper()
        if not symbol:
            continue
        listing_date = parse_nse_date(record.get("DATE OF LISTING"))
        if listing_date is None:
            unreadable += 1
        face_value = pd.to_numeric(str(record.get("FACE VALUE") or "").strip(), errors="coerce")
        result[symbol] = {
            "name": str(record.get("NAME OF COMPANY") or symbol).strip(),
            "series": str(record.get("SERIES") or "").strip().upper(),
            "listing_date": listing_date,
            "isin": (str(record.get("ISIN NUMBER") or "").strip() or None),
            "face_value": float(face_value) if pd.notna(face_value) else None,
        }
    # A few unreadable dates are tolerable (those stocks fail Stage 1); a
    # format change that breaks most of them is not.
    if result and unreadable > 0.05 * len(result):
        raise NseSchemaError(f"EQUITY_L: {unreadable} of {len(result)} listing dates unreadable")
    return result


_BONUS_RE = re.compile(r"bonus[^0-9]{0,20}?(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", re.I)
_FROM_TO_RE = re.compile(
    r"from\s*(?:rs\.?|re\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*/?-?.*?\bto\s*(?:rs\.?|re\.?|inr|₹)?\s*(\d+(?:\.\d+)?)",
    re.I | re.S,
)


def classify_action(subject: str) -> tuple[str, Optional[float]]:
    """Return ``(kind, factor)`` for one corporate-action subject line.

    ``factor`` divides prices before the ex-date (multiplies volume). Kinds:
    ``bonus``/``split``/``consolidation`` (adjustable), ``demerger``,
    ``rights``, ``scheme``, ``capital_reduction`` (not adjustable from the
    announcement) and ``other`` (dividends, AGMs, interest: no price change
    the screen needs to correct).
    """
    text = " ".join(str(subject or "").split())
    lower = text.lower()
    if re.search(r"de-?merger|spin[- ]?off", lower):
        return "demerger", None
    if re.search(r"scheme of arrangement|amalgamation|\bmerger\b", lower):
        return "scheme", None
    if re.search(r"capital reduction|reduction of (share )?capital", lower):
        return "capital_reduction", None
    if re.search(r"\brights?\b", lower):
        return "rights", None

    is_bonus = "bonus" in lower
    is_split = bool(re.search(r"split|sub-?\s?division", lower))
    is_consolidation = "consolidat" in lower
    if not (is_bonus or is_split or is_consolidation):
        return "other", None

    factor = 1.0
    kind = "bonus" if is_bonus else ("split" if is_split else "consolidation")
    if is_bonus:
        match = _BONUS_RE.search(text)
        if not match:
            return kind, None
        new, held = float(match.group(1)), float(match.group(2))
        if new <= 0 or held <= 0:
            return kind, None
        factor *= (new + held) / held
    if is_split or is_consolidation:
        match = _FROM_TO_RE.search(text)
        if not match:
            return ("split" if is_split else "consolidation"), None
        old_fv, new_fv = float(match.group(1)), float(match.group(2))
        if old_fv <= 0 or new_fv <= 0 or old_fv == new_fv:
            return ("split" if is_split else "consolidation"), None
        if is_split and old_fv < new_fv:
            # A "split" to a larger face value contradicts itself.
            return "split", None
        factor *= old_fv / new_fv
        if not is_bonus:
            kind = "split" if old_fv > new_fv else "consolidation"
    return kind, factor


def parse_corporate_actions(records) -> tuple[list, dict]:
    """Turn the NSE corporate-actions JSON into events.

    Returns ``(events, problems)``; ``problems`` maps a symbol to a reason when
    a price-affecting action has no readable ex-date, so it can be excluded.
    """
    if records is None:
        raise NseSchemaError("corporate actions response is empty")
    if isinstance(records, dict):
        records = records.get("data", records.get("rows"))
    if not isinstance(records, list):
        raise NseSchemaError("corporate actions response is not a list")
    events: set = set()
    problems: dict[str, str] = {}
    for item in records:
        if not isinstance(item, dict):
            raise NseSchemaError("corporate actions entry is not an object")
        if "symbol" not in item or "subject" not in item or "exDate" not in item:
            raise NseSchemaError(f"corporate actions entry missing fields: {sorted(item)}")
        symbol = str(item.get("symbol") or "").strip().upper()
        subject = " ".join(str(item.get("subject") or "").split())
        if not symbol:
            continue
        kind, factor = classify_action(subject)
        ex_date = parse_nse_date(item.get("exDate"))
        if ex_date is None:
            if kind in ADJUSTABLE_KINDS or kind in UNADJUSTABLE_PRICE_KINDS:
                problems.setdefault(symbol, f"corporate_action_undated:{subject}")
            continue
        events.add(CorporateActionEvent(symbol=symbol, ex_date=ex_date, kind=kind,
                                        subject=subject, factor=factor))
    return sorted(events, key=lambda e: (e.symbol, e.ex_date, e.subject)), problems


# ---------------------------------------------------------------------------
# Download client with a validated cache
# ---------------------------------------------------------------------------


def _default_nse_factory(folder: Path, timeout: int):
    from nse import NSE
    return NSE(download_folder=folder, server=False, timeout=timeout)


class NseEodClient:
    """Fetch NSE end-of-day files once, validate them, and keep them on disk."""

    # Truncation guards: real files have ~2,600 EQ bhavcopy rows, ~2,300
    # EQUITY_L rows and ~3,000 sec_list rows.
    min_bhav_rows = 500
    min_equity_rows = 1000
    min_band_rows = 500

    def __init__(self, cache_dir, *, timeout: int = 20, delay_s: float = 1.0,
                 nse_factory: Optional[Callable] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 today: Optional[Callable[[], date]] = None):
        self.cache_dir = Path(cache_dir)
        self.timeout = int(timeout)
        self.delay_s = max(0.0, float(delay_s))
        self._nse_factory = nse_factory or _default_nse_factory
        self._nse = None
        self._sleep = sleep
        self._today = today or (lambda: date.today())
        self._last_download = 0.0
        for sub in ("bhav", "bands", "ref", "surveillance", "actions", "_session"):
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)

    # -- low level ----------------------------------------------------------
    def close(self):
        if self._nse is not None:
            try:
                self._nse.exit()
            except Exception:
                pass
            self._nse = None

    def _client(self):
        if self._nse is None:
            try:
                self._nse = self._nse_factory(self.cache_dir / "_session", self.timeout)
            except Exception as exc:
                raise NseFileUnavailable(f"NSE website unreachable: {exc}") from exc
        return self._nse

    def _pace(self):
        wait = self.delay_s - (time.monotonic() - self._last_download)
        if wait > 0:
            self._sleep(wait)
        self._last_download = time.monotonic()

    def _download_text(self, url: str) -> str:
        """GET one archive file with the NSE session and return its text.

        Status codes are checked here rather than in the library, which writes
        any response body to disk: a 404 or an HTML error page means the file
        is not published, not that it has a new format.
        """
        nse = self._client()
        self._pace()
        try:
            response = nse._session.get(url, timeout=self.timeout)
        except Exception as exc:  # requests/httpx timeouts and connection errors
            raise NseFileUnavailable(f"NSE download failed for {url}: {exc}") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 404:
            raise NseFileNotPublished(f"{url}: not published (404)")
        if not 200 <= status < 300:
            raise NseFileUnavailable(f"{url}: HTTP {status}")
        content_type = str((getattr(response, "headers", {}) or {}).get("content-type", "")).lower()
        if "text/html" in content_type:
            raise NseFileNotPublished(f"{url}: NSE returned a web page, file not available")
        raw = response.content or b""
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise NseSchemaError(f"{url}: file is not text")

    def _request_json(self, url: str, params: Optional[dict] = None):
        nse = self._client()
        self._pace()
        try:
            response = nse._req(url, params=params)
        except Exception as exc:
            raise NseFileUnavailable(f"NSE request failed for {url}: {exc}") from exc
        try:
            return response.json()
        except Exception as exc:
            raise NseSchemaError(f"{url} did not return JSON: {exc}") from exc

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    # -- bhavcopy -----------------------------------------------------------
    def bhav_path(self, session: date) -> Path:
        return self.cache_dir / "bhav" / f"sec_bhavdata_full_{session:%d%m%Y}.csv"


    def bhavcopy(self, session: date) -> tuple[pd.DataFrame, bool]:
        """Return ``(EQ rows, downloaded)`` for one session, validated."""
        path = self.bhav_path(session)
        if path.exists():
            try:
                frame = parse_bhavcopy(path.read_text(encoding="utf-8"), min_rows=self.min_bhav_rows)
                if frame.attrs["session_date"] == session:
                    return frame, False
            except NseSchemaError as exc:
                logger.warning("Discarding invalid cached bhavcopy %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
        text = self._download_text(bhavcopy_url(session))
        frame = parse_bhavcopy(text, min_rows=self.min_bhav_rows)
        if frame.attrs["session_date"] != session:
            raise NseSchemaError(
                f"bhavcopy for {session} is dated {frame.attrs['session_date']}"
            )
        self._write_atomic(path, text)
        return frame, True


    # -- reference files ----------------------------------------------------
    def _latest_cached(self, folder: str, prefix: str, suffix: str, as_of: date,
                       max_age_days: int) -> Optional[tuple[Path, date]]:
        best = None
        for path in (self.cache_dir / folder).glob(f"{prefix}*{suffix}"):
            stamp = path.name[len(prefix):-len(suffix)] if suffix else path.name[len(prefix):]
            try:
                stamp_date = datetime.strptime(stamp, "%Y%m%d").date()
            except ValueError:
                continue
            if stamp_date > as_of or (as_of - stamp_date).days > max_age_days:
                continue
            if best is None or stamp_date > best[1]:
                best = (path, stamp_date)
        return best

    def equity_list(self, as_of: date, *, max_age_days: int = 5) -> tuple[dict, date]:
        path = self.cache_dir / "ref" / f"EQUITY_L_{as_of:%Y%m%d}.csv"
        if path.exists():
            try:
                return parse_equity_list(path.read_text(encoding="utf-8"), min_rows=self.min_equity_rows), as_of
            except NseSchemaError:
                path.unlink(missing_ok=True)
        try:
            text = self._download_text(EQUITY_LIST_URL)
            parsed = parse_equity_list(text, min_rows=self.min_equity_rows)
            self._write_atomic(path, text)
            return parsed, as_of
        except (NseFileUnavailable, NseSchemaError) as exc:
            cached = self._latest_cached("ref", "EQUITY_L_", ".csv", as_of, max_age_days)
            if cached is None:
                raise
            logger.warning("EQUITY_L download failed (%s); using cached copy from %s", exc, cached[1])
            return parse_equity_list(cached[0].read_text(encoding="utf-8"), min_rows=self.min_equity_rows), cached[1]


    def stock_industry(self, symbol: str) -> dict:
        """NSE's classification for one stock from its quote page (``secInfo``)."""
        nse = self._client()
        info = None
        for series in ("EQ", "BE", "BZ"):        # 19 Oct: BE/BZ stocks answer only with their own series
            self._pace()
            try:
                try:
                    quote = nse.quote(symbol, series=series)
                except TypeError:                # an older nse library: quote(symbol), EQ only
                    if series != "EQ":
                        break
                    quote = nse.quote(symbol)
            except Exception as exc:
                if series == "EQ":
                    raise NseFileUnavailable(f"NSE quote failed for {symbol}: {exc}") from exc
                continue                         # (a wrong series can fail instead of answering empty)
            info = (quote or {}).get("secInfo") if isinstance(quote, dict) else None
            if isinstance(info, dict) and any(info.get(k) for k in ("sector", "industryInfo", "industry", "macro")):
                break
            info = None
        if not isinstance(info, dict):
            raise NseSchemaError(f"NSE quote for {symbol} has no secInfo (EQ, BE or BZ)")

        def clean(key):
            value = str(info.get(key) or "").strip()
            return value or None
        result = {"macro": clean("macro"), "sector": clean("sector"),
                  "industry": clean("industryInfo") or clean("industry"), "basic_industry": clean("basicIndustry")}
        if not any(result.values()):
            raise NseSchemaError(f"NSE quote for {symbol} has an empty classification")
        return result

    def trading_holidays(self) -> dict:
        """NSE's trading holidays for the equity segment: ``{date: description}``
        (NSE lists the current year, and the next once announced, about December)."""
        data = self._request_json(HOLIDAYS_URL, params={"type": "trading"})
        rows = data.get("CM") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows:
            raise NseSchemaError("NSE holiday list has no equity (CM) rows")
        out = {}
        for r in rows:
            d = parse_nse_date(r.get("tradingDate"))
            if d is None:
                raise NseSchemaError(f"NSE holiday list: unreadable date {r.get('tradingDate')!r}")
            out[d] = str(r.get("description") or "holiday").strip()
        return out

    def corporate_actions(self, start: date, end: date) -> tuple[list, dict]:
        path = self.cache_dir / "actions" / f"actions_{start:%Y%m%d}_{end:%Y%m%d}.json"
        try:
            records = self._request_json(
                ACTIONS_URL,
                params={"index": "equities", "from_date": f"{start:%d-%m-%Y}",
                        "to_date": f"{end:%d-%m-%Y}"},
            )
            if (end - start).days > 14 and isinstance(records, list) and not records:
                # Every NSE fortnight has dividends and other actions; an
                # empty answer means the request was refused or changed.
                raise NseSchemaError(f"corporate actions empty for {start}..{end}")
            parsed = parse_corporate_actions(records)
            self._write_atomic(path, json.dumps(records))
            return parsed
        except (NseFileUnavailable, NseSchemaError) as exc:
            if path.exists():
                logger.warning("Corporate actions fetch failed (%s); using cached copy", exc)
                return parse_corporate_actions(json.loads(path.read_text(encoding="utf-8")))
            raise


    # -- housekeeping -------------------------------------------------------
