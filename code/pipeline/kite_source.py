"""Kite: the backup for daily prices and the source of live prices on screen
(Redesign v2, Phase 2). Nothing the strategy decides depends on it.

Kite's full quote gives, per stock: last price, today's open/high/low, the
previous close (``ohlc.close``), volume and average price. After the close the
last price is the day's closing price (NSE's official close can differ by a
few paise: NSE's file replaces these rows as soon as it is read).
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger("trader.pipeline")
BATCH = 500                    # Kite allows up to 500 instruments per quote call
INDEX_INSTRUMENT = "NSE:NIFTY 50"


def _client():
    from pipeline.kite_client import get_client, is_configured
    if not is_configured():
        raise RuntimeError("Kite is not configured (no API key or today's login)")
    return get_client()


SERIES_SUFFIXES = ("-BE", "-BZ")   # Kite's names for NSE's trade-for-trade series (HFCL-BE)


def quotes(symbols, client=None, pause_s: float = 0.4) -> dict:
    """``{symbol: quote}`` for NSE symbols; missing symbols are left out. A
    stock in NSE's BE/BZ series is quoted by Kite as SYMBOL-BE / SYMBOL-BZ:
    those not found by their plain name are asked for again with the suffixes."""
    client = client or _client()
    out = _ask(client, {f"NSE:{s}": s for s in symbols}, pause_s)
    missing = [s for s in symbols if s not in out]
    if missing:
        out.update(_ask(client, {f"NSE:{s}{x}": s for s in missing for x in SERIES_SUFFIXES}, pause_s))
    return out


def _ask(client, names: dict, pause_s: float) -> dict:
    keys, out = list(names), {}
    for i in range(0, len(keys), BATCH):
        reply = client.quote(keys[i:i + BATCH]) or {}
        for key, q in reply.items():
            if key in names:
                out.setdefault(names[key], q)
        if i + BATCH < len(keys):
            time.sleep(pause_s)
    return out


def live_prices(symbols, client=None) -> dict:
    """``{symbol: last price}`` for the screen's 30-second refresh. Empty when
    Kite is unavailable (the screen then shows the last close)."""
    try:
        return {s: float(q["last_price"]) for s, q in quotes(symbols, client).items()
                if q.get("last_price")}
    except Exception as exc:
        logger.info("Kite live prices unavailable: %s", exc)
        return {}


def day_opens(symbols, day: date, client=None) -> dict:
    """``{symbol: today's opening price}`` for the morning fills: only stocks
    whose last trade is on ``day`` (a stock not traded today still shows an
    older day's open). Raises when Kite is not logged in."""
    out = {}
    for s, q in quotes(symbols, client).items():
        opened = (q.get("ohlc") or {}).get("open")
        traded = q.get("last_trade_time")
        if not opened or float(opened) <= 0 or not traded:
            continue
        traded_on = traded.date() if hasattr(traded, "date") and callable(traded.date) \
            else date.fromisoformat(str(traded)[:10])
        if traded_on == day:
            out[s] = float(opened)
    return out


def traded_prices(symbols, day: date, client=None) -> dict:
    """``{symbol: last price}`` for stocks whose last trade is on ``day`` - the
    stop during the day never acts on yesterday's price. Raises when Kite is
    not logged in."""
    out = {}
    for s, q in quotes(symbols, client).items():
        price, traded = q.get("last_price"), q.get("last_trade_time")
        if not price or float(price) <= 0 or not traded:
            continue
        traded_on = traded.date() if hasattr(traded, "date") and callable(traded.date) \
            else date.fromisoformat(str(traded)[:10])
        if traded_on == day:
            out[s] = float(price)
    return out


def _traded_on(q):
    """The date of a quote's last trade (or, for an index, its timestamp); None if unknown."""
    when = q.get("last_trade_time") or q.get("timestamp")
    if not when:
        return None
    try:
        return when.date() if hasattr(when, "date") and callable(when.date) else date.fromisoformat(str(when)[:10])
    except ValueError:
        return None


def _today_ist() -> date:
    from core import timeutil
    return timeutil.ist_today()


def day_rows(symbols, day: date, client=None, check: dict | None = None) -> pd.DataFrame:
    """One day's prices from Kite, shaped like NSE's rows (the backup). Today:
    Kite's quotes, only stocks whose last trade is on ``day`` (an untraded stock
    shows an older day's prices, 22 Oct 2026). (2 Nov 2026) An earlier day:
    Kite's daily history - NSE's website may be unreachable for days. ``check``:
    {symbol: (date, NSE's close)} to compare that history with (see history_rows)."""
    if day < _today_ist():
        return history_rows(symbols, day, client, check=check)
    rows = []
    for s, q in quotes(symbols, client).items():
        ohlc = q.get("ohlc") or {}
        last = q.get("last_price")
        if not last or not ohlc.get("open") or _traded_on(q) != day:
            continue
        volume = q.get("volume") or q.get("volume_traded") or 0
        avg = q.get("average_price") or last
        rows.append(dict(symbol=s, date=day, open=ohlc.get("open"), high=ohlc.get("high"), low=ohlc.get("low"),
                         close=float(last), prev_close=ohlc.get("close"), volume=float(volume),
                         turnover=float(volume) * float(avg), series="EQ"))
    return pd.DataFrame(rows)


def index_close(day: date | None = None, client=None) -> dict:
    """NIFTY 50 from Kite; with ``day``, only if Kite's quote is from that day
    (an earlier day: from Kite's daily history)."""
    client = client or _client()
    if day is not None and day < _today_ist():
        token = _tokens(client).get(INDEX_INSTRUMENT.split(":", 1)[1])
        if token is None:
            raise RuntimeError("NIFTY 50 is not on Kite's instrument list")
        candle = _candles(client, token, day, day).get(day)
        if not candle:
            raise RuntimeError(f"Kite has no NIFTY 50 day candle for {day}")
        return {k: candle[k] for k in ("open", "high", "low", "close")}
    q = (client.quote([INDEX_INSTRUMENT]) or {}).get(INDEX_INSTRUMENT) or {}
    ohlc = q.get("ohlc") or {}
    if not q.get("last_price"):
        raise RuntimeError("Kite returned no NIFTY 50 quote")
    if day is not None and _traded_on(q) != day:
        raise RuntimeError(f"Kite's NIFTY 50 quote is not from {day}")
    return {"open": ohlc.get("open"), "high": ohlc.get("high"), "low": ohlc.get("low"),
            "close": float(q["last_price"])}


# ------------------------------------------------------------------ Kite's daily history (2 Nov 2026)
HISTORY_PAUSE_S = 0.35         # Kite's historical API allows about 3 requests a second
_TOKENS: dict = {}             # {date fetched: {tradingsymbol: token}}
_HISTORY: dict = {}            # {symbol: {date: candle}} for the range below
_HISTORY_RANGE: list = [None, None]


def _tokens(client) -> dict:
    """{trading symbol: instrument token} for NSE (stocks and indices), once a day."""
    today = _today_ist()
    if today not in _TOKENS:
        _TOKENS.clear()
        rows = client.instruments("NSE") or []
        _TOKENS[today] = {r["tradingsymbol"]: r["instrument_token"] for r in rows if r.get("tradingsymbol")}
    return _TOKENS[today]


def _token_for(tokens: dict, symbol: str):
    for name in (symbol, *(f"{symbol}{x}" for x in SERIES_SUFFIXES)):
        if name in tokens:
            return tokens[name]
    return None


def _candles(client, token, first: date, last: date) -> dict:
    out = {}
    for c in client.historical_data(token, first, last, "day") or []:
        when = c.get("date")
        d = when.date() if hasattr(when, "date") and callable(when.date) else date.fromisoformat(str(when)[:10])
        if c.get("close"):
            out[d] = c
    return out


CHECK_TOLERANCE = 0.005       # Kite's close vs NSE's on the same day: more apart = adjusted for a later action


def history_rows(symbols, day: date, client=None, *, sleep=time.sleep, check: dict | None = None) -> pd.DataFrame:
    """``day``'s prices from Kite's daily candles, shaped like NSE's rows. The first
    call reads every day from ``day`` to yesterday for all ``symbols`` at once (one
    request a stock), so the next missed days come from memory, not 15 more minutes
    of requests. A stock that fails is left out (NSE's file fills it later).
    (3 Nov 2026 review) Kite's daily candles are adjusted for later splits and bonuses:
    with ``check`` ({symbol: (date, NSE's close)}) the history starts on that date and a
    stock whose Kite close then differs from NSE's is left out - no price beats a wrong one."""
    client = client or _client()
    last = _today_ist() - timedelta(days=1)
    check = check or {}
    start = min([day] + [d for d, _ in check.values()])
    first, upto = _HISTORY_RANGE
    want = [s for s in symbols if s not in _HISTORY]
    if first is None or start < first or day > upto or want:
        if first is None or start < first or day > upto:
            _HISTORY.clear()
            want = list(symbols)
            _HISTORY_RANGE[:] = [start, last]
        first, upto = _HISTORY_RANGE
        tokens = _tokens(client)
        logger.info("Kite daily history: %d stocks, %s to %s", len(want), first, upto)
        failed = 0
        for i, s in enumerate(want):
            token = _token_for(tokens, s)
            if token is None:
                _HISTORY[s] = {}
                continue
            try:
                _HISTORY[s] = _candles(client, token, first, upto)
            except Exception as exc:                          # noqa: BLE001
                failed += 1
                if failed >= 20 and failed > i // 2:
                    raise RuntimeError(f"Kite's daily history keeps failing ({exc})") from exc
                if "too many" in str(exc).lower():
                    sleep(2.0)
            sleep(HISTORY_PAUSE_S)
    rows = []
    for s in symbols:
        c = (_HISTORY.get(s) or {}).get(day)
        if not c or not c.get("open"):
            continue
        if s in check:
            ref_day, nse_close = check[s]
            ref = (_HISTORY.get(s) or {}).get(ref_day)
            if not ref or abs(float(ref["close"]) / nse_close - 1) > CHECK_TOLERANCE:
                continue                                      # adjusted (or unknown): left out
        volume = float(c.get("volume") or 0)
        rows.append(dict(symbol=s, date=day, open=c["open"], high=c.get("high"), low=c.get("low"),
                         close=float(c["close"]), prev_close=None, volume=volume,
                         turnover=volume * float(c["close"]), series="EQ"))
    return pd.DataFrame(rows)
