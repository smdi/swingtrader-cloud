"""News about the stocks you hold, from Google News (Indian media), with a
good / bad / neutral label - for reading only, it never buys or sells
(15 Oct 2026, user's decision). Automatic selling stays on NSE's official
filings (pipeline/news.py).

Fetched at most once an hour (by the 15-minute update) for each holding,
searching for the company's name; a headline is kept only when it names the
company. The label comes from the words in the headline (a short finance word
list), and the words that decided it are shown beside it, so a wrong label is
easy to spot.
"""
from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

logger = logging.getLogger("trader.pipeline")
FETCH_EVERY = timedelta(hours=1)
KEEP_DAYS = 14
IST = timezone(timedelta(hours=5, minutes=30))

GOOD = (
    r"surg\w*", r"soar\w*", r"jump\w*", r"rall(y|ies|ied)", r"zoom\w*", r"gain\w*", r"climb\w*", r"rebound\w*",
    r"record high", r"52-week high", r"all-time high", r"upper circuit", r"beats? (estimates|expectations)",
    r"profit (rises|jumps|grows|up|surges)", r"revenue (rises|jumps|grows|up)", r"upgrad\w*",
    r"(raises?|hikes?) (target|guidance)", r"bags?\b", r"wins? (an? )?(order|contract|deal)",
    r"(order|contract) win", r"buyback", r"bonus (issue|shares)", r"(higher|record|special) dividend",
    r"approval", r"strong (results|growth|demand|quarter)", r"outperform\w*", r"top gainer", r"expan(ds|sion)",
)
BAD = (
    r"fall\w*", r"\bfell\b", r"slump\w*", r"plung\w*", r"crash\w*", r"drop\w*", r"slid(e|es|ing)?\b", r"tumbl\w*",
    r"sink\w*", r"\bsank\b", r"declin\w*", r"lower circuit", r"52-week low", r"sell-?off", r"top loser",
    r"loss(es)? (widen\w*|deepen\w*)", r"net loss", r"miss(es)? (estimates|expectations)",
    r"profit (falls|drops|declines|down|slumps)", r"revenue (falls|drops|declines|down)", r"downgrad\w*",
    r"(cuts?|lowers?) (target|guidance)", r"\bfraud\w*", r"\braid\w*", r"\bprobe\b", r"penalt\w*", r"\bban(ned)?\b",
    r"recall\w*", r"layoffs?", r"job cuts", r"resign\w*", r"steps? down", r"default\w*", r"insolvency",
    r"complaints?", r"show[- ]cause", r"weak (results|demand|quarter|growth)", r"concern\w*", r"\bwoes?\b",
    r"pledge", r"block deal", r"stake sale",
)
_GOOD = [re.compile(r"\b" + w, re.I) for w in GOOD]
_BAD = [re.compile(r"\b" + w, re.I) for w in BAD]


def sentiment(title: str) -> tuple[str, list]:
    """('good' | 'bad' | 'neutral', the words that decided it). More good words
    than bad: good; more bad: bad; none or a tie: neutral. A headline asking a
    question ("Buy, hold or sell?", "More upside ahead?") is an opinion: neutral
    unless it also reports a fact with the same number of words either way."""
    text = re.sub(r"\s+-\s+[^-]+$", "", title or "")          # drop " - Source name" at the end
    good = [m.group(0) for rx in _GOOD for m in rx.finditer(text)]
    bad = [m.group(0) for rx in _BAD for m in rx.finditer(text)]
    if text.strip().endswith("?") and abs(len(good) - len(bad)) <= 1:
        return "neutral", good + bad
    if len(good) > len(bad):
        return "good", good
    if len(bad) > len(good):
        return "bad", bad
    return "neutral", good + bad


def name_variants(symbol: str, name: str | None) -> list:
    """Ways a headline may name the company: symbol, full name, first word(s)."""
    base = re.sub(r"\b(limited|ltd\.?|india|industries|corporation|company)\b\.?", "", name or "", flags=re.I)
    base = re.sub(r"\s+", " ", base).strip()
    words = base.split()
    out = {symbol.upper()}
    if base:
        out.add(base)
    if words and len(words[0]) >= 3:
        out.add(words[0])
    if len(words) >= 2:
        out.add(" ".join(words[:2]))
    return sorted(out, key=len, reverse=True)


def query_url(symbol: str, name: str | None) -> str:
    words = (name or symbol)
    words = re.sub(r"\b(limited|ltd\.?)$", "", words, flags=re.I).strip()
    q = quote(f'"{words}" share')
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def parse_rss(content: bytes, symbol: str, name: str | None) -> list:
    """Headlines that name the company, newest first."""
    variants = [re.compile(r"\b" + re.escape(v) + r"\b", re.I) for v in name_variants(symbol, name)]
    rows, seen = [], set()
    for item in ET.fromstring(content).findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = (item.findtext("source") or "").strip()
        try:
            when = parsedate_to_datetime(item.findtext("pubDate")).astimezone(IST).replace(tzinfo=None)
        except (TypeError, ValueError):
            continue
        headline = re.sub(r"\s+-\s+[^-]+$", "", title)
        key = headline.lower()
        if not title or key in seen or not any(v.search(headline) for v in variants):
            continue
        seen.add(key)
        label, words = sentiment(title)
        rows.append({"id": hashlib.sha1(f"{symbol}|{link or title}".encode()).hexdigest(), "symbol": symbol,
                     "at": when, "title": headline, "source": source, "link": link if link.startswith("https://")
                     else None, "sentiment": label, "words": ", ".join(dict.fromkeys(w.lower() for w in words))})
    rows.sort(key=lambda r: r["at"], reverse=True)
    return rows


def refresh(conn, now: datetime, get=None, force: bool = False) -> dict:
    """Fetch headlines for every holding, at most once an hour."""
    from store import db as store
    last = store.get_meta(conn, "media_news_at")
    if not force and last and now - datetime.fromisoformat(last) < FETCH_EVERY:
        return {"skipped": True}
    if get is None:
        import requests

        def get(url):
            return requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).content
    held = conn.execute("SELECT p.symbol, i.name FROM paper_positions p "
                        "LEFT JOIN instruments i ON i.symbol = p.symbol").fetchall()
    added, failed, fetched = 0, [], []
    for symbol, name in held:          # all fetching first: no write lock is held while the network is slow
        try:
            fetched.extend(parse_rss(get(query_url(symbol, name)), symbol, name)[:30])
        except Exception as exc:
            failed.append(f"{symbol}: {str(exc)[:80]}")
    for r in fetched:                  # then one short write
        cur = conn.execute("INSERT OR IGNORE INTO media_news (id, symbol, at, title, source, link, sentiment, words) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (r["id"], r["symbol"], r["at"].isoformat(sep=" "), r["title"], r["source"], r["link"],
                            r["sentiment"], r["words"]))
        added += cur.rowcount
    conn.execute("DELETE FROM media_news WHERE at < ?", ((now - timedelta(days=KEEP_DAYS)).isoformat(sep=" "),))
    store.set_meta(conn, "media_news_at", now.isoformat(sep=" ", timespec="seconds"))
    conn.commit()
    if failed:
        logger.info("Google News not read for %s", "; ".join(failed[:5]))
    return {"stocks": len(held), "added": added, "failed": failed}
