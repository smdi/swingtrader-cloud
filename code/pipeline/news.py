"""Company announcements from NSE, and the red-flag sale (user's rule, 15 Oct 2026).

Every evening-update tick (15 minutes) reads NSE's corporate announcements of
the last few days. An announcement for a stock the paper portfolio holds that
matches a red flag below - serious, company-specific bad news - places a sell
order (reason 'news'): at today's close if the market is open, else at the
next official open. The stock is not bought back for COOLDOWN sessions.

Only official exchange filings are read (not media headlines), and only clear
red flags count; routine filings (results dates, AGM notices, independent
directors leaving, ...) never trigger a sale. This rule cannot be backtested
(NSE's announcement history is not stored here: announcements without a red
flag are kept KEEP_DAYS, flagged ones for good), so every sale it makes is
listed on the Portfolio page with the filing and a link.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, time, timedelta

logger = logging.getLogger("trader.pipeline")
ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements"
LOOKBACK_DAYS = 3
KEEP_DAYS = 30                   # (22 Oct 2026) announcements with no red flag are deleted after this
MARKET_OPEN, MARKET_CLOSE = time(9, 15), time(15, 30)

_ROLE = r"(managing director|chief executive|\bceo\b|chief financial|\bcfo\b|whole[- ]?time director|\bm\.?d\.?\b)"
RED_FLAGS = (   # (label, pattern, exceptions within the same sentence)
    ("auditor resigned",
     r"resign\w*\W+(?:\w+\W+){0,10}?(statutory |joint |secretarial )?auditor|auditor\w*\W+(?:\w+\W+){0,10}?resign",
     None),
    ("missed a payment (default)",
     r"\bdefault\w*\b(?:\W+\w+){0,8}?\W+(payment|interest|principal|loan|debt|ncd|debenture|borrowing|repayment)|"
     r"non[- ]?payment of (interest|principal|dues)|delay(ed)? in (payment|servicing|repayment)",
     r"\bno (default|delay)|\bnil\b|not (been )?default\w*|no instance|\bnot applicable\b"),
    ("fraud / forensic audit", r"\bfraud\w*\b|forensic audit|siphon\w*|misappropriat\w*|embezzl\w*", None),
    ("regulator action",
     r"\bsebi\b(?:\W+\w+){0,8}?\W+(order|penalt\w*|debar\w*|restrain\w*)|show[- ]cause notice|"
     r"enforcement directorate|search and seizure|income[- ]tax (search|raid|survey)|\bcbi\b",
     r"settlement order|consent order|in the matter of (an|other) (entity|company)|insider trading|"
     r"goods and services tax|\bgst\b|commercial taxes|income[- ]tax (department|demand|assessment)|customs|"
     r"excise|service tax|demand notice|section 7[34]\b|"
     r"listing obligations|trading window|structural digital|regulation \d+"),
    ("insolvency", r"insolvency|\bcirp\b|winding[- ]up|liquidat\w*|committee of creditors|resolution professional|"
                   r"\bnclt\b(?:\W+\w+){0,8}?\W+(admit\w*|initiat\w*|moratorium)",
     r"acqui\w*|resolution plan|approv\w*|amalgamation|merger|appointed date|\bnclat\b|appeal|rectif\w*|"
     r"scheme of (amalgamation|arrangement|merger)|(subsidiar\w*|voluntar\w*|mutual fund|investment\w*|step[- ]down)"
     r"(?:\W+\w+){0,4}?\W+liquidat|liquidat\w*(?:\W+\w+){0,4}?\W+(subsidiar\w*|mutual fund|investment)|"
     r"against (a |the )?(customer|debtor|vendor|supplier|borrower)|filed by the company|company has filed|"
     # 19 Oct: a court's stay on a CIRP is relief; closing a (dormant) subsidiary is routine
     r"\bstay\b|voluntary winding|winding[- ]up(?:\W+\w+){0,10}?\W+subsidiar\w*|"
     r"subsidiar\w*(?:\W+\w+){0,6}?\W+(winding|liquidat)"),
    ("top manager resigned",
     r"(resign\w*|cessation|stepp?(ed|ing)? down|steps down)(?:\W+\w+){0,6}?\W+" + _ROLE + r"|"
     + _ROLE + r"(?:\W+\w+){0,6}?\W+(resign\w*|stepp?(ed|ing|s)? down|ceased?\b|has quit)",
     r"\bappoint\w*|superannuat\w*|retire\w*"),
    ("credit rating downgraded", r"downgrad\w*", r"\bno downgrad"),
    ("pledged shares sold by lenders", r"invocation of pledge|pledge\w*(?:\W+\w+){0,4}?\W+invoked", None),
    ("trading suspended", r"suspension of trading|trading(?:\W+\w+){0,4}?\W+suspend\w*",
     # 19 Oct: debentures/bonds suspended at maturity - the shares trade on
     r"revocation|debenture|\bncds?\b|\bbonds?\b|commercial paper|maturity|redemption"),
)
_COMPILED = [(label, re.compile(rx, re.I), re.compile(no, re.I) if no else None) for label, rx, no in RED_FLAGS]
# Within a sentence: words that mean the filing denies the news or is routine.
_BENIGN = re.compile(r"clarification|no such|not received any|upgrad\w*|re-?appoint\w*|reaffirm\w*|"
                     r"independent director|non[- ]executive|company secretary", re.I)
# 19 Oct: "... Whole Time Director of IL JIN" (Amber's subsidiary), "CIRP proceedings of Bobble" (not Affle):
# the person or the case belongs to another company named after "of"
_OF_OTHER = {
    "top manager resigned": re.compile(_ROLE + r"(?:\s+and\s+[\w ]+?)?\s+of\s+([A-Z][\w&.-]*)", re.I),
    "insolvency": re.compile(r"(?:\bcirp\b|insolvency|proceedings|winding[- ]up|liquidation)\W+(?:\w+\W+){0,2}?"
                             r"of\s+([A-Z][\w&.-]*)", re.I),
}
_OF_SELF = {"the", "its", "our", "this", "company", "m/s"}


def _about_another_company(label: str, unit: str, company: str | None) -> bool:
    """True when the matched role or case is named as another company's ("of IL JIN" for Amber)."""
    rx = _OF_OTHER.get(label)
    if not rx or not company:
        return False
    own = {w.strip(".,()").lower() for w in company.split()}
    for m in rx.finditer(unit):
        name = m.group(m.lastindex).strip(".,()").lower()
        if name and name not in _OF_SELF and name[0].isalpha() and name not in own:
            return True
    return False


_SENTENCE = re.compile(r"(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bM/s)(?<!\bLtd)(?<!\bNo)(?<!\bCo)\.\s+(?=[A-Z])|[;\n]+")


# Routine filings by their NSE subject: never a red flag, whatever regulation they quote
# (15 Oct, first live read: "Trading Window" and "Structural Digital Database" filings
# quote the SEBI (Prohibition of Insider Trading) Regulations).
_ROUTINE_SUBJECT = re.compile(
    r"^(trading window|structural digital database|closure of trading window|compliance certificate|"
    r"certificate under|shareholders? meeting|agm|egm|newspaper (publication|advertisement)|"
    r"board meeting( intimation)?|analysts?/institutional investor meet|investor presentation|"
    r"loss of share certificates?|issue of duplicate|copy of newspaper|record date|dividend|"
    r"esop|allotment|change in (registrar|rta)|book closure|updates? on (esop|allotment))", re.I)


def red_flag(subject: str, text: str = "", company: str | None = None) -> str | None:
    """The red flag an announcement raises, or None. Each sentence of the
    filing is read with its subject; a sentence that denies the news or is
    routine (an independent director leaving, a rating upgrade, ...) raises
    nothing, whatever the other sentences say. ``company``: the filer's name
    (NSE's sm_name) - a manager leaving, or an insolvency case, of another
    company named in the filing (a subsidiary, a debtor) is not a red flag."""
    subject = (subject or "").strip()
    if _ROUTINE_SUBJECT.search(subject):
        return None
    parts = [p.strip() for p in _SENTENCE.split(text or "") if p and p.strip()] or [""]
    for part in parts:
        unit = f"{subject}. {part}"
        if _BENIGN.search(unit):
            continue
        for label, rx, no in _COMPILED:
            if rx.search(unit) and not (no and no.search(unit)) and \
                    not _about_another_company(label, unit, company):
                return label
    return None


def _parse_time(value) -> datetime | None:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except (TypeError, ValueError):
            continue
    return None


def parse(records) -> list:
    """NSE's announcement records -> plain rows (fields NSE uses, checked)."""
    if isinstance(records, dict):
        records = records.get("data") or records.get("announcements") or []
    if not isinstance(records, list):
        raise ValueError("NSE announcements: unexpected reply")
    rows = []
    for r in records:
        if not isinstance(r, dict) or not r.get("symbol"):
            continue
        when = _parse_time(r.get("an_dt") or r.get("sort_date") or r.get("dt"))
        if when is None:
            continue
        subject = str(r.get("desc") or r.get("subject") or "").strip()
        text = str(r.get("attchmntText") or r.get("text") or "").strip()
        key = str(r.get("seq_id") or "") or hashlib.sha1(
            f"{r['symbol']}|{when.isoformat()}|{subject}|{text[:200]}".encode()).hexdigest()
        rows.append({"id": key, "symbol": str(r["symbol"]).strip().upper(), "at": when, "subject": subject,
                     "text": text, "link": r.get("attchmntFile") or None,
                     "flag": red_flag(subject, text, r.get("sm_name"))})
    return rows


def fetch(client, start: date, end: date) -> list:
    return parse(client._request_json(ANNOUNCEMENTS_URL, params={
        "index": "equities", "from_date": f"{start:%d-%m-%Y}", "to_date": f"{end:%d-%m-%Y}"}))


def store_rows(conn, rows, now: datetime) -> list:
    """Save announcements; returns the ones not seen before."""
    new = []
    for r in rows:
        link = r["link"] if str(r["link"] or "").startswith("https://") else None
        cur = conn.execute("INSERT OR IGNORE INTO announcements (id, symbol, at, subject, text, link, flag, seen_at) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (r["id"], r["symbol"], r["at"].isoformat(sep=" "), r["subject"], r["text"][:2000],
                            link, r["flag"], now.isoformat(sep=" ", timespec="seconds")))
        if cur.rowcount:
            new.append(r)
    return new


def unacted(conn, now: datetime) -> list:
    """Red-flag announcements of the last LOOKBACK_DAYS not yet turned into a
    sale (a stock bought later the same day, or one with another order waiting,
    is looked at again on every tick)."""
    since = (now - timedelta(days=LOOKBACK_DAYS)).isoformat(sep=" ")
    rows = conn.execute("SELECT id, symbol, at, subject, link, flag FROM announcements WHERE flag IS NOT NULL "
                        "AND acted IS NULL AND at >= ? ORDER BY at", (since,)).fetchall()
    return [{"id": i, "symbol": s, "at": datetime.fromisoformat(a), "subject": sub, "link": link, "flag": flag}
            for i, s, a, sub, link, flag in rows]


def act(conn, rows, now: datetime, cal) -> list:
    """Sell orders for held stocks with a red-flag announcement made after
    they were bought. Returns [(symbol, flag, order day, at)]."""
    from strategy import live
    out = []
    held = dict(conn.execute("SELECT symbol, opened_on FROM paper_positions"))
    waiting = {r[0] for r in conn.execute("SELECT symbol FROM paper_orders WHERE status = 'pending'")}
    industry = dict(conn.execute("SELECT symbol, industry FROM sectors"))
    for r in rows:
        s = r["symbol"]
        if not r["flag"] or s not in held or s in waiting:
            continue
        # news known before the stock was bought (it was bought at an open, so
        # anything after the previous session's close counts) is not acted on
        if r["flag"].startswith("fraud") and industry.get(s) == "Banks":
            # banks must report every fraud found in a borrower's account: routine for a bank
            conn.execute("UPDATE announcements SET acted = 'bank: borrower fraud report' WHERE id = ?", (r["id"],))
            continue
        bought = date.fromisoformat(held[s])
        if r["at"] < datetime.combine(cal.prev_trading_day(bought), MARKET_CLOSE):
            conn.execute("UPDATE announcements SET acted = 'before purchase' WHERE id = ?", (r["id"],))
            continue
        today = now.date()
        if cal.is_trading_day(today) and MARKET_OPEN <= now.time() <= MARKET_CLOSE:
            day, at = today, "close"
        elif cal.is_trading_day(today) and now.time() < MARKET_OPEN:
            day, at = today, "open"
        else:
            day, at = cal.next_trading_day(today), "open"
        signal = r["at"].date()
        if not cal.is_trading_day(signal):
            signal = cal.prev_trading_day(signal)
        order = live.place_sell(conn, s, day, at, reason="news", signal_day=signal,
                                detail=json.dumps({"flag": r["flag"], "subject": r["subject"], "link": r["link"],
                                                   "at": r["at"].isoformat(sep=" ")}))
        conn.execute("UPDATE announcements SET acted = ? WHERE id = ?", (f"order {order}", r["id"]))
        waiting.add(s)
        out.append((s, r["flag"], day.isoformat(), at))
    return out


def check(conn, client, now: datetime, cal=None) -> dict:
    """One tick: read NSE's recent announcements, save them, sell on red flags."""
    from strategy import live
    if cal is None:
        from core import market_calendar as cal
    if not conn.execute("SELECT 1 FROM paper_positions LIMIT 1").fetchone():
        return {"read": 0, "sales": []}
    rows = fetch(client, now.date() - timedelta(days=LOOKBACK_DAYS), now.date())
    with live.BOOK_LOCK:
        live.begin(conn)
        try:
            new = store_rows(conn, rows, now)
            sales = act(conn, unacted(conn, now), now, cal)
            conn.execute("DELETE FROM announcements WHERE flag IS NULL AND at < ?",
                         ((now - timedelta(days=KEEP_DAYS)).isoformat(sep=" "),))
            if sales:
                conn.execute("INSERT INTO job_log (job, started_at, finished_at, status, detail) VALUES "
                             "('news', ?, ?, 'ok', ?)", (now.isoformat(sep=" ", timespec="seconds"),) * 2
                             + (json.dumps({"sales": sales}),))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    for s, flag, day, at in sales:
        logger.warning("Red-flag announcement for %s (%s): sell order at the official %s on %s", s, flag, at, day)
    return {"read": len(rows), "new": len(new), "sales": sales}
