"""Strength portfolio core (Redesign v2, Phase 1).

One set of rules, used unchanged by the backtest and by the live paper
portfolio, so the two cannot drift apart:

  decide()   at a close: rank the screened stocks by past return and name the
             stocks to keep (the top ``2 * n``) and the order to buy in;
  execute()  at the next open: sell holdings not kept, then buy the best
             ranked stocks not held, in equal slots, until ``n`` are held;
  mark()     value the book at a close;
  carry_corporate_action()  a split or bonus on a held stock keeps its value.

Rules and numbers are those of scripts/strength_portfolio.py (14 Oct study,
'top20 monthly 6m': +23.1%/yr 2010-18, +27.1%/yr 2019-26 after costs);
tests/test_strategy_core.py checks this module reproduces that script exactly.

Arrays are indexed by stock position: every row passed in for one day has one
entry per stock, in the same order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

SLIPPAGE = 0.001                 # 0.1% each side (as the backtest)
JUMP_LOW, JUMP_HIGH = 0.55, 1.8  # a day-to-day ratio outside this is a split/bonus, not a move
LOOKBACK_SESSIONS = {"6m": 126}
SKIP_12_1 = (21, 252)            # 12 months back, skipping the most recent month


@dataclass(frozen=True)
class Rules:
    holdings: int = 20           # n: stocks held
    keep_rank: int = 40          # a holding is sold when it ranks below this (2 * n)
    lookback: str = "nse"        # "nse" (NSE-style, live from 15 Oct 2026), "6m" or "12-1m"
    rebalance: str = "month_end"  # "month_end", "week_end" or "every:<sessions>"
    stop_loss: float | None = None  # disaster stop: sell a holding whose close is this far below its buy price
    sector_cap: int | None = None   # at most this many holdings from one sector (buys skip a full sector)
    stop_intraday: bool = False     # the stop acts during the day: sold when the price touches it (16 Oct 2026)


@dataclass(frozen=True)
class Overlay:
    """Safety rules on top of the monthly rules (Phase 6 tests; the disaster stop
    is live through Rules.stop_loss).
    Each is checked at a session's close and acted on at the next open."""
    stop_loss: float | None = None      # sell a holding whose close is this fraction below its buy price
    trailing: float | None = None       # ... this fraction below its highest close since it was bought
    market_filter: int | None = None    # NIFTY below its N-session average: hold cash, no buying
    vol_exit: float | None = None       # sell when 20-day volatility > this x its 126-day volatility
    intraday: bool = False              # stops act during the day: sold the day the low touches the stop price,
                                        # at that price (at the open when it opens below it) - needs the lows

    @property
    def name(self) -> str:
        parts = [f"stop {self.stop_loss:.0%}" if self.stop_loss else "",
                 f"trailing {self.trailing:.0%}" if self.trailing else "",
                 f"NIFTY < {self.market_filter}-day avg: cash" if self.market_filter else "",
                 f"volatility x{self.vol_exit:g}: sell" if self.vol_exit else "",
                 "during the day" if self.intraday else ""]
        return ", ".join(p for p in parts if p) or "no safety rule"


@dataclass
class Costs:
    buy_rate: float              # statutory charges as a fraction of the buy value
    sell_rate: float             # ... of the sell value (DP charge excluded)
    dp_per_sell: float = 15.34   # depository charge per stock sold
    slippage: float = SLIPPAGE

    @classmethod
    def from_config(cls):
        from config import Config
        from core.charges import buy_charge_rate, sell_charge_rate
        return cls(buy_charge_rate(), sell_charge_rate(),
                   float(getattr(Config, "DP_CHARGE_PER_SELL_DAY", 15.34)))


@dataclass
class Book:
    cash: float
    shares: np.ndarray           # shares held per stock position

    @classmethod
    def new(cls, capital: float, stocks: int) -> "Book":
        return cls(float(capital), np.zeros(stocks))


@dataclass(frozen=True)
class Target:
    keep: frozenset              # stock positions that may stay held
    buy_order: tuple             # stock positions to buy from, best first


@dataclass(frozen=True)
class Fill:
    stock: int
    side: str                    # "BUY" or "SELL"
    shares: float
    price: float                 # the official open (before slippage)
    fill_price: float            # after slippage
    value: float                 # rupees paid (buy, charges included) or received (sell, net)
    charges: float               # statutory charges + DP charge
    slippage_cost: float


@dataclass
class ExecutionReport:
    fills: list = field(default_factory=list)
    costs: float = 0.0
    turnover: float = 0.0


def scores(close_ff: np.ndarray, lookback: str, listed: np.ndarray | None = None) -> np.ndarray:
    """Ranking score per (session, stock) from forward-filled closes: the past
    return ("6m", "12-1m") or the NSE-style score ("nse", needs ``listed``)."""
    T, S = close_ff.shape
    if lookback == "nse":
        if listed is None:
            raise ValueError("the NSE-style score needs the screened stocks (listed)")
        return nse_style_scores(close_ff, listed)
    if lookback in LOOKBACK_SESSIONS:
        k = LOOKBACK_SESSIONS[lookback]
        past = np.vstack([np.full((k, S), np.nan), close_ff[:-k]])
        return close_ff / past - 1
    if lookback == "12-1m":
        skip, total = SKIP_12_1
        a = np.vstack([np.full((skip, S), np.nan), close_ff[:-skip]])
        b = np.vstack([np.full((total, S), np.nan), close_ff[:-total]])
        return a / b - 1
    raise ValueError(f"unknown lookback {lookback!r}")


def nse_style_scores(close_ff: np.ndarray, listed: np.ndarray) -> np.ndarray:
    """Momentum the way NSE's momentum indices describe it: the 6- and 12-month
    returns, each divided by the stock's yearly volatility, turned into
    z-scores across the screened stocks of the day and averaged."""
    T, S = close_ff.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.vstack([np.full((1, S), np.nan), np.diff(np.log(close_ff), axis=0)])
    import pandas as pd
    vol = pd.DataFrame(logret).rolling(252, min_periods=200).std().to_numpy() * np.sqrt(252)

    def past(k):
        return np.vstack([np.full((k, S), np.nan), close_ff[:-k]])

    out = np.zeros((T, S))
    for k in (126, 252):
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (close_ff / past(k) - 1) / vol
        x = np.where(listed & np.isfinite(x), x, np.nan)
        import warnings
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # days with no history yet
            mean = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
        out += (x - mean) / np.where(sd > 0, sd, np.nan) / 2
    return out


def volatility_ratio(close_ff: np.ndarray) -> np.ndarray:
    """20-session volatility divided by 126-session volatility, per (session, stock)."""
    import pandas as pd
    with np.errstate(divide="ignore", invalid="ignore"):
        r = pd.DataFrame(np.diff(np.log(close_ff), axis=0, prepend=np.nan))
    return (r.rolling(20, min_periods=18).std() / r.rolling(126, min_periods=100).std()).to_numpy()


def decide(score_row: np.ndarray, listed_row: np.ndarray, close_row: np.ndarray,
           rules: Rules) -> Target:
    """At a close: the stocks to keep (top ``keep_rank``) and to buy from.

    Only stocks on that day's screened list with a score and a close count."""
    eligible = listed_row & np.isfinite(score_row) & np.isfinite(close_row)
    ranked = np.where(eligible, score_row, -np.inf)
    order = np.argsort(-ranked)
    order = [int(s) for s in order if np.isfinite(ranked[s])][:rules.keep_rank]
    return Target(frozenset(order), tuple(order))


def carry_corporate_action(book: Book, ratio_row: np.ndarray) -> int:
    """A held stock whose close moved outside JUMP_LOW..JUMP_HIGH in a day had
    a split, bonus or consolidation: scale its shares so its value is kept.
    Returns how many holdings were adjusted."""
    bad = (book.shares > 0) & np.isfinite(ratio_row) & ~((ratio_row > JUMP_LOW) & (ratio_row < JUMP_HIGH))
    if bad.any():
        book.shares[bad] /= ratio_row[bad]
    return int(bad.sum())


def execute(book: Book, target: Target, open_row: np.ndarray, last_close_row: np.ndarray,
            rules: Rules, costs: Costs, groups: np.ndarray | None = None) -> ExecutionReport:
    """At the open after a decision: sells first, then buys in equal slots.

    A stock with no open that day is sold at its last close (it may have
    stopped trading) and is not bought."""
    report = ExecutionReport()
    tradeable = np.isfinite(open_row) & (open_row > 0)
    price_now = np.where(tradeable, open_row, last_close_row)
    for s in np.flatnonzero(book.shares > 0):
        if int(s) in target.keep:
            continue
        qty = book.shares[s]
        gross = qty * price_now[s] * (1 - costs.slippage)
        fee = gross * costs.sell_rate + costs.dp_per_sell
        book.cash += gross - fee
        slip = qty * price_now[s] * costs.slippage
        report.costs += fee + slip
        report.turnover += gross
        report.fills.append(Fill(int(s), "SELL", float(qty), float(price_now[s]),
                                 float(price_now[s] * (1 - costs.slippage)), float(gross - fee), float(fee),
                                 float(slip)))
        book.shares[s] = 0.0
    held = int((book.shares > 0).sum())
    value = book.cash + float(np.nansum(book.shares * price_now))
    slot = value / rules.holdings
    capped = rules.sector_cap is not None and groups is not None
    if capped:     # holdings per sector after the sales (-1: sector unknown, never capped)
        count = np.bincount(groups[(book.shares > 0) & (groups >= 0)], minlength=int(groups.max(initial=-1)) + 1)
    for s in target.buy_order:
        if held >= rules.holdings or book.cash < slot * 0.5:
            break
        if book.shares[s] > 0 or not tradeable[s]:
            continue
        if capped and groups[s] >= 0 and count[groups[s]] >= rules.sector_cap:
            continue
        spend = min(slot, book.cash)
        fill_price = open_row[s] * (1 + costs.slippage)
        qty = spend / (fill_price * (1 + costs.buy_rate))
        book.shares[s] = qty
        book.cash -= spend
        cost = spend - qty * open_row[s]
        report.costs += cost
        report.turnover += spend
        report.fills.append(Fill(int(s), "BUY", float(qty), float(open_row[s]), float(fill_price),
                                 float(spend), float(qty * fill_price * costs.buy_rate),
                                 float(qty * open_row[s] * costs.slippage)))
        held += 1
        if capped and groups[s] >= 0:
            count[groups[s]] += 1
    return report


def mark(book: Book, close_ff_row: np.ndarray) -> float:
    """Portfolio value at a close (a stock without a price keeps its last close)."""
    return book.cash + float(np.nansum(book.shares * close_ff_row))


def is_rebalance_day(rules: Rules, sessions: list, t: int, first: int) -> bool:
    """Decide at this session's close? ``month_end``: the last session of a
    calendar month. ``week_end``: the last session of a calendar week (a
    Friday, or the day before a Friday holiday). ``every:K``: every K
    sessions from the first."""
    if t + 1 >= len(sessions):
        return False
    if rules.rebalance == "month_end":
        today, tomorrow = _as_date(sessions[t]), _as_date(sessions[t + 1])
        return (today.year, today.month) != (tomorrow.year, tomorrow.month)
    if rules.rebalance == "week_end":
        today, tomorrow = _as_date(sessions[t]), _as_date(sessions[t + 1])
        return today.isocalendar()[:2] != tomorrow.isocalendar()[:2]
    if rules.rebalance.startswith("every:"):
        # "every:K" or "every:K:O" - every K sessions, starting O sessions late
        # (O measures how much the result depends on the day chosen).
        parts = rules.rebalance.split(":")
        k, offset = int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        return t - first >= offset and (t - first - offset) % k == 0
    raise ValueError(f"unknown rebalance {rules.rebalance!r}")


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromordinal(int(value))


@dataclass
class SimulationResult:
    values: np.ndarray           # portfolio value per session (NaN before the start)
    costs: float
    turnover: float
    corporate_actions: int
    fills: list                  # (session index, Fill)
    rebalances: list             # session indexes where a decision was made
    book: "Book | None" = None   # the book after the last session (experiments: today's holdings)
    pending: "Target | None" = None  # a decision waiting for the next open (only with open_end)
    entry: "np.ndarray | None" = None  # buy price per stock still held (with a stop; adjusted for splits)


def simulate(sessions, opn, close, close_ff, listed, *, rules: Rules, costs: Costs,
             capital: float, start_index: int, score_matrix: np.ndarray | None = None,
             overlay: Overlay | None = None, index_close: np.ndarray | None = None,
             groups: np.ndarray | None = None, low: np.ndarray | None = None,
             no_buy: np.ndarray | None = None, decide_first: bool = False,
             open_end: bool = False) -> SimulationResult:
    """The backtest: the live loop (decide at a close, execute at the next
    open, mark at the close) run over history with the same functions.
    ``overlay``: safety rules tested in Phase 6 (``index_close`` needed for
    the market filter, ``low`` - the day's lows, adjusted like ``opn`` - for
    stops during the day). ``no_buy[t, s]``: stock s is not bought by the
    decision at session t (a holding stays while it ranks) - research only.
    Experiments (strategy/experiments.py, 21 Oct 2026): ``decide_first`` makes
    a decision at the first session (as the live book's first evening);
    ``open_end``: the last session is today - ``sessions`` then has one more
    entry, the next session's date, so a decision (or a stop) at today's close
    is kept in ``pending`` for the next open."""
    if overlay is None and rules.stop_loss:
        overlay = Overlay(stop_loss=rules.stop_loss, intraday=rules.stop_intraday)
    if overlay is not None and overlay != Overlay():
        return _simulate_with_overlay(sessions, opn, close, close_ff, listed, rules=rules, costs=costs,
                                      capital=capital, start_index=start_index, score_matrix=score_matrix,
                                      overlay=overlay, index_close=index_close, groups=groups, low=low,
                                      no_buy=no_buy, decide_first=decide_first, open_end=open_end)
    T, S = close.shape
    score = scores(close_ff, rules.lookback, listed) if score_matrix is None else score_matrix
    book = Book.new(capital, S)
    values = np.full(T, np.nan)
    pending = None
    total_costs = turnover = 0.0
    actions = 0
    fills, rebalances = [], []
    for t in range(start_index, T):
        if t > start_index:
            actions += carry_corporate_action(book, close_ff[t] / close_ff[t - 1])
        if pending is not None:
            report = execute(book, pending, opn[t], close_ff[t - 1], rules, costs, groups)
            pending = None
            total_costs += report.costs
            turnover += report.turnover
            fills.extend((t, f) for f in report.fills)
        values[t] = mark(book, close_ff[t])
        if is_rebalance_day(rules, sessions, t, start_index) or (decide_first and t == start_index):
            if t + 1 < T or open_end:
                pending = _without(decide(score[t], listed[t], close[t], rules), no_buy, t)
                rebalances.append(t)
    return SimulationResult(values, total_costs, turnover, actions, fills, rebalances, book,
                            pending if open_end else None)


COOLDOWN = 21                    # sessions a stock sold by a safety rule is not bought back


def _without(target: Target, no_buy, t: int) -> Target:
    if no_buy is None:
        return target
    return Target(target.keep, tuple(b for b in target.buy_order if not no_buy[t, b]))


def _simulate_with_overlay(sessions, opn, close, close_ff, listed, *, rules, costs, capital, start_index,
                           score_matrix, overlay: Overlay, index_close, groups=None, low=None,
                           no_buy=None, decide_first=False, open_end=False) -> SimulationResult:
    """simulate() plus the safety rules. A holding that breaks a rule at a close
    is sold at the next open (its money waits in cash for the next monthly
    decision) and is not bought again for COOLDOWN sessions (nor while its
    volatility is still too high); with the market filter on, everything is sold while NIFTY is
    below its average and the book buys again on the first close above it."""
    import pandas as pd
    T, S = close.shape
    if overlay.intraday and low is None:
        raise ValueError("a stop during the day needs the day's lows")
    score = scores(close_ff, rules.lookback, listed) if score_matrix is None else score_matrix
    vol_ratio = volatility_ratio(close_ff) if overlay.vol_exit else None
    if overlay.market_filter:
        if index_close is None:
            raise ValueError("the market filter needs index_close")
        average = pd.Series(index_close).rolling(overlay.market_filter, min_periods=overlay.market_filter).mean()
        risk_off_at = np.asarray(index_close < average.to_numpy())
    else:
        risk_off_at = np.zeros(T, dtype=bool)
    book = Book.new(capital, S)
    entry, peak = np.full(S, np.nan), np.full(S, np.nan)
    values = np.full(T, np.nan)
    pending = None
    total_costs = turnover = 0.0
    actions = 0
    fills, rebalances = [], []
    was_off = False
    barred_until = np.full(S, -1)
    for t in range(start_index, T):
        if t > start_index:
            ratio = close_ff[t] / close_ff[t - 1]
            bad = (book.shares > 0) & np.isfinite(ratio) & ~((ratio > JUMP_LOW) & (ratio < JUMP_HIGH))
            entry[bad] *= ratio[bad]
            peak[bad] *= ratio[bad]
            actions += carry_corporate_action(book, ratio)
        if pending is not None:
            report = execute(book, pending, opn[t], close_ff[t - 1], rules, costs, groups)
            pending = None
            total_costs += report.costs
            turnover += report.turnover
            for f in report.fills:
                fills.append((t, f))
                if f.side == "BUY":
                    entry[f.stock], peak[f.stock] = f.fill_price, f.price
                else:
                    entry[f.stock] = peak[f.stock] = np.nan
        if overlay.intraday:          # stops during the day: sold at the stop price the day the low touches it
            with np.errstate(invalid="ignore"):
                level = np.full(S, -np.inf)
                if overlay.stop_loss:
                    level = np.fmax(level, entry * (1 - overlay.stop_loss))
                if overlay.trailing:
                    level = np.fmax(level, peak * (1 - overlay.trailing))
                hit = (book.shares > 0) & np.isfinite(level) & np.isfinite(low[t]) & (low[t] <= level)
            for s in np.flatnonzero(hit):
                px = float(min(opn[t, s], level[s])) if np.isfinite(opn[t, s]) and opn[t, s] > 0 else float(level[s])
                qty = book.shares[s]
                gross = qty * px * (1 - costs.slippage)
                fee = gross * costs.sell_rate + costs.dp_per_sell
                slip = qty * px * costs.slippage
                book.cash += gross - fee
                book.shares[s] = 0.0
                total_costs += fee + slip
                turnover += gross
                fills.append((t, Fill(int(s), "SELL", float(qty), px, px * (1 - costs.slippage), float(gross - fee),
                                      float(fee), float(slip))))
                entry[s] = peak[s] = np.nan
                barred_until[s] = t + COOLDOWN
        values[t] = mark(book, close_ff[t])
        held = book.shares > 0
        peak = np.where(held, np.fmax(peak, close_ff[t]), np.nan)
        out = np.zeros(S, dtype=bool)
        with np.errstate(invalid="ignore"):
            if overlay.stop_loss and not overlay.intraday:
                out |= held & (close_ff[t] <= entry * (1 - overlay.stop_loss))
            if overlay.trailing and not overlay.intraday:
                out |= held & (close_ff[t] <= peak * (1 - overlay.trailing))
            if overlay.vol_exit:
                out |= held & (vol_ratio[t] > overlay.vol_exit)
        exits = {int(i) for i in np.flatnonzero(out)}
        barred_until[out] = t + COOLDOWN
        barred = set(np.flatnonzero(barred_until >= t).tolist())
        if overlay.vol_exit:
            with np.errstate(invalid="ignore"):
                barred |= set(np.flatnonzero(vol_ratio[t] > overlay.vol_exit).tolist())
        off = bool(risk_off_at[t])
        can_act = t + 1 < T or open_end
        if (is_rebalance_day(rules, sessions, t, start_index) or (was_off and not off)
                or (decide_first and t == start_index)):
            if can_act:
                if off:
                    pending = Target(frozenset(), ())
                else:
                    target = _without(decide(score[t], listed[t], close[t], rules), no_buy, t)
                    pending = Target(target.keep - exits,
                                     tuple(b for b in target.buy_order if b not in exits and b not in barred))
                rebalances.append(t)
        elif off and held.any() and can_act:
            pending = Target(frozenset(), ())
        elif exits and can_act:
            pending = Target(frozenset(int(i) for i in np.flatnonzero(held)) - exits, ())
        was_off = off
    return SimulationResult(values, total_costs, turnover, actions, fills, rebalances, book,
                            pending if open_end else None, entry)
