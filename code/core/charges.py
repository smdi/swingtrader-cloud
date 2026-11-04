"""Statutory charges on an NSE equity delivery trade (10 Oct 2026 audit).

One model for the live paper book (strategy/live.py) and the backtest
(strategy/strength.py, via strength.Costs.from_config), so both deduct the
same rupees for the same trade. Before, COMMISSION_PCT = 0.1% a
side covered STT only.

Per side, on the traded value (rates in config.py, CHARGES_*):
  * STT 0.1% on buy and on sell (delivery);
  * stamp duty 0.015% on buy only;
  * NSE exchange transaction charge 0.00297%;
  * SEBI turnover fee Rs 10 per crore (0.0001%);
  * brokerage (default 0: delivery is free at discount brokers);
  * GST 18% on brokerage + exchange charge + SEBI fee.
The depository (DP) charge per stock sold per day stays a separate fixed
rupee amount (DP_CHARGE_PER_SELL_DAY), and slippage stays separate.
"""
from __future__ import annotations

from config import Config


def _rate(name: str, default: float) -> float:
    return float(getattr(Config, name, default) or 0.0)


def _side_rate(*, buy: bool) -> float:
    exchange = _rate("CHARGES_EXCHANGE_TXN_RATE", 0.0000297)
    sebi = _rate("CHARGES_SEBI_FEE_RATE", 0.000001)
    brokerage = _rate("CHARGES_BROKERAGE_RATE", 0.0)
    gst = _rate("CHARGES_GST_RATE", 0.18) * (brokerage + exchange + sebi)
    stamp = _rate("CHARGES_STAMP_DUTY_BUY_RATE", 0.00015) if buy else 0.0
    return _rate("CHARGES_STT_RATE", 0.001) + stamp + exchange + sebi + brokerage + gst


def buy_charge_rate() -> float:
    """Charges as a fraction of the buy value."""
    return _side_rate(buy=True)


def sell_charge_rate() -> float:
    """Charges as a fraction of the sell value (DP charge excluded)."""
    return _side_rate(buy=False)


def buy_charges(value: float) -> float:
    """Rupee charges on buying ``value`` rupees of shares for delivery."""
    return float(value) * buy_charge_rate()


def sell_charges(value: float) -> float:
    """Rupee charges on selling ``value`` rupees of delivered shares
    (the per-stock DP charge is added by the caller)."""
    return float(value) * sell_charge_rate()


__all__ = ["buy_charge_rate", "buy_charges", "sell_charge_rate", "sell_charges"]
