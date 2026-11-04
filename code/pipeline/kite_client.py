"""Kite Connect client with persisted tokens and automatic session renewal.

Used for the daily universe snapshot, live quotes, and primary historical OHLCV
fetches. yfinance is retained as a per-symbol fallback when Kite is unavailable.

Auth flow:
  1. Run scripts/kite_login.py once (opens login URL, exchanges the
     request_token for access + refresh tokens, saves to
     data/kite_tokens.json).
  2. Access tokens expire after ~1 day; the session-expiry hook renews them
     silently with renew_access_token() using the stored refresh token.
"""
import json
import os
import threading
import time

from kiteconnect import KiteConnect

from config import Config
from core.logging_config import get_logger

logger = get_logger("kite_client")

_token_lock = threading.Lock()


def is_configured() -> bool:
    return bool(Config.KITE_API_KEY)


def load_tokens() -> dict:
    try:
        if Config.KITE_TOKEN_FILE and Config.KITE_TOKEN_FILE.exists():
            return json.loads(Config.KITE_TOKEN_FILE.read_text(encoding="utf-8"))
    except OSError as e:
        logger.error(f"Could not read Kite tokens from {Config.KITE_TOKEN_FILE}: {e}")
    return {}


def save_tokens(access_token: str, refresh_token: str, api_key: str = ""):
    tokens = load_tokens()
    tokens["access_token"] = access_token
    tokens["refresh_token"] = refresh_token
    tokens["api_key"] = api_key or Config.KITE_API_KEY
    tokens["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        Config.KITE_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        Config.KITE_TOKEN_FILE.write_text(
            json.dumps(tokens, indent=2), encoding="utf-8"
        )
        logger.info(f"Kite tokens saved to {Config.KITE_TOKEN_FILE}")
    except OSError as e:
        logger.error(f"Could not save Kite tokens: {e}")


def get_client() -> KiteConnect:
    """Build a KiteConnect instance with tokens loaded and auto-renew wired."""
    if not is_configured():
        raise RuntimeError("KITE_API_KEY not set — add it to .env or environment")

    tokens = load_tokens()
    if tokens.get("api_key") and tokens["api_key"] != Config.KITE_API_KEY:
        raise RuntimeError(
            "Saved tokens belong to a different API key — re-run scripts/kite_login.py"
        )

    kite = KiteConnect(api_key=Config.KITE_API_KEY)
    access = tokens.get("access_token") or os.getenv("KITE_ACCESS_TOKEN", "")
    if access:
        kite.set_access_token(access)

    def _renew():
        """Renew the access token from the stored refresh token and persist both."""
        with _token_lock:
            fresh = load_tokens()
            refresh = fresh.get("refresh_token")
            if not refresh:
                raise RuntimeError(
                    "Kite session expired and no refresh token stored — "
                    "re-run scripts/kite_login.py"
                )
            data = kite.renew_access_token(refresh, Config.KITE_API_SECRET)
            kite.set_access_token(data["access_token"])
            save_tokens(
                data["access_token"], data.get("refresh_token", refresh),
                Config.KITE_API_KEY,
            )
            logger.info("Kite access token renewed automatically")

    kite.set_session_expiry_hook(_renew)
    return kite
