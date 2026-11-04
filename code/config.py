"""Settings. Secrets (KITE_API_KEY, KITE_API_SECRET) live in .env, never here."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path):
    """Minimal .env loader. Never overrides real environment variables."""
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv(BASE_DIR / ".env")


def _secret(path: Path) -> str:
    """A random key kept in data/secret_key (signs the CSRF cookie)."""
    try:
        if path.exists() and len(path.read_text().strip()) >= 32:
            return path.read_text().strip()
        key = os.urandom(32).hex()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key)
        return key
    except OSError:
        return os.urandom(32).hex()


def _flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _old_db() -> Path:
    archived = BASE_DIR / "data" / "archive" / "old_app" / "trader.db"
    return archived if archived.exists() else BASE_DIR / "data" / "trader.db"


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or _secret(BASE_DIR / "data" / "secret_key")
    SESSION_COOKIE_NAME = "swingtrader_session"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # The paper portfolio
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000.0"))
    # Background work: the evening update every 15 minutes (data, then the book)
    BACKGROUND_JOBS_ENABLED = _flag("BACKGROUND_JOBS_ENABLED")
    V2_EVENING_ENABLED = _flag("V2_EVENING_ENABLED")
    LIVE_REFRESH_SECONDS = int(os.getenv("LIVE_REFRESH_SECONDS", "30"))

    # Statutory charges per side, fractions of the traded value (Oct 2026):
    # STT 0.1% (delivery, both sides); stamp duty 0.015% (buy); NSE exchange
    # charge 0.00297%; SEBI fee Rs 10/crore; GST 18% on brokerage + exchange + SEBI.
    CHARGES_STT_RATE = float(os.getenv("CHARGES_STT_RATE", "0.001"))
    CHARGES_STAMP_DUTY_BUY_RATE = float(os.getenv("CHARGES_STAMP_DUTY_BUY_RATE", "0.00015"))
    CHARGES_EXCHANGE_TXN_RATE = float(os.getenv("CHARGES_EXCHANGE_TXN_RATE", "0.0000297"))
    CHARGES_SEBI_FEE_RATE = float(os.getenv("CHARGES_SEBI_FEE_RATE", "0.000001"))
    CHARGES_GST_RATE = float(os.getenv("CHARGES_GST_RATE", "0.18"))
    CHARGES_BROKERAGE_RATE = float(os.getenv("CHARGES_BROKERAGE_RATE", "0.0"))
    DP_CHARGE_PER_SELL_DAY = float(os.getenv("DP_CHARGE_PER_SELL_DAY", "15.34"))  # Zerodha, incl. GST

    # The screen: minimum 20-day median turnover (rupees)
    SCANNER_EOD_MIN_MEDTV_RUPEES = float(os.getenv("SCANNER_EOD_MIN_MEDTV_RUPEES", "50000000"))

    # NSE's daily files
    NSE_EOD_TIMEOUT_S = int(os.getenv("NSE_EOD_TIMEOUT_S", "20"))
    NSE_EOD_DOWNLOAD_DELAY_S = float(os.getenv("NSE_EOD_DOWNLOAD_DELAY_S", "1.0"))

    # Kite: live prices on screen, and the backup for a day's prices
    KITE_API_KEY = os.getenv("KITE_API_KEY", "")
    KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
    KITE_TOKEN_FILE = BASE_DIR / "data" / "kite_tokens.json"

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BASE_DIR / "logs" / "trader.log"

    # The old app's database (archived; read only by scripts/build_v2_db.py)
    OLD_DB_PATH = _old_db()
