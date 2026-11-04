import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config import Config


def setup_logging(app=None):
    logger = logging.getLogger("trader")
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = Path(Config.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        Config.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if app:
        app.logger.handlers = logger.handlers
        app.logger.setLevel(logger.level)

    return logger


def get_logger(name=None):
    return logging.getLogger(f"trader.{name}" if name else "trader")
