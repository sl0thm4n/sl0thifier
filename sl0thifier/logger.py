import logging
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)

logger: logging.Logger = logging.getLogger("sl0thifier")
logger.setLevel(LOG_LEVEL)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("🦥 %(message)s"))
    logger.addHandler(handler)
