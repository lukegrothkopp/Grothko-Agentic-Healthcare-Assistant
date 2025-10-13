import logging
import os

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger("gha")
_logger.setLevel(_LEVEL)
if not _logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(_formatter)
    _logger.addHandler(ch)


def get_logger(name: str):
    logger = _logger.getChild(name)
    logger.setLevel(_LEVEL)
    return logger
