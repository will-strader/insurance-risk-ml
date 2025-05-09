import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_LEVEL = logging.INFO

def get_logger(name: str) -> logging.Logger:
    """Return a logger with the project’s standard format."""
    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
    return logging.getLogger(name)
