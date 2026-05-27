import logging
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or return a logger configured with a StreamHandler.

    Avoid adding duplicate handlers when called multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not getattr(logger, "__configured", False):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        logger.propagate = False
        logger.__configured = True

    return logger
