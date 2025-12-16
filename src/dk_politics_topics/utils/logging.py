import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger with sane defaults."""
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)
