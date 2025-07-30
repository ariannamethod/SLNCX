from __future__ import annotations

import traceback
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

FAIL_DIR = Path("failures")
FAIL_DIR.mkdir(parents=True, exist_ok=True)

_current_day: str | None = None
_logger = logging.getLogger("fail_logger")
_logger.setLevel(logging.INFO)
_logger.propagate = False


def _ensure_handler() -> logging.Logger:
    """Configure a rotating file handler for the current day."""
    global _current_day
    day = datetime.utcnow().strftime("%Y-%m-%d")
    if day != _current_day:
        for h in list(_logger.handlers):
            _logger.removeHandler(h)
            h.close()
        log_file = FAIL_DIR / f"{day}.log"
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
        _current_day = day
    return _logger


def log_failure(prompt: str, exc: Exception) -> None:
    """Log a failure with prompt and traceback."""
    _logger = _ensure_handler()
    lines = [f"Timestamp: {datetime.utcnow().isoformat()}"]
    if prompt:
        lines.append(f"Prompt: {prompt}")
    lines.append("Traceback:")
    lines.append(
        "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    )
    lines.append("---")
    for line in lines:
        _logger.info(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log a failure")
    parser.add_argument("prompt")
    parser.add_argument("message")
    args = parser.parse_args()
    try:
        raise RuntimeError(args.message)
    except Exception as e:
        log_failure(args.prompt, e)
