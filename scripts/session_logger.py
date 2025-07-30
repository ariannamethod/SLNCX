from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs/wulf")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_current_day: str | None = None
_logger = logging.getLogger("session_logger")
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
        log_file = LOG_DIR / f"{day}.jsonl"
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


def log_session(prompt: str, response: str, user: str | None = None) -> None:
    """Append a single session entry to today's log."""
    day = datetime.utcnow().strftime("%Y-%m-%d")
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": response,
    }
    if user:
        entry["user"] = user
    logger = _ensure_handler()
    logger.info(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log a single Wulf session")
    parser.add_argument("prompt")
    parser.add_argument("response")
    parser.add_argument("--user")
    args = parser.parse_args()
    log_session(args.prompt, args.response, user=args.user)
