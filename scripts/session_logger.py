from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path('logs/wulf')
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_session(prompt: str, response: str, user: str | None = None) -> None:
    """Append a single session entry to today's log."""
    day = datetime.utcnow().strftime('%Y-%m-%d')
    log_file = LOG_DIR / f"{day}.jsonl"
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": response,
    }
    if user:
        entry["user"] = user
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log a single Wulf session")
    parser.add_argument("prompt")
    parser.add_argument("response")
    parser.add_argument("--user")
    args = parser.parse_args()
    log_session(args.prompt, args.response, user=args.user)
