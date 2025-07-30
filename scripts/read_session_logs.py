from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path('logs/wulf')


def read_logs(day: str | None = None):
    """Yield log entries for the given day (default: today)."""
    day = day or datetime.utcnow().strftime('%Y-%m-%d')
    log_file = LOG_DIR / f"{day}.jsonl"
    if not log_file.exists():
        return
    with log_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print session log entries")
    parser.add_argument("--day", help="YYYY-MM-DD of the log to read")
    args = parser.parse_args()
    for entry in read_logs(args.day):
        print(json.dumps(entry, ensure_ascii=False))
