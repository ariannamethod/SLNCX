from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path

FAIL_DIR = Path('failures')
FAIL_DIR.mkdir(parents=True, exist_ok=True)


def log_failure(prompt: str, exc: Exception) -> None:
    """Log a failure with prompt and traceback."""
    day = datetime.utcnow().strftime('%Y-%m-%d')
    log_file = FAIL_DIR / f"{day}.log"
    with log_file.open('a', encoding='utf-8') as f:
        f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
        if prompt:
            f.write(f"Prompt: {prompt}\n")
        f.write("Traceback:\n")
        f.write(
            ''.join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        )
        f.write("\n---\n")


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
