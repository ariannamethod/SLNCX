from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

DATASET_DIR = Path("datasets")
LOG_DIR = Path("logs/wulf")
PENDING_DIR = Path("mem/pending_processing")
HASH_FILE = DATASET_DIR / ".hashes.json"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PENDING_DIR.mkdir(parents=True, exist_ok=True)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hashes() -> dict[str, str]:
    if HASH_FILE.exists():
        return json.loads(HASH_FILE.read_text())
    return {}


def save_hashes(hashes: dict[str, str]) -> None:
    HASH_FILE.write_text(json.dumps(hashes, indent=2))


def scan() -> None:
    hashes = load_hashes()
    updated = False
    for file in DATASET_DIR.iterdir():
        if file.is_file():
            h = file_hash(file)
            if hashes.get(file.name) != h:
                shutil.copy2(file, LOG_DIR / file.name)
                (PENDING_DIR / file.name).touch()
                hashes[file.name] = h
                updated = True
    if updated:
        save_hashes(hashes)


if __name__ == "__main__":
    scan()
