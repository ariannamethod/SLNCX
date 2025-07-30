import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('datasets')
LOG_DIR = Path('logs/wulf')
INDEX_FILE = LOG_DIR / 'dataset_index.json'

LOG_DIR.mkdir(parents=True, exist_ok=True)


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_index() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding='utf-8'))
    return {}


def save_index(index: dict) -> None:
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')


def watch() -> None:
    index = load_index()
    updated = False
    for file in DATA_DIR.iterdir():
        if not file.is_file():
            continue
        h = file_hash(file)
        if index.get(file.name) != h:
            shutil.copy2(file, LOG_DIR / file.name)
            (LOG_DIR / f'{file.stem}.pending').write_text(datetime.utcnow().isoformat(), encoding='utf-8')
            index[file.name] = h
            updated = True
    if updated:
        save_index(index)


if __name__ == '__main__':
    watch()
