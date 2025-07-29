from __future__ import annotations

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

LOG_DIR = Path('logs/wulf')


def prune_file(path: Path, threshold: float = 0.1) -> None:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not data:
        path.unlink(missing_ok=True)
        return
    texts = [f"{d['prompt']} {d['response']}" for d in data]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    scores = np.asarray(matrix.mean(axis=1)).ravel()
    kept = [d for d, score in zip(data, scores) if score > threshold]
    if kept:
        path.write_text(
            json.dumps(kept, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    else:
        path.unlink()


def prune_logs() -> None:
    for file in LOG_DIR.glob('*.json'):
        prune_file(file)


if __name__ == "__main__":
    prune_logs()
