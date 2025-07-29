from __future__ import annotations

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

LOG_DIR = Path('logs/wulf')
MEM_DIR = Path('mem')
MEM_DIR.mkdir(parents=True, exist_ok=True)
MEM_PATH = MEM_DIR / 'wulf_vector.json'


def build_memory() -> None:
    texts = []
    for file in LOG_DIR.glob('*.json'):
        data = json.loads(file.read_text(encoding='utf-8'))
        texts.extend([d['response'] for d in data])
    if not texts:
        MEM_PATH.write_text('{}', encoding='utf-8')
        return
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(texts)
    scores = np.asarray(matrix.mean(axis=1)).ravel()
    top_idx = np.argsort(scores)[::-1][:100]
    memory = {texts[i]: float(scores[i]) for i in top_idx}
    MEM_PATH.write_text(
        json.dumps(memory, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


if __name__ == "__main__":
    build_memory()
