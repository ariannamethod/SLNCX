import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

from nanogpt_model import GPT, GPTConfig

DATA_DIRS = [Path('scripts'), Path('logs'), Path('mem'), Path('datasets')]
TRAIN_CORPUS = Path('datasets/wulf_corpus.txt')
MODEL_OUT = Path('out/wulf_pretrained.pt')


def collect_texts() -> list[str]:
    exts = {'.py', '.json', '.jsonl', '.txt', '.md', '.sh'}
    texts = []
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for path in d.rglob('*'):
            if path.suffix.lower() in exts and path.is_file():
                try:
                    texts.append(path.read_text(encoding='utf-8'))
                except Exception:
                    continue
    return texts


def dedupe(texts: list[str]) -> list[str]:
    seen = {}
    for t in texts:
        h = hashlib.sha1(t.encode('utf-8')).hexdigest()
        if h not in seen:
            seen[h] = t
    return list(seen.values())


def filter_dense(texts: list[str], threshold: float = 0.05) -> list[str]:
    if not texts:
        return []
    vec = TfidfVectorizer().fit_transform(texts)
    scores = np.asarray(vec.mean(axis=1)).ravel()
    return [t for t, s in zip(texts, scores) if s > threshold]


def build_corpus() -> str:
    texts = collect_texts()
    texts = dedupe(texts)
    texts = filter_dense(texts)
    corpus = '\n'.join(texts)
    TRAIN_CORPUS.write_text(corpus, encoding='utf-8')
    return corpus


def simple_pretrain(text: str, steps: int = 50) -> None:
    enc = tiktoken.get_encoding('gpt2')
    tokens = torch.tensor(enc.encode(text), dtype=torch.long)
    block_size = 64
    cfg = GPTConfig(block_size=block_size, n_layer=2, n_head=2, n_embd=128)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        if len(tokens) <= block_size + 1:
            break
        i = torch.randint(0, len(tokens) - block_size - 1, (1,)).item()
        x = tokens[i:i+block_size][None, ...]
        y = tokens[i+1:i+block_size+1][None, ...]
        logits, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.save({'model_args': cfg.__dict__, 'model': model.state_dict()}, MODEL_OUT)


def main() -> None:
    corpus = build_corpus()
    if corpus:
        simple_pretrain(corpus)


if __name__ == '__main__':
    main()
