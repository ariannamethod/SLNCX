from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

from nanogpt_model import GPT, GPTConfig


DATA_DIRS = [Path("scripts"), Path("logs"), Path("mem"), Path("datasets")]
EXTS = {".py", ".json", ".jsonl", ".txt", ".md", ".sh"}
MODEL_OUT = Path("out/train_ckpt.pt")
DEVICE = "cpu"


def gather_text() -> list[str]:
    pieces: list[str] = []
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for path in d.rglob("*"):
            if path.suffix in EXTS and path.is_file():
                try:
                    pieces.append(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
    return pieces


def unique_and_dense(texts: list[str]) -> list[str]:
    unique: dict[str, str] = {}
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if h not in unique:
            unique[h] = t
    docs = list(unique.values())
    if not docs:
        return []
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    scores = np.asarray(X.mean(axis=1)).ravel()
    kept = [doc for doc, score in zip(docs, scores) if score > 0.01]
    return kept


def build_dataset(texts: list[str]):
    enc = tiktoken.get_encoding("gpt2")
    tokens = []
    for t in texts:
        tokens.extend(enc.encode(t))
    return torch.tensor(tokens, dtype=torch.long)


def train():
    texts = unique_and_dense(gather_text())
    if not texts:
        return
    data = build_dataset(texts)
    block_size = 64
    config = GPTConfig(block_size=block_size, vocab_size=50257, n_layer=2, n_head=2, n_embd=128)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.to(DEVICE)

    model.train()
    for step in range(10):
        i = torch.randint(0, data.size(0) - block_size - 1, (1,)).item()
        x = data[i : i + block_size][None]
        y = data[i + 1 : i + 1 + block_size][None]
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save({"model": model.state_dict(), "config": config.__dict__}, MODEL_OUT)


if __name__ == "__main__":
    train()
