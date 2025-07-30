from pathlib import Path

import torch
import tiktoken

from nanogpt_model import GPTConfig, GPT
from scripts.session_logger import log_session
from scripts.fail_log import log_failure


CKPT_PATH = Path("out/ckpt.pt")
DEVICE = "cpu"


def _load_model() -> GPT:
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model


_model = None
_tokenizer = tiktoken.get_encoding("gpt2")


def generate(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 200) -> str:
    """Run a single inference step with the model on CPU."""
    global _model
    if _model is None:
        _model = _load_model()
    start_ids = _tokenizer.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...]
    with torch.no_grad():
        y = _model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    out = _tokenizer.decode(y[0].tolist())
    return out


def infer_and_log(prompt: str, **kwargs) -> str:
    try:
        response = generate(prompt, **kwargs)
        log_session(prompt, response)
        return response
    except Exception as e:  # noqa: BLE001
        log_failure(prompt, e)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a prompt through Wulf")
    parser.add_argument("prompt")
    parser.add_argument("--tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    print(infer_and_log(args.prompt, max_new_tokens=args.tokens, temperature=args.temperature))
