import argparse
from pathlib import Path

import torch
import tiktoken

from nanogpt_model import GPTConfig, GPT
from scripts.session_logger import log_session
from scripts.fail_log import log_failure

CKPT_PATH = Path('out/ckpt.pt')


def load_model() -> GPT:
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate(prompt: str, max_new_tokens: int = 100) -> str:
    model = load_model()
    enc = tiktoken.get_encoding('gpt2')
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long)[None, ...]
    with torch.no_grad():
        y = model.generate(idx, max_new_tokens, temperature=0.8, top_k=200)
    return enc.decode(y[0].tolist())


def main(prompt: str, user: str | None = None) -> None:
    try:
        response = generate(prompt)
        log_session(prompt, response, user=user)
        print(response)
    except Exception as e:
        log_failure(prompt, e)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Wulf inference')
    parser.add_argument('prompt', help='prompt text')
    parser.add_argument('--user')
    args = parser.parse_args()
    main(args.prompt, user=args.user)
