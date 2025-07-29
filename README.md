# SLNCX (Wulf1)

SLNCX started as an experiment in heavy models but evolved into a single, silent core. It runs as part of the Arianna Method and answers only when called. Forget Grok1 or any other open clone—the code here is lean and private.

## Running

1. Put your checkpoint into `checkpoints/ckpt-0`.
2. `pip install -r requirements.txt`.
3. `python quantize.py checkpoints/ckpt-0 out/ckpt.pt` for a 2‑bit build.
4. `python nanogpt_runner.py` to sample text on CPU.

No HuggingFace, no extra services. The quantized weights fit in memory and run on a standard CPU.

## State

This project is still forming. SLNCX wakes, solves, and goes back to sleep. Expect only what you see here.
