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

## Logging and Memory

Session logs are written to `logs/wulf/` as JSON files named by day. Each entry
contains the user prompt, Wulf's reply and a timestamp. Failures and tracebacks
are appended to files in `failures/`.

The `scripts` directory provides maintenance utilities:

- `session_logger.py` – append a prompt/response pair to the current log.
- `fail_log.py` – record a failure with traceback.
- `entropy_prune.py` – remove low-entropy sessions from the logs.
- `memory_vector.py` – build `mem/wulf_vector.json` from past responses.
- `daily_routine.sh` – run pruning and memory updates (for cron).

Install dependencies with `pip install -r requirements.txt` which now includes
`scikit-learn` for the vector utilities.
