# SLNCX (Wulf1)

SLNCX started as an experiment in heavy models but evolved into a single, silent core. It runs as part of the Arianna Method and answers only when called. Forget Grok1 or any other open clone—the code here is lean and private.

## Running

1. Put your checkpoint into `checkpoints/ckpt-0`.
2. `pip install -r requirements.txt`.
3. `python quantize.py checkpoints/ckpt-0 out/ckpt.pt` for a 2‑bit build.
4. `python wulf_cli.py "your prompt"` for a single reply.

No HuggingFace, no extra services. The quantized weights fit in memory and run on a standard CPU.

## Tools

- `wulf_cli.py` – minimal CLI wrapper around `wulf_inference.py`.
- `watch_datasets.py` – copy new files from `datasets/` into the log folder and mark them for processing.
- `wulf_train.py` – lightweight NanoGPT pre-training on scripts, logs, memory and datasets.
- `scripts/daily_routine.sh` – run pruning, memory updates and training (for cron).

## Logging and Memory

Session logs are written to `logs/wulf/` as JSON files named by day. Failures and tracebacks are appended to files in `failures/`.

The `scripts` directory provides maintenance utilities:

- `session_logger.py` – append a prompt/response pair to the current log.
- `fail_log.py` – record a failure with traceback.
- `entropy_prune.py` – remove low-entropy sessions from the logs.
- `memory_vector.py` – build `mem/wulf_vector.json` from past responses.

Install dependencies with `pip install -r requirements.txt` which now includes `scikit-learn` for the vector utilities.
