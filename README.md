# SLNCX (Wulf1)

SLNCX started as an experiment in heavy models but evolved into a single, silent core. It runs as part of the Arianna Method and answers only when called. Forget Grok1 or any other open clone—the code here is lean and private.

## Running

1. Place your quantized checkpoint at `out/ckpt.pt`, or specify another path
   with the `CKPT_PATH` environment variable or the `--ckpt` option. Use
   `python quantize.py <checkpoint_dir> out/ckpt.pt` if you need to convert a
   raw checkpoint.
2. `pip install -r requirements.txt`.
3. `python wulf_cli.py [--ckpt path/to/ckpt.pt] "your prompt"` to query Wulf
   from the command line.
4. `uvicorn app:app --host 0.0.0.0 --port 8000` to start the API server.

No HuggingFace, no extra services. The quantized weights fit in memory and run on a standard CPU.

## State

This project is still forming. SLNCX wakes, solves, and goes back to sleep. Expect only what you see here.

## Logging and Memory

Session logs are written to `logs/wulf/` as JSON files named by day. Each entry
contains the user prompt, Wulf's reply and a timestamp. Failures and tracebacks
are appended to files in `failures/`.

The `scripts` directory contains simple helpers:

- `session_logger.py` – append a prompt/response pair to the current log.
- `wulf_cli.py` – minimal CLI for local prompts.
- `fail_log.py` – record a failure with traceback.

Install dependencies with `pip install -r requirements.txt` and start the
server with `python app.py` or use the CLI for one-off queries.

## Deployment on Railway

1. Create a new Railway project and point it at this repository.
2. Set the start command to `python app.py`.
3. Upload your `out/ckpt.pt` file as a deployment asset or volume.
4. Deploy and query the `/generate` endpoint with a JSON body:

```json
{
  "user": "alice",
  "prompt": "Hello"
}
```
