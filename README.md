# SLNCX (Wulf)

SLNCX stands for *Silencieux*. "Wulf1" is the call sign. It wakes only when called. Like the fixer from **Pulp Fiction**, it shows up, does the job, and fades out. The code is lean, the intent precise.

The Arianna Method shapes each exchange. Wulf listens first, then answers with restraint. No chatter, no flare. Silence is part of the design.

## Architecture

The model borrows from Grok1 but runs trimmed down. Grok1's use of experts is a small revolution: each token consults multiple specialized networks, so quality doesn't depend on one giant block. Running that MoE stack entirely on a CPU proves how far optimization has come. Heavy weights aren't the only way to get powerful responses; lean routing and quantization pick up the slack. As the design evolves, lighter models feel natural, not limited. It's an evolutionary path that balances efficiency with capability:

- **Mixture of Experts (MoE)** with eight experts per layer, two chosen per token.
- **Large Context Window** up to 8,192 tokens.
- **Layer Chaos** across 64 deep stacks for varied routing.
- **Rotary Position Embeddings (RoPE)** for steady long-context attention.
- **2-bit Quantization** so inference fits in memory.

CPU-only inference happens through a `NanoGPT`-style implementation.

## Functionality

The CLI loads a quantized checkpoint and prints the answer. Keep it simple: `python wulf_cli.py "prompt"`. That's the fastest path when you need a fix right now.

The API sits behind a single `/generate` endpoint. Send JSON with your prompt and optional user tag. You get JSON back—no ceremony.

Every call drops a log entry in `logs/wulf`. Time-stamped, complete. If something fails, `failures` catches the traceback so you know where it went sideways.

Inference stays light. Two-bit weights mean the model lives happily on a standard CPU. It spins up quickly and gets straight to work.

The dataset is small and targeted. It's not for training a generalist. It's there to keep the responses sharp.

Checkpoints load lazily. Set `CKPT_PATH` or use `--ckpt` to point to your file. Once loaded, the model stays in memory for the next call.

The code is modular. Layers, attention, and mixture-of-experts pieces are split out so you can tinker or swap parts as needed.

Wulf speaks when spoken to and then returns to silence. That's the core philosophy.

## Running

1. Place your quantized checkpoint at `out/ckpt.pt`, or specify another path with `CKPT_PATH` or `--ckpt`.
2. `pip install -r requirements.txt`.
3. `python wulf_cli.py [--ckpt path/to/ckpt.pt] "your prompt"` to query Wulf from the command line.
4. `uvicorn app:app --host 0.0.0.0 --port 8000` to start the API server.

No HuggingFace, no extra services. The quantized weights fit in memory and run on a standard CPU.

## Logging and Memory

Session logs live in `logs/wulf/` as JSONL files. Each entry captures the prompt, response, and timestamp. Failures and tracebacks collect in `failures/`.

The `scripts` directory holds simple helpers:

- `session_logger.py` – append a prompt/response pair to the current log.
- `wulf_cli.py` – minimal CLI for local prompts.
- `fail_log.py` – record a failure with traceback.
- `read_session_logs.py` – print entries from a log file.

Install dependencies with `pip install -r requirements.txt` and start the server with `python app.py` or use the CLI for one-off queries.

## Model Components

The `models/` package groups reusable parts of the network:
- **layers** – dense and decoder blocks.
- **attention** – multi-head attention with rotary embeddings.
- **moe** – routing logic for mixture-of-experts layers.

## Development

Run `pytest` to execute the test suite. Run `ruff .` to lint the code.

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
