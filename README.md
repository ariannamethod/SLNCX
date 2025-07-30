# Wulf1

Wulf1 is a lean local language model. It runs entirely offline and wakes only when called.

## Local setup

1. Place your quantized checkpoint at `out/ckpt.pt`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the API with `python app.py`.
4. POST to `http://localhost:8000/generate` with JSON:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"user": "alice", "prompt": "Hello"}'
```

You can also test quickly with `python wulf_cli.py "your prompt"`.

### Maintenance tools

- `watch_datasets.py` – copy new datasets into `logs/wulf/`.
- `wulf_train.py` – run a lightweight self‑training pass.
- `entropy_prune.py` and `memory_vector.py` help clean logs and update memory.

## Deploying on Railway

1. Create a Railway project from this repo.
2. Set the service command to `python app.py`.
3. Upload your `out/ckpt.pt` checkpoint to the project.
4. Deploy. Railway will expose the API on its assigned port.

Logs are stored in `logs/wulf/` and failures in `failures/`.
