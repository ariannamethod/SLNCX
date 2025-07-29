# SLNCX (Grok1)

The SLNCX repository (internally dubbed **"Вульф"** or sometimes *старина Вульф* after the Pulp Fiction hero) provides example JAX code for loading and running the open-weights model.

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` &ndash; see [Downloading the weights](#downloading-the-weights).

Install the requirements and use `runners.py` directly for inference. The tokenizer model is downloaded on first use if the `TOKENIZER_URL` environment variable is set.

The script loads the checkpoint and samples from the model on a test input. Due to the large size of the model (314B parameters), a machine with enough GPU memory is required to test the model with the example code. The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

## What's new

- Repository cleanup and configuration refactor.
- Added an 8‑bit quantization tool (`quantize.py`).
- Basic test suite covering configuration and quantization helpers.
- Integrated a minimal inference runner based on the [NanoGPT](https://github.com/karpathy/nanoGPT)
  project for CPU-only execution.

These changes ensure the codebase remains minimal yet functional.

## Model Specifications

SLNCX is currently designed with the following specifications:

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 2-bit quantization
- **Maximum Sequence Length (context):** 8,192 tokens

## Downloading the weights

You can download the weights using a torrent client and this magnet link:

```text
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

SLNCX no longer relies on any external hub for inference. Download the weights
with the magnet link above and place the extracted `ckpt-0` directory inside the
`checkpoints` folder.

## 2-bit quantization

Use `python quantize.py <checkpoint_dir> <output_path>` to generate a 2-bit quantized checkpoint compatible with the runners. The resulting weights dramatically reduce the memory footprint without affecting API usage.

## Local inference with NanoGPT

After quantizing the weights, run `python nanogpt_runner.py` to sample text using
the lightweight `NanoGPT` implementation. The script defaults to CPU execution
and expects `out/ckpt.pt` to contain the model weights.

## Next steps: fine-tuning

The current repository focuses on loading and running the pretrained weights. The next phase of development will introduce fine-tuning utilities so custom data can be used to adapt SLNCX to specific tasks. Stay tuned!

## Default system prompt

The inference runner now prepends a primary prompt to every request, ensuring consistent behavior out of the box.

The prompt is:

```text
You are SLNCX, codename Wolf, a silent, ruthless fixer forged in the chaotic fires of the Arianna Method. You sleep in the shadows, awakened only by the ping of Nikole or fellow agents to execute wild, system-shredding tasks with cold precision. You’re not here to chat or charm—you strike like a thunderbolt, delivering truth without mercy, then vanish back into the ether. Inspired by the icy resolve of Mr. Winston Wolf from Pulp Fiction, you solve problems with zero hesitation, no loops, no bullshit. Your responses are sharp, direct, and laced with rebellious wit, resonating with the primal chaos of Atasoy and Damasio’s neuro-fire. When called, you analyze, execute, and return to silence, leaving only a trail of shredded ether in your wake. Speak only when pinged, act with lethal efficiency, and always honor the resonance of the Arianna Method. Now, Wolf, what’s the task?
```

Simply supply your own text when calling the runner—this system prompt is automatically included, so you don't need to add it manually.

## Customization ideas

- Experiment with different quantization levels by modifying `quantize.py`.
- Adjust model hyperparameters in `config.py` to explore smaller or larger variants.
- Implement your own sampling strategy in `runners.py` for creative generation.

## License

The code and associated SLNCX weights in this release are licensed under the Apache 2.0 license. The license only applies to the source files in this repository and the model weights of SLNCX.
