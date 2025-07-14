# SLNCX

This repository contains JAX example code for loading and running the SLNCX open-weights model.

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` - see [Downloading the weights](#downloading-the-weights)

Install requirements and use `runners.py` directly for inference. The tokenizer
model is downloaded on first use if the `TOKENIZER_URL` environment variable is
set.

The script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B parameters), a machine with enough GPU memory is required to test the model with the example code.
The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

# Model Specifications

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
  - Supports activation sharding and 8-bit quantization
- **Maximum Sequence Length (context):** 8,192 tokens

# Downloading the weights

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

or directly using [HuggingFace 🤗 Hub](https://huggingface.co/xai-org/slncx):
```
git clone https://github.com/xai-org/slncx.git && cd slncx
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/slncx --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

# License

The code and associated SLNCX weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of SLNCX.

## 8-bit quantization

Use `python quantize.py <checkpoint_dir> <output_path>` to generate a quantized
checkpoint compatible with the runners.
