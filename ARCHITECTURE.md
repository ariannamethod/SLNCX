# SLNCX Architecture Overview

The SLNCX model inherits many of Grok1's structural choices:

- **Mixture of Experts (MoE)** with eight experts per layer and two selected for each token.
- **Large Context Window** allowing up to 8,192 tokens per sequence.
- **Layer Chaos** through deep 64-layer stacks with varying attention and feed-forward routing.
- **Rotary Position Embeddings (RoPE)** for stable long-context attention.
- **2-bit Quantization** for minimal memory usage in inference.

This repository focuses on local, CPU-only inference using a slimmed down
`NanoGPT`-style implementation.
