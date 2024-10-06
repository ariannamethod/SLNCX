# Grok-1

This repository contains JAX example code for loading and running the **Grok-1** open-weights model, developed by xAI, founded by Elon Musk. Grok-1 is designed to tackle a variety of natural language processing tasks effectively. This document will guide you through the setup, usage, and specifications of the model.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Model Specifications](#model-specifications)
- [Downloading Weights](#downloading-weights)
- [Usage](#usage)
- [License](#license)

## Overview

Grok-1 is an advanced AI model characterized by its large parameter count and a unique architectural approach utilizing a Mixture of Experts (MoE) framework. This model not only serves as a powerful tool for NLP applications but also provides an exciting opportunity for developers and researchers to explore cutting-edge AI technologies.

## Getting Started

To set up and run Grok-1, follow these steps:

1. **Clone the repository:**
   ```shell
   git clone https://github.com/xai-org/grok-1.git
   cd grok-1
   ```

2. **Install required dependencies:**
   ```shell
   pip install -r requirements.txt
   ```

3. **Download the model weights:**  
   Ensure that you download the checkpoint and place the `ckpt-0` directory in `checkpoints` (see [Downloading Weights](#downloading-weights)).

## Model Specifications

Grok-1 is currently designed with the following specifications:

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Maximum Sequence Length (context):** 8,192 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 8-bit quantization

## Downloading Weights

You can download the weights using two methods:

1. **Using a Torrent Client:**  
   Download the weights using the following magnet link:
   ```
   magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
   ```

2. **Directly from Hugging Face Hub:**  
   Clone the repository and use the following commands:
   ```shell
   git clone https://github.com/xai-org/grok-1.git && cd grok-1
   pip install huggingface_hub[hf_transfer]
   huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
   ```

## Usage

To test the code, run the following command:
```shell
python run.py
```
This script loads the checkpoint and samples from the model on a test input. 

**Note:** Due to the large size of the model (314B parameters), a machine with sufficient GPU memory is required to test the model with the example code. The current implementation of the MoE layer may not be fully optimized; it was chosen to facilitate correctness validation without the need for custom kernels.

## License

The code and associated Grok-1 weights in this release are licensed under the Apache 2.0 license. This license only applies to the source files in this repository and the model weights of Grok-1.
