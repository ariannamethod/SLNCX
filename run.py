# Copyright 2024 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model


CKPT_PATH = "./checkpoints/"


def create_grok_1_model() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )


def create_inference_runner(model: LanguageModelConfig, checkpoint_path: str, tokenizer_path: str) -> InferenceRunner:
    return InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=model,
            bs_per_device=0.125,
            checkpoint_path=checkpoint_path,
        ),
        name="local",
        load=checkpoint_path,
        tokenizer_path=tokenizer_path,
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )


def generate_text(inference_runner: InferenceRunner, prompt: str, max_len: int = 100, temperature: float = 0.01) -> str:
    gen = inference_runner.run()
    return sample_from_model(gen, prompt, max_len=max_len, temperature=temperature)


def main():
    grok_1_model = create_grok_1_model()
    inference_runner = create_inference_runner(grok_1_model, CKPT_PATH, "./tokenizer.model")
    inference_runner.initialize()

    inp = "The answer to life the universe and everything is of course"
    output = generate_text(inference_runner, inp)
    print(f"Output for prompt: {inp}\n{output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
