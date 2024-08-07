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
import os
from cryptography.fernet import Fernet
from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model

# Secure Key Management
KEY_ENV_VAR = 'ENCRYPTION_KEY'
KEY = os.getenv(KEY_ENV_VAR)
if not KEY:
    raise ValueError(f"Encryption key must be set in the environment variable {KEY_ENV_VAR}")
cipher_suite = Fernet(KEY)

# Define paths
CKPT_PATH = os.getenv('CHECKPOINT_PATH', './checkpoints/')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', './tokenizer.model')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_model() -> LanguageModelConfig:
    """Initialize and return the language model configuration."""
    try:
        model_config = LanguageModelConfig(
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
                num_experts=8,
                num_selected_experts=2,
                data_axis="data",
                model_axis="model",
            ),
        )
        logging.info("Model initialized successfully.")
        return model_config
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        raise

def initialize_inference_runner(model: LanguageModelConfig) -> InferenceRunner:
    """Initialize and return the inference runner."""
    try:
        inference_runner = InferenceRunner(
            pad_sizes=(1024,),
            runner=ModelRunner(
                model=model,
                bs_per_device=0.125,
                checkpoint_path=CKPT_PATH,
            ),
            name="local",
            load=CKPT_PATH,
            tokenizer_path=TOKENIZER_PATH,
            local_mesh_config=(1, 8),
            between_hosts_config=(1, 1),
        )
        inference_runner.initialize()
        logging.info("Inference runner initialized successfully.")
        return inference_runner
    except Exception as e:
        logging.error(f"Error initializing inference runner: {e}")
        raise

def encrypt_message(message: str) -> str:
    """Encrypt the message using Fernet encryption."""
    try:
        encrypted_message = cipher_suite.encrypt(message.encode())
        return encrypted_message.decode()
    except Exception as e:
        logging.error(f"Error encrypting message: {e}")
        raise

def decrypt_message(encrypted_message: str) -> str:
    """Decrypt the message using Fernet encryption."""
    try:
        decrypted_message = cipher_suite.decrypt(encrypted_message.encode())
        return decrypted_message.decode()
    except Exception as e:
        logging.error(f"Error decrypting message: {e}")
        raise

def generate_text(prompt: str, runner: InferenceRunner) -> str:
    """Generate text from the given prompt using the inference runner."""
    try:
        logging.info("Running inference...")
        gen = runner.run()
        return sample_from_model(gen, prompt, max_len=100, temperature=0.01)
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise

def main():
    try:
        logging.info("Initializing model...")
        model = initialize_model()
        
        logging.info("Setting up inference runner...")
        inference_runner = initialize_inference_runner(model)
        
        prompt = "The answer to life the universe and everything is of course"
        logging.info("Generating output...")
        output = generate_text(prompt, inference_runner)
        
        encrypted_output = encrypt_message(output)
        decrypted_output = decrypt_message(encrypted_output)
        
        logging.info(f"Output for prompt: {prompt}")
        print(decrypted_output)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

