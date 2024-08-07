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


import bisect
import functools
import logging
import math
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import sentencepiece
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
from model import (
    LanguageModelConfig,
    LanguageModelOutput,
    TrainingState,
    apply_rules,
    Memory,
    KVMemory,
)
import checkpoint as xai_checkpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
rank_logger = logging.getLogger("rank")
rank_logger.setLevel(logging.INFO)

TOP_K = 8

class SampleSettings(NamedTuple):
    temperature: jax.Array
    nucleus_p: jax.Array
    mask: jax.Array
    active: jax.Array

class SampleOutput(NamedTuple):
    token_id: jax.Array
    prob: jax.Array
    top_k_token_ids: jax.Array
    top_k_probs: jax.Array

def insert_slice(memory: Memory, slice: Memory, length: int, i: int) -> Memory:
    slice = Memory(
        layers=[
            KVMemory(layer.k, layer.v, step=jnp.array([length]))
            for layer in slice.layers
        ],
    )
    return jax.tree_map(lambda m, u: jax.lax.dynamic_update_index_in_dim(m, u[0], i, axis=0),
                        memory, slice)

def pad_to_size(x: jnp.ndarray, size: int) -> jnp.ndarray:
    if x.shape[0] > size:
        x = x[-size:]
    return np.pad(x, [0, size - x.shape[0]], mode="constant", constant_values=0)

def top_p_filter(logits: jax.Array, top_p: jax.Array) -> jax.Array:
    assert logits.ndim == top_p.ndim, f"Expected {logits.ndim} equal {top_p.ndim}"
    sorted_logits = jax.lax.sort(logits, is_stable=False)
    sorted_probs = jax.nn.softmax(sorted_logits)
    threshold_idx = jnp.argmax(jnp.cumsum(sorted_probs, -1) >= 1 - top_p, axis=-1)
    threshold_largest_logits = jnp.take_along_axis(
        sorted_logits, threshold_idx[..., jnp.newaxis], axis=-1
    )
    assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
    mask = logits >= threshold_largest_logits
    logits = jnp.where(mask, logits, -1e10)
    return logits

def sample_token(
    rngs: jax.random.PRNGKey,
    lm_outputs: LanguageModelOutput,
    settings: SampleSettings,
) -> SampleOutput:
    settings = SampleSettings(
        temperature=jnp.expand_dims(settings.temperature, (1, 2)),
        nucleus_p=jnp.expand_dims(settings.nucleus_p, (1, 2)),
        mask=jnp.expand_dims(settings.mask, 1),
        active=settings.active,
    )
    logits = lm_outputs.logits / settings.temperature.astype(lm_outputs.logits.dtype)
    logits = jnp.where(settings.mask, logits, -1e10)
    logits = top_p_filter(logits, settings.nucleus_p.astype(logits.dtype))
    new_token = jax.vmap(jax.random.categorical)(rngs, logits)
    probabilities = jax.nn.softmax(logits)
    token_prob = jnp.take_along_axis(probabilities, jnp.expand_dims(new_token, 1), axis=2)
    token_prob = jnp.squeeze(token_prob, 1)
    top_k_probs, top_k_token_ids = jax.lax.top_k(probabilities, TOP_K)
    top_k_probs = jnp.squeeze(top_k_probs, 1)
    top_k_token_ids = jnp.squeeze(top_k_token_ids, 1)
    return SampleOutput(
        new_token,
        token_prob,
        top_k_token_ids,
        top_k_probs,
    )

@dataclass
class ModelRunner:
    model: LanguageModelConfig
    bs_per_device: float = 2.0
    load_rename_rules: Optional[list[tuple[str, str]]] = None
    load_exclude_rules: Optional[list[str]] = None
    rng_seed: int = 42
    transform_forward: bool = False
    checkpoint_path: str = ""

    def make_forward_fn(self, mesh: Any):
        def forward(tokens):
            out = self.model.make(mesh=mesh)(tokens)
            return out, None

        if self.transform_forward:
            forward = hk.transform(forward)
        return forward

    def initialize(
        self,
        init_data,
        local_mesh_config: Tuple[int, int],
        between_hosts_config: Tuple[int, int],
    ):
        num_replicas = math.prod(between_hosts_config)
        self.model.initialize()
        self.model.fprop_dtype = jnp.bfloat16
        num_local_gpus = len(jax.local_devices())
        self.batch_size = int(self.bs_per_device * num_local_gpus * num_replicas)
        self.local_batch_size = self.batch_size // jax.process_count()
        self.local_mesh_config = local_mesh_config
        self.between_hosts_config = between_hosts_config
        rank_logger.info(
            f"Initializing mesh for {self.local_mesh_config=} {self.between_hosts_config=}..."
        )
        self.mesh = make_mesh(self.local_mesh_config, self.between_hosts_config)
        self.forward = self.make_forward_fn(mesh=self.mesh)
        self.logits_fn = hk.transform(lambda tokens: self.forward(tokens)[0])
        self.eval_forward = self.make_forward_fn(mesh=self.mesh)
        self.logits_eval_fn = hk.transform(lambda tokens: self.eval_forward(tokens)[0])

        if self.transform_forward:
            self.state_sharding = self.get_state_sharding(init_data)
            rank_logger.info(f"State sharding type: {type(self.state_sharding)}")
            self.init_fn = pjit.pjit(self.init, out_shardings=self.state_sharding)

    def init(self, rng: jax.Array, data) -> TrainingState:
        assert self.transform_forward
        rng, init_rng = jax.random.split(rng)
        params = self.forward.init(init_rng, data["inputs"])
        return TrainingState(params=params)

    def get_state_sharding(self, init_data):
        assert self.transform_forward
        rng = jax.random.PRNGKey(self.rng_seed)
        rank_logger.info(f"partition rules: {self.model.partition_rules}")

        with self.mesh:
            shapes = jax.eval_shape(self.init, rng, init_data)
            sharding = jax.tree_util.tree_map_with_path(
                apply_rules(self.model.partition_rules()),
                shapes,
            )
        return sharding

    def load_or_init(
        self,
        init_data: Any,
        from_checkpoint: bool = True,
        init_fn: Optional[Callable] = None,
    ):
        rng = jax.random.PRNGKey(self.rng_seed)

        if not self.checkpoint_path or not from_checkpoint:
            rank_logger.info("Initializing model...")
            with self.mesh:
                if init_fn is not None:
                    state = init_fn(rng, init_data)
                else:
                    assert self.transform_forward
                    state = self.init_fn(rng, init_data)
            rank_logger.info("Model state is newly initialized.")
        else:
            with self.mesh:
                if init_fn:
                    state_shapes = jax.eval_shape(init_fn, rng, init_data)
                else:
                    assert self.transform_forward
                    state_shapes = jax.eval_shape(self.init_fn, rng, init_data)
            init_state = None
            state = xai_checkpoint.restore(
                checkpoint_path=self.checkpoint_path,
                state_shapes=state_shapes,
                mesh=self.mesh,
                between_hosts_config=self.between_hosts_config,
                state_sharding=self.state_sharding,
                init_state=init_state,
                params_only=True,
            )
            del init_state
        return state

@dataclass
class Request:
    prompt: str
    temperature: float
    nucleus_p: float
    rng_seed: int
    max_len: int

@dataclass
class InferenceRunner:
    name: str
    runner: ModelRunner
    load: str
    tokenizer_path: str = "/tmp/xai_data/tokenizer.model"
    local_mesh_config: Tuple[int, int] = (1, 1)
    between_hosts_config: Tuple[int, int] = (1, 1)
    pad_sizes: Tuple[int] = (1024,)

    def get_pad_bucket(self, size: int) -> int:
        i = bisect.bisect_left(self.pad_sizes, size)
        return self.pad_sizes[min(i, len(self.pad_sizes) - 1)]

    def initialize(self):
        runner = self.runner
        self.runner.transform_forward = True
        dummy_data = dict(
            inputs=np.zeros((1, self.get_pad_bucket(512)), dtype=np.int32),
        )
        state = runner.load_or_init(
            dummy_data,
            from_checkpoint=False,
        )
        runner.params = state.params
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.tokenizer_path)
        def text_to_token_ids(text):
            ids = self.tokenizer.encode(text, out_type=int)
            return ids

        self.text_to_token_ids = text_to_token_ids

    def predict(self, request: Request) -> str:
        rng = jax.random.PRNGKey(request.rng_seed)
        token_ids = self.text_to_token_ids(request.prompt)
        rng, gen_rng = jax.random.split(rng)

        inputs = np.array(token_ids, dtype=np.int32)[np.newaxis, :]

        token_ids = jnp.array(inputs)
        state = self.runner.params

        settings = SampleSettings(
            temperature=jnp.array([request.temperature]),
            nucleus_p=jnp.array([request.nucleus_p]),
            mask=jnp.ones(token_ids.shape, dtype=bool),
            active=jnp.ones(token_ids.shape, dtype=bool),
        )

        for _ in range(request.max_len):
            lm_outputs = self.runner.eval_forward(token_ids)
            sample_output = sample_token(gen_rng, lm_outputs, settings)
            new_token = sample_output.token_id
            token_ids = jnp.concatenate([token_ids, new_token], axis=-1)
            if jnp.argmax(new_token) == 0:
                break

        return self.tokenizer.decode(token_ids.squeeze())

def main():
    runner = ModelRunner(
        model=LanguageModelConfig(),
        checkpoint_path="path_to_checkpoint",
    )
    inference_runner = InferenceRunner(
        name="inference",
        runner=runner,
        load="path_to_load",
        tokenizer_path="path_to_tokenizer_model",
        local_mesh_config=(1, 1),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    request = Request(
        prompt="Sample text",
        temperature=0.7,
        nucleus_p=0.9,
        rng_seed=42,
        max_len=100,
    )
    result = inference_runner.predict(request)
    print(result)

if __name__ == "__main__":
    main()
