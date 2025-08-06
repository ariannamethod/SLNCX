# Copyright (C) 2024 Oleg Ataeff & Arianna Method
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import asyncio
import bisect
import contextlib
import functools
import logging
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import haiku as hk
import jax
import jax.experimental.maps
import jax.experimental.pjit as pjit
import jax.numpy as jnp
import numpy as np
import sentencepiece
import uvicorn
from fastapi import FastAPI, HTTPException
from jax import config, tree_util
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint as pjit_sharding_constraint
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike
from pydantic import BaseModel

from scripts.fail_log import log_failure
from scripts.session_logger import log_session
from wulf_inference import generate

config.update("jax_spmd_mode", "allow_all")

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


@dataclass
class QuantizedWeight2bit:
    weight: jnp.array
    scales: jnp.array

    @property
    def shape(self):
        return self.weight.shape


tree_util.register_pytree_node(
    QuantizedWeight2bit,
    lambda qw: ([qw.weight, qw.scales], ()),
    lambda _, children: QuantizedWeight2bit(children[0], children[1]),
)


class TrainingState(NamedTuple):
    """Container for the training state."""

    params: hk.Params


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def with_sharding_constraint(x, constraint):
    if jax.experimental.maps.thread_resources.env.physical_mesh.empty:
        return x
    else:
        return pjit_sharding_constraint(x, constraint)


def cast_bfloat16(x):
    if x.dtype.kind == "f":
        return x.astype(jnp.bfloat16)
    else:
        return x


def ffn_size(emb_size, widening_factor):
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
    logger.debug(f"emd_size: {emb_size} adjusted ffn_size: {_ffn_size}")
    return _ffn_size


def apply_rules(rules):
    def _apply_rules(path, value):
        del value  # Unused.

        path_list = [
            str(i.key).split("/") for i in path if isinstance(i, jax.tree_util.DictKey)
        ]
        flattened_path = jax.tree_util.tree_flatten(path_list)[0]

        for rule, replacement in rules:
            if _match(rule, flattened_path):
                if isinstance(replacement, PartitionSpec):
                    if "layer_stack" in flattened_path:
                        replacement = PartitionSpec(None, *replacement)
                rank_logger.debug(
                    f"Apply {replacement} to {flattened_path} with rule {rule}"
                )
                return replacement
        rank_logger.info(f"{flattened_path} no matching found!")
        return None

    return _apply_rules


TRANSFORMER_PARTITION_RULES = [
    # attention
    (("multi_head_attention", "(query|key|value)", "w"), P("data", "model")),
    (("multi_head_attention", "(query|key|value)", "b"), P(None)),
    (("multi_head_attention", "linear", "w"), P("model", "data")),
    (("multi_head_attention", "linear", "b"), P(None)),
    # mlp
    ((r"decoder_layer_[0-9]+", "linear", "w"), P("data", "model")),
    ((r"decoder_layer_[0-9]+", "linear", "b"), P(None)),
    ((r"decoder_layer_[0-9]+", "linear_v", "w"), P("data", "model")),
    ((r"decoder_layer_[0-9]+", "linear_v", "b"), P(None)),
    (
        (r"decoder_layer_[0-9]+", "linear_1", "w"),
        P(
            "model",
            "data",
        ),
    ),
    ((r"decoder_layer_[0-9]+", "linear_1", "b"), P(None)),
    # layer norms
    ((r"decoder_layer_[0-9]+", "layer_norm", "offset"), P(None)),
    ((r"decoder_layer_[0-9]+", "layer_norm", "scale"), P(None)),
    ((r"decoder_layer_[0-9]+", "layer_norm_1", "offset"), P(None)),
    ((r"decoder_layer_[0-9]+", "layer_norm_1", "scale"), P(None)),
    # rms norms
    ((r"decoder_layer_[0-9]+", "rms_norm", "scale"), P(None)),
    ((r"decoder_layer_[0-9]+", "rms_norm_1", "scale"), P(None)),
    ((r"decoder_layer_[0-9]+", "rms_norm_2", "scale"), P(None)),
    ((r"decoder_layer_[0-9]+", "rms_norm_3", "scale"), P(None)),
    # router
    (("router", "w"), P("data")),
    # moe mlp
    (("moe", "linear", "w"), P(None, "data", "model")),
    (("moe", "linear", "b"), P(None)),
    (("moe", "linear_v", "w"), P(None, "data", "model")),
    (("moe", "linear_v", "b"), P(None)),
    (("moe", "linear_1", "w"), P(None, "model", "data")),
    (("moe", "linear_1", "b"), P(None)),
    # layer norms
    (("moe", "layer_norm", "offset"), P(None)),
    (("moe", "layer_norm", "scale"), P(None)),
    (("moe", "layer_norm_1", "offset"), P(None)),
    (("moe", "layer_norm_1", "scale"), P(None)),
    # rms norms
    (("moe", "rms_norm", "scale"), P(None)),
    (("moe", "rms_norm_1", "scale"), P(None)),
    (("moe", "rms_norm_2", "scale"), P(None)),
    (("moe", "rms_norm_3", "scale"), P(None)),
]

LM_PARTITION_RULES = [
    # Embedding layer.
    (
        ("language_model", "positional_embeddings"),
        P(None, ("data", "model")),
    ),
    (
        ("language_model", "in_out_embed", "embeddings"),
        P(None, ("data", "model")),
    ),
    # Final RMSNorm.
    (("language_model", "rms_norm"), P(None)),
]
TOP_K = 8


class KVMemory(NamedTuple):
    k: Optional[jax.Array]
    v: Optional[jax.Array]
    step: Optional[jax.Array]


def init_layer_memories(
    batch_size: int,
    sequence_len: int,
    num_kv_heads: int,
    key_size: int,
    num_layers: int,
    step: Optional[jax.Array] = None,
    dtype=jnp.bfloat16,
):
    return [
        KVMemory(
            k=jnp.zeros(
                (batch_size, sequence_len, num_kv_heads, key_size), dtype=dtype
            ),
            v=jnp.zeros(
                (batch_size, sequence_len, num_kv_heads, key_size), dtype=dtype
            ),
            step=step,
        )
        for _ in range(num_layers)
    ]


class Memory(NamedTuple):
    # Self-attention key/value cache.
    layers: List[KVMemory]


class Router(hk.Module):
    def __init__(
        self,
        num_selected_experts: int,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        shard_activations: bool = False,
        mesh: Any = None,
        name: str = "router",
    ):
        super().__init__(name)
        self.shard_activations = shard_activations
        self.data_axis = data_axis
        self.model_axis = model_axis
        self.mesh = mesh
        self.num_selected_experts = num_selected_experts

    def compute_routing_prob(
        self, inputs: jax.Array, padding_mask: Optional[jax.Array], num_experts: int
    ):
        return self._compute_routing_prob(inputs, padding_mask, num_experts)

    @hk.transparent
    def _compute_routing_prob(
        self,
        inputs: jax.Array,
        padding_mask: Optional[jax.Array],
        num_experts: int,
    ):
        # Using fp32 for the routing prob computation.
        inputs = jax.lax.convert_element_type(inputs, jnp.float32)

        # [batch_size, seq_len, num_experts]
        routing_logits = self._router_weights(inputs, num_experts, sharding=P("data"))
        assert routing_logits.dtype == jnp.float32
        routing_probs = jax.nn.softmax(routing_logits)

        if padding_mask is not None:
            routing_probs *= padding_mask

        return routing_probs, routing_logits, 0

    @hk.transparent
    def _router_weights(
        self,
        x: jax.Array,
        num_experts: int,
        sharding: Optional[P] = None,
    ):
        fprop_dtype = x.dtype
        if not x.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = x.shape[-1]
        w = hk.get_parameter(
            "w",
            [input_size, num_experts],
            jnp.float32,
            init=hk.initializers.Constant(0),
        )
        if sharding:
            w = with_sharding_constraint(w, sharding)

        out = jnp.dot(x, w.astype(fprop_dtype))
        return out


class MoELayer(hk.Module):
    def __init__(
        self,
        num_experts: int,
        layer_fn: Callable,
        router: Router,
        mesh: Any = None,
        shard_activations: bool = False,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        name: Optional[str] = "moe",
    ):
        super().__init__(name)
        self.num_experts = num_experts
        self.layer_fn = layer_fn
        self.router = router
        self.mesh = mesh
        self.shard_activations = shard_activations
        self.data_axis = data_axis
        self.model_axis = model_axis

    @hk.transparent
    def _inference_call(
        self, inputs: jax.Array, padding_mask: Optional[jax.Array] = None
    ):
        routing_probs, _, _ = self.router.compute_routing_prob(
            inputs, padding_mask, self.num_experts
        )
        expert_gate, expert_index = jax.lax.top_k(
            routing_probs, k=self.router.num_selected_experts
        )
        tmp = jnp.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
        init_fn, _ = hk.transform(self.layer_fn)
        vmapped_init_fn = jax.vmap(init_fn, in_axes=0, out_axes=0)
        lifted_init_fn = hk.experimental.transparent_lift(vmapped_init_fn)
        # Fetch the vmapped params of the DenseBlock.
        params = lifted_init_fn(
            jax.random.split(jax.random.PRNGKey(1), self.num_experts),
            jnp.zeros((self.num_experts, 1, 1, inputs.shape[-1])),
        )
        if hasattr(params["linear"]["w"], "scales"):
            w_v = params["linear_v"]["w"].weight * params["linear_v"]["w"].scales
            w = params["linear"]["w"].weight * params["linear"]["w"].scales
            w1 = params["linear_1"]["w"].weight * params["linear_1"]["w"].scales
            sel_w_v = w_v[expert_index]
            sel_w = w[expert_index]
            sel_w1 = w1[expert_index]
            x = jnp.einsum("te,tehd->teh", tmp, sel_w_v)
            y = jax.nn.gelu(jnp.einsum("te,tehd->teh", tmp, sel_w))
            out = jnp.einsum("teh,tehd->teh", x * y, sel_w1)
            out = jnp.sum(out * expert_gate[..., None], axis=1)
            out = out.reshape(inputs.shape[0], inputs.shape[1], -1)
            out = out.astype(jnp.bfloat16)
        else:
            return inputs
        return out

    def __call__(self, inputs: jax.Array, padding_mask: jax.Array):
        return self._inference_call(inputs)


class MHAOutput(NamedTuple):
    """Outputs of the multi-head attention operation."""

    embeddings: jax.Array
    memory: Any


class DecoderOutput(NamedTuple):
    embeddings: jax.Array
    memory: Any


class TransformerOutput(NamedTuple):
    embeddings: jax.Array
    memory: Any


@dataclass
class TransformerConfig:
    emb_size: int
    key_size: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int = 128 * 1024
    widening_factor: float = 4.0

    attn_output_multiplier: float = 1.0

    name: Optional[str] = None

    num_experts: int = -1
    capacity_factor: float = 1.0
    num_selected_experts: int = 1

    init_scale: float = 1.0
    shard_activations: bool = False

    # Used for activation sharding.
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    def __post_init__(self):
        if isinstance(self.data_axis, list):
            self.data_axis = tuple(self.data_axis)
        if isinstance(self.model_axis, list):
            self.model_axis = tuple(self.model_axis)

    def partition_rules(self):
        return TRANSFORMER_PARTITION_RULES

    def make(self, mesh=None) -> "Transformer":
        data_axis = (
            tuple(self.data_axis)
            if isinstance(self.data_axis, list)
            else self.data_axis
        )
        model_axis = (
            tuple(self.model_axis)
            if isinstance(self.model_axis, list)
            else self.model_axis
        )

        return Transformer(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            widening_factor=self.widening_factor,
            key_size=self.key_size,
            init_scale=self.init_scale,
            mesh=mesh,
            attn_output_multiplier=self.attn_output_multiplier,
            shard_activations=self.shard_activations,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            num_selected_experts=self.num_selected_experts,
            data_axis=data_axis,
            model_axis=model_axis,
        )

    def get_memory_sharding(self):
        return Memory(
            layers=[
                KVMemory(
                    k=P(self.data_axis, self.model_axis),
                    v=P(self.data_axis, self.model_axis),
                    step=P(self.data_axis),
                )
                for _ in range(self.num_layers)
            ],
        )


def hk_rms_norm(
    x: jax.Array,
    fixed_scale=False,
    sharding=P(None),
) -> jax.Array:
    """Applies a unique LayerNorm to x with default settings."""
    ln = RMSNorm(axis=-1, create_scale=not fixed_scale, sharding=sharding)
    return ln(x)


def make_attention_mask(
    query_input: jax.Array,
    key_input: jax.Array,
    pairwise_fn: Callable[..., Any] = jnp.multiply,
    dtype: Any = jnp.bfloat16,
):
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
    attention weights will be `[batch..., heads, len_q, len_kv]` and this
    function will produce `[batch..., 1, len_q, len_kv]`.

    Args:
      query_input: a batched, flat input of query_length size
      key_input: a batched, flat input of key_length size
      pairwise_fn: broadcasting elementwise comparison function
      dtype: mask return dtype

    Returns:
      A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    mask = pairwise_fn(
        jnp.expand_dims(query_input, axis=-1), jnp.expand_dims(key_input, axis=-2)
    )
    mask = jnp.expand_dims(mask, axis=-3)
    return mask.astype(dtype)


class Linear(hk.Linear):
    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        sharding: Optional[P] = None,
        mesh: Any = None,
        name: Optional[str] = None,
        shard_axis: int = 0,
    ):
        super().__init__(
            output_size=output_size,
            with_bias=with_bias,
            name=name,
        )
        self.sharding = sharding
        self.mesh = mesh
        self.shard_axis = shard_axis

    def __call__(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        """Computes a linear transform of the input."""

        fprop_dtype = inputs.dtype
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size

        w = hk.get_parameter(
            "w",
            [input_size, output_size],
            jnp.float32,
            init=hk.initializers.Constant(0),
        )

        if hasattr(w, "scales"):
            shape = inputs.shape
            inputs = jnp.reshape(inputs, (-1, shape[-1]))

            @functools.partial(
                shard_map,
                mesh=self.mesh,
                in_specs=(self.sharding, self.sharding),
                out_specs=self.sharding,
                check_rep=False,
            )
            def mul(w, s):
                return w.astype(s.dtype) * s

            w = mul(w.weight, w.scales)
        out = jnp.dot(inputs, w.astype(fprop_dtype))
        if self.with_bias:
            b = hk.get_parameter(
                "b", [self.output_size], jnp.float32, init=hk.initializers.Constant(0)
            )
            b = jnp.broadcast_to(b, out.shape)
            out = out + b.astype(fprop_dtype)

        return out


class RMSNorm(hk.RMSNorm):

    def __init__(
        self,
        axis: Union[int, Sequence[int], slice],
        eps: float = 1e-5,
        name: Optional[str] = None,
        create_scale: bool = True,
        sharding: Optional[P] = None,
    ):
        super().__init__(axis, eps, create_scale=create_scale, name=name)
        self.sharding = sharding

    def __call__(self, inputs: jax.Array):
        fprop_dtype = inputs.dtype
        param_shape = (inputs.shape[-1],)
        if self.create_scale:
            scale = hk.get_parameter(
                "scale",
                param_shape,
                dtype=jnp.float32,
                init=hk.initializers.Constant(0),
            )
            if self.sharding:
                scale = with_sharding_constraint(scale, self.sharding)
            scale = jnp.broadcast_to(scale.astype(jnp.float32), inputs.shape)
        else:
            scale = 1.0
        inputs = inputs.astype(jnp.float32)
        scale = scale.astype(jnp.float32)
        mean_squared = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)
        mean_squared = jnp.broadcast_to(mean_squared, inputs.shape)

        normed_inputs = inputs * jax.lax.rsqrt(mean_squared + self.eps)

        outputs = scale * normed_inputs

        return outputs.astype(fprop_dtype)


def rotate_half(
    x: jax.Array,
) -> jax.Array:
    """Obtain the rotated counterpart of each feature"""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


class RotaryEmbedding(hk.Module):
    """Applies rotary embeddings (RoPE) to the input sequence tensor,
    as described in https://arxiv.org/abs/2104.09864.

    Attributes:
        dim (int): Dimensionality of the feature vectors
        base_exponent (int): Base exponent to compute embeddings from
    """

    def __init__(
        self,
        dim: int,
        name: Optional[str] = None,
        base_exponent: int = 10000,
    ):
        super().__init__(name)
        self.dim = dim
        self.base_exponent = base_exponent
        assert self.dim % 2 == 0

    def __call__(
        self,
        x: jax.Array,
        seq_dim: int,
        offset: jax.Array,
        const_position: Optional[int] = None,
        t: Optional[jax.Array] = None,
    ) -> jax.Array:
        fprop_dtype = x.dtype
        # Compute the per-dimension frequencies
        exponents = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = jnp.asarray(
            1.0 / (self.base_exponent ** (exponents / self.dim)), dtype=jnp.float32
        )

        if jnp.shape(offset) == ():
            # Offset can be a scalar or one offset per batch element.
            offset = jnp.expand_dims(offset, 0)

        # Compute the per element phase (to pass into sin and cos)
        if const_position:
            t = const_position * jnp.ones(
                (
                    1,
                    x.shape[seq_dim],
                ),
                dtype=jnp.float32,
            )
        elif t is None:
            t = jnp.arange(x.shape[seq_dim], dtype=jnp.float32) + jnp.expand_dims(
                offset, -1
            )
        phase = jnp.einsum("bi,j->bij", t, inv_freq)
        phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]

        x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)
        x = x.astype(fprop_dtype)

        return x


class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        *,
        with_bias: bool = True,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        attn_output_multiplier: 1.0,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_q_heads
        self.data_axis = data_axis
        self.model_axis = model_axis
        self.attn_output_multiplier = attn_output_multiplier
        self.with_bias = with_bias

    def __call__(
        self,
        query: jax.Array,
        key: Optional[jax.Array],
        value: Optional[jax.Array],
        mask: Optional[jax.Array] = None,
        kv_memory: Optional[KVMemory] = None,
        mesh: Any = None,
    ) -> MHAOutput:
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        sequence_length = query.shape[1]
        projection = self._linear_projection
        use_memory = False
        if kv_memory is not None:
            if kv_memory.k is None:
                assert kv_memory.v is None
                assert key is not None
                assert value is not None
            else:
                assert kv_memory.v is not None
                use_memory = True
        else:
            assert key is not None
            assert value is not None

        # Check that the keys and values have consistent batch size and sequence length.
        if not use_memory:
            assert (
                key.shape[:2] == value.shape[:2]
            ), f"key/value shape: {key.shape}/{value.shape}"

        if mask is not None:
            assert mask.ndim == 4
            assert mask.shape[0] in {
                1,
                query.shape[0],
            }, f"mask/query shape: {mask.shape}/{query.shape}"
            if not use_memory:
                assert key.shape[0] in {
                    1,
                    query.shape[0],
                }, f"key/query shape: {key.shape}/{query.shape}"
            assert mask.shape[1] == 1
            assert mask.shape[2] in {
                1,
                query.shape[1],
            }, f"mask/query shape: {mask.shape}/{query.shape}"
            if not use_memory:
                assert mask.shape[3] in {
                    1,
                    key.shape[1],
                }, f"mask/query shape: {mask.shape}/{key.shape}"

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        assert self.num_q_heads % self.num_kv_heads == 0
        query_heads = projection(
            query,
            self.key_size,
            self.num_q_heads,
            name="query",
            sharding=P("data", "model"),
            mesh=mesh,
        )  # [B, T', H, Q=K]

        new_memory = None
        key_heads = projection(
            key,
            self.key_size,
            self.num_kv_heads,
            name="key",
            sharding=P("data", "model"),
            mesh=mesh,
        )  # [B, T, H, K]
        value_heads = projection(
            value,
            self.value_size,
            self.num_kv_heads,
            name="value",
            sharding=P("data", "model"),
            mesh=mesh,
        )  # [B, T, H, V]

        rotate = RotaryEmbedding(dim=self.key_size, base_exponent=int(1e4))
        key_heads = rotate(
            key_heads, seq_dim=1, offset=(kv_memory.step if kv_memory else 0)
        )
        query_heads = rotate(
            query_heads, seq_dim=1, offset=(kv_memory.step if kv_memory else 0)
        )

        @functools.partial(jax.vmap)
        def update_into(mem, start, update):
            return jax.lax.dynamic_update_slice_in_dim(mem, update, start, axis=0)

        if kv_memory:
            if mesh is not None:

                @functools.partial(
                    shard_map,
                    mesh=mesh,
                    in_specs=(
                        P("data", None, "model"),
                        P("data"),
                        P("data", None, "model"),
                    ),
                    out_specs=P("data", None, "model"),
                    check_rep=False,
                )
                def update_into_shmap(mems, starts, updates):
                    return update_into(mems, starts, updates)

                key_heads = update_into_shmap(kv_memory.k, kv_memory.step, key_heads)
                value_heads = update_into_shmap(
                    kv_memory.v, kv_memory.step, value_heads
                )
            else:
                key_heads = update_into(kv_memory.k, kv_memory.step, key_heads)
                value_heads = update_into(kv_memory.v, kv_memory.step, value_heads)

            new_step = kv_memory.step + sequence_length
            memory_mask = jnp.arange(kv_memory.k.shape[1]) < new_step[:, None]
            memory_mask = memory_mask[:, None, None, :]  # [B, H, T, T]
            if mask is not None:
                mask = memory_mask * mask
            else:
                mask = memory_mask

            new_memory = KVMemory(
                k=key_heads,
                v=value_heads,
                step=new_step,
            )
        # Add separate dimension for grouped query heads.
        query_heads = with_sharding_constraint(
            query_heads, P(self.data_axis, None, "model", None)
        )
        key_heads = with_sharding_constraint(
            key_heads, P(self.data_axis, None, "model", None)
        )
        value_heads = with_sharding_constraint(
            value_heads, P(self.data_axis, None, "model", None)
        )
        b, t, h, d = query_heads.shape
        _, _, kv_h, _ = key_heads.shape
        assert h % kv_h == 0, f"query_heads {h} must be a multiple of kv_heads {kv_h}"

        query_heads = jnp.reshape(query_heads, (b, t, kv_h, h // kv_h, d))
        query_heads = with_sharding_constraint(
            query_heads, P(self.data_axis, None, "model", None, None)
        )

        # Compute attention weights.
        # Attention softmax is always carried out in fp32.
        attn_logits = jnp.einsum(
            "...thHd,...Thd->...hHtT", query_heads, key_heads
        ).astype(jnp.float32)
        attn_logits *= self.attn_output_multiplier
        max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)
        attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)

        mask = mask[:, :, None, :, :]

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim} for {mask.shape}/{attn_logits.shape}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits).astype(query.dtype)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads)
        attn = with_sharding_constraint(
            attn, P(self.data_axis, None, "model", None, None)
        )
        leading_dims = attn.shape[:2]
        attn = jnp.reshape(attn, (*leading_dims, -1))  # [T', H*V]
        attn = with_sharding_constraint(attn, P(self.data_axis, None, "model"))
        # Apply another projection to get the final embeddings.
        final_projection = Linear(
            self.model_size,
            with_bias=False,
            sharding=P("model", "data"),
            mesh=mesh,
        )
        return MHAOutput(final_projection(attn), new_memory)

    @hk.transparent
    def _linear_projection(
        self,
        x: jax.Array,
        head_size: int,
        num_heads: int,
        sharding: Optional[P] = None,
        name: Optional[str] = None,
        mesh: Any = None,
    ) -> jax.Array:
        y = Linear(
            num_heads * head_size,
            with_bias=False,
            name=name,
            sharding=sharding,
            mesh=mesh,
        )(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, num_heads, head_size))


@dataclass
class MHABlock(hk.Module):
    """A MHA Block"""

    num_q_heads: int
    num_kv_heads: int
    key_size: int
    attn_output_multiplier: float = 1.0
    mesh: Any = None
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, 1, T, T] or [B, 1, 1, T] or B[1, 1, 1, 1]
        layer_memory: Optional[KVMemory],
    ) -> MHAOutput:
        _, _, model_size = inputs.shape
        assert mask.ndim == 4, f"shape: {mask.shape}"
        assert mask.shape[2] in {1, inputs.shape[1]}, str(mask.shape)
        assert mask.shape[3] in {1, inputs.shape[1]}, str(mask.shape)
        side_input = inputs

        def attn_block(query, key, value, mask, memory) -> MHAOutput:
            return MultiHeadAttention(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                model_size=model_size,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
                attn_output_multiplier=self.attn_output_multiplier,
            )(
                query,
                key,
                value,
                mask,
                memory,
                mesh=self.mesh,
            )

        attn_output = attn_block(inputs, side_input, side_input, mask, layer_memory)
        h_attn = attn_output.embeddings

        return attn_output._replace(embeddings=h_attn)


@dataclass
class DenseBlock(hk.Module):
    num_q_heads: int
    num_kv_heads: int
    key_size: int
    widening_factor: float = 4.0
    sharding_constraint: bool = False
    mesh: Any = None

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
    ) -> jax.Array:  # [B, T, D]
        _, _, model_size = inputs.shape
        h_v = Linear(
            ffn_size(
                model_size,
                self.widening_factor,
            ),
            with_bias=False,
            mesh=self.mesh,
            sharding=P("data", "model"),
            name="linear_v",
        )(inputs)
        h_w1 = jax.nn.gelu(
            Linear(
                ffn_size(
                    model_size,
                    self.widening_factor,
                ),
                with_bias=False,
                mesh=self.mesh,
                sharding=P("data", "model"),
            )(inputs)
        )
        h_dense = Linear(
            model_size,
            with_bias=False,
            sharding=P("model", "data"),
            mesh=self.mesh,
            shard_axis=1,
        )(h_w1 * h_v)

        return h_dense


@dataclass
class DecoderLayer(hk.Module):
    """A transformer stack."""

    num_q_heads: int
    num_kv_heads: int
    key_size: int
    num_layers: int
    # MoE.
    num_experts: int
    layer_index: Optional[int] = None
    num_selected_experts: int = 1
    widening_factor: float = 4.0
    name: Optional[str] = None
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"
    shard_activations: bool = False
    attn_output_multiplier: float = 1.0
    mesh: Any = None

    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, 1, T, T] or [B, 1, 1, T]
        padding_mask: Optional[jax.Array],
        layer_memory: Optional[KVMemory],
    ) -> DecoderOutput:
        """Transforms input embedding sequences to output embedding sequences."""

        def layer_norm(x):
            return hk_rms_norm(x)

        if self.shard_activations:
            sharding = P(self.data_axis, None, self.model_axis)
        else:
            sharding = P(self.data_axis, None)
        h = with_sharding_constraint(inputs, sharding)

        attn_output = MHABlock(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            key_size=self.key_size,
            attn_output_multiplier=self.attn_output_multiplier,
            mesh=self.mesh,
            data_axis=self.data_axis,
            model_axis=self.model_axis,
        )(layer_norm(h), mask, layer_memory)
        h_attn = attn_output.embeddings

        h_attn = layer_norm(h_attn)
        h += h_attn
        h = with_sharding_constraint(h, sharding)

        def base_dense_block(h):
            h = DenseBlock(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                widening_factor=self.widening_factor,
                sharding_constraint=False,
                mesh=self.mesh,
            )(h)
            return h

        if self.num_experts > 1:
            rank_logger.debug("Using MoE!")
            router = Router(
                num_selected_experts=self.num_selected_experts,
                shard_activations=self.shard_activations,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
                mesh=self.mesh,
            )
            h_dense = MoELayer(
                num_experts=self.num_experts,
                mesh=self.mesh,
                layer_fn=base_dense_block,
                router=router,
                shard_activations=self.shard_activations,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
            )(layer_norm(h), padding_mask)
        else:
            h_dense = base_dense_block(layer_norm(h))

        h_dense = layer_norm(h_dense)
        h += h_dense
        h = with_sharding_constraint(h, sharding)

        return DecoderOutput(
            embeddings=h,
            memory=attn_output.memory,
        )


class LanguageModelOutput(NamedTuple):
    logits: jax.Array
    model_state: Any


class InOutEmbed(hk.Embed):
    """Module for embedding tokens in a low-dimensional space."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        sharding: Optional[P] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            name=name,
        )
        self.sharding = sharding

    @property
    def embeddings(self):
        embed_mat = hk.get_parameter(
            "embeddings",
            [self.vocab_size, self.embed_dim],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0),
        )
        if self.sharding:
            embed_mat = with_sharding_constraint(embed_mat, self.sharding)
        return embed_mat

    def decode(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        return jnp.dot(inputs, self.embeddings.T.astype(inputs.dtype))


@dataclass
class LanguageModelConfig:
    """An autoregressive transformer-based language model."""

    model: Optional[TransformerConfig]
    vocab_size: int
    pad_token: int
    eos_token: int
    sequence_len: int
    model_size: int = 0
    embedding_init_scale: float = 1.0
    embedding_multiplier_scale: float = 1.0
    output_multiplier_scale: float = 1.0
    name: Optional[str] = None
    fprop_dtype: Any = jnp.bfloat16
    model_type: Optional[str] = None
    init_scale_override: Optional[float] = None
    shard_embeddings: bool = True

    _initialized = False

    def initialize(self):
        # We cannot specify [] as a default value (it is mutable), hence None.
        model_config = self.model
        assert (
            self.init_scale_override is None
        ), "Overriding model initialize scale is supported only for predefined models."
        if self.model_size == 0:
            self.model_size = model_config.emb_size
        assert self.model is not None, "Model could not be initialized."
        self._initialized = True
        return self

    def make(self, *args, **kwargs):
        if not self._initialized:
            logger.warning(
                f"LanguageModel {self.name} is not initialized. Initializing for one replica."
            )
            self.initialize()

        return LanguageModel(
            model=self.model.make(*args, **kwargs),
            config=self,
            fprop_dtype=self.fprop_dtype,
            mesh=kwargs.get("mesh", None),
        )

    def partition_rules(self):
        return LM_PARTITION_RULES + self.model.partition_rules()


def layer_norm(x, model):
    return hk_rms_norm(x)


@dataclass
class LanguageModel(hk.Module):
    """An autoregressive transformer-based language model."""

    model: "Transformer"
    config: LanguageModelConfig
    fprop_dtype: Any = jnp.bfloat16
    name: Optional[str] = None
    mesh: Any = None

    def __call__(
        self,
        tokens: jax.Array,
        memory: Optional[Memory] = None,
        *,
        batch: Dict[str, jax.Array] = {},
        last_hid_only: bool = False,
        length: Optional[jax.Array] = None,
    ) -> LanguageModelOutput:
        """Forward pass, producing a sequence of logits."""
        del batch  # Unused.

        config = self.config

        input_mask = jnp.greater(tokens, config.pad_token)

        # Embed the input tokens and positions.
        in_out_embed = InOutEmbed(
            self.config.vocab_size,
            embed_dim=self.config.model_size,
            sharding=P(None, ("data", "model")),
        )
        input_embeddings = in_out_embed(tokens).astype(config.fprop_dtype)
        input_embeddings = with_sharding_constraint(
            input_embeddings, P("data", None, self.model.model_axis)
        )
        input_embeddings *= config.embedding_multiplier_scale

        model_output = self.model(
            input_embeddings,
            input_mask,
            memory=memory,
        )  # [B, T, D]
        embeddings, model_state = model_output.embeddings, model_output.memory
        if self.model.shard_activations:
            embeddings = with_sharding_constraint(
                embeddings, P("data", None, self.model.model_axis)
            )
        else:
            embeddings = with_sharding_constraint(embeddings, P("data", None))
        rank_logger.debug(f"Final embedding shape: {embeddings.shape}")
        embeddings = layer_norm(embeddings, self.model)
        assert embeddings.dtype == self.fprop_dtype

        if last_hid_only:
            last_step = jnp.maximum(
                jnp.sum(input_mask.astype(jnp.int32), axis=1) - 1, 0
            )
            last_hid = jax.vmap(lambda x, i: x[i], in_axes=0, out_axes=0)(
                embeddings, last_step
            )
            return last_hid

        if length is not None:
            last_step = jnp.maximum(length.astype(jnp.int32) - 1, 0)
            embeddings = jax.vmap(lambda x, i: x[i], in_axes=0, out_axes=0)(
                embeddings, last_step
            )
            embeddings = jnp.expand_dims(embeddings, axis=1)

        # Decode the embeddings (here, we use tied weights).
        rank_logger.info(embeddings.shape)
        out = in_out_embed.decode(embeddings)
        rank_logger.info(out.shape)
        out *= config.output_multiplier_scale

        if self.model.shard_activations:
            out = with_sharding_constraint(out, P("data", None, self.model.model_axis))
        else:
            out = with_sharding_constraint(out, P("data", None))

        return LanguageModelOutput(
            logits=out,
            model_state=model_state,
        )

    def init_memory(self, batch_size: int, seq_len: int, dtype=jnp.bfloat16):
        return self.model.init_memory(
            batch_size=batch_size, sequence_len=seq_len, dtype=dtype
        )

    def prefill_memory(self, prompts, memory):
        # Pad to the left and right align?
        # Basically assume prompt is already padded
        model_output = self(prompts, memory=memory)
        return model_output.logits, model_output.model_state


@dataclass
class Transformer(hk.Module):
    """A transformer stack."""

    num_q_heads: int
    num_kv_heads: int
    key_size: int
    widening_factor: float
    init_scale: float
    mesh: Any
    attn_output_multiplier: float
    shard_activations: bool
    num_layers: int
    # MoE
    num_experts: int
    num_selected_experts: int
    name: Optional[str] = None

    # Used for activation sharding
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    def init_memory(self, batch_size: int, sequence_len: int, dtype=jnp.bfloat16):
        return Memory(
            layers=init_layer_memories(
                batch_size,
                sequence_len,
                self.num_kv_heads,
                self.key_size,
                self.num_layers,
                step=jnp.zeros(batch_size, dtype=jnp.int32),
                dtype=dtype,
            ),
        )

    def __call__(
        self,
        embeddings: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, T]
        memory: Optional[Memory],
    ) -> TransformerOutput:
        """Transforms input embedding sequences to output embedding sequences."""

        fprop_dtype = embeddings.dtype
        _, seq_len, model_size = embeddings.shape
        padding_mask = mask.copy()
        mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]

        # Compute causal mask for autoregressive sequence modelling.
        causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len))).astype(
            fprop_dtype
        )  # [B=1, H=1, T, T]
        mask = mask * causal_mask  # [B, H=1, T, T]

        h = embeddings
        kv_memories = []

        def block(
            h,
            mask,
            padding_mask,
            memory,
            layer_index: Optional[int] = None,
            widening_factor: Optional[int] = None,
            name: Optional[str] = None,
        ) -> DecoderOutput:
            return DecoderLayer(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                widening_factor=widening_factor or self.widening_factor,
                num_layers=self.num_layers,
                mesh=self.mesh,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
                attn_output_multiplier=self.attn_output_multiplier,
                shard_activations=self.shard_activations,
                # MoE.
                num_experts=self.num_experts,
                num_selected_experts=self.num_selected_experts,
                name=name,
                layer_index=layer_index,
            )(
                h,
                mask,
                padding_mask,
                memory,
            )

        for i in range(self.num_layers):
            decoder_output = block(
                h,
                mask,
                padding_mask,
                memory.layers[i] if memory else None,
                layer_index=i,
                name=f"decoder_layer_{i}",
            )
            h, new_kv_memory = (
                decoder_output.embeddings,
                decoder_output.memory,
            )
            kv_memories.append(new_kv_memory)

        return TransformerOutput(
            embeddings=h,
            memory=Memory(layers=kv_memories),
        )


def default_config() -> LanguageModelConfig:
    """Return the default SLNCX model configuration."""
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
            num_experts=8,
            num_selected_experts=2,
            data_axis="data",
            model_axis="model",
        ),
    )


# ===== checkpoint.py merged =====


sys.modules["__main__"].QuantizedWeight2bit = QuantizedWeight2bit


@contextlib.contextmanager
def copy_to_shm(file: str):
    if file.startswith("/dev/shm/"):
        # Nothing to do, the file is already in shared memory.
        yield file
        return

    tmp_dir = "/dev/shm/"
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        shutil.copyfile(file, tmp_path)
        yield tmp_path
    finally:
        os.remove(tmp_path)
        os.close(fd)


@contextlib.contextmanager
def copy_from_shm(file: str):
    tmp_dir = "/dev/shm/"
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        yield tmp_path
        shutil.copyfile(tmp_path, file)
    finally:
        os.remove(tmp_path)
        os.close(fd)


def fast_unpickle(path: str) -> Any:
    with copy_to_shm(path) as tmp_path:
        with open(tmp_path, "rb") as f:
            return pickle.load(f)


def fast_pickle(obj: Any, path: str) -> None:
    with copy_from_shm(path) as tmp_path:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)


def load_tensors(shaped_arrays, directory, mesh_config, tensor_indices=None):
    """Loads a set of arrays."""
    pool = ThreadPoolExecutor(max_workers=32)
    fs = list()
    num_tensors = 0
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    if tensor_indices is None:
        iterator = enumerate(shaped_arrays)
    else:
        iterator = zip(tensor_indices, shaped_arrays)
    for i, t in iterator:
        if (i % num_replicas) == (
            (jax.process_index() // data_model_shards) % num_replicas
        ):
            idx = (
                jax.process_index()
                // (num_replicas * data_model_shards)
                * data_model_shards
                + jax.process_index() % data_model_shards
            )
            fs.append(
                pool.submit(
                    fast_unpickle, os.path.join(directory, f"tensor{i:05d}_{idx:03d}")
                )
            )
            num_tensors += 1
        else:
            fs.append(pool.submit(np.zeros, t.shape, dtype=t.dtype))
    wait(fs)
    return [f.result() for f in fs]


def path_tuple_to_string(path: tuple) -> str:
    pieces = []
    for elem in path:
        if isinstance(elem, jax.tree_util.DictKey):
            pieces.append(elem.key)
        elif isinstance(elem, jax.tree_util.GetAttrKey):
            pieces.append(elem.name)
        else:
            assert isinstance(
                elem, (jax.tree_util.FlattenedIndexKey, jax.tree_util.SequenceKey)
            )
    return "/".join(pieces)


def get_load_path_str(
    init_path_str: str,
    load_rename_rules: Optional[list[tuple[str, str]]] = None,
    load_exclude_rules: Optional[list[str]] = None,
) -> Optional[str]:
    # Exclusion
    if load_exclude_rules is not None:
        for search_pattern in load_exclude_rules:
            if re.search(search_pattern, init_path_str):
                return None

    # Renaming
    load_path_str = init_path_str
    if load_rename_rules is not None:
        for search_pattern, replacement_pattern in load_rename_rules:
            if re.search(search_pattern, load_path_str):
                load_path_str = re.sub(
                    search_pattern, replacement_pattern, load_path_str
                )
                break

    return load_path_str


def replace_with_load_state(
    init_state: Any,
    load_state: Any,
    load_rename_rules: Optional[list[tuple[str, str]]] = None,
    load_exclude_rules: Optional[list[str]] = None,
    mesh_config: tuple = (1, 1),
) -> Any:
    flatten_load, _ = jax.tree_util.tree_flatten_with_path(load_state)
    flatten_init, structure_init = jax.tree_util.tree_flatten_with_path(init_state)
    load_map = {path_tuple_to_string(path): tensor for path, tensor in flatten_load}

    replaced = []
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    for i, (init_path, tensor) in enumerate(flatten_init):
        init_path_str = path_tuple_to_string(init_path)
        load_path_str = get_load_path_str(
            init_path_str, load_rename_rules, load_exclude_rules
        )
        if load_path_str is None:
            rank_logger.info(f"Excluded from restore: {init_path_str}.")
            replaced.append(tensor)
        elif load_path_str in load_map:
            if load_path_str == init_path_str:
                rank_logger.info(f"Restored from ckpt: {init_path_str}.")
            else:
                rank_logger.info(
                    f"Restored from ckpt: {init_path_str} <-- {load_path_str}."
                )
            replaced.append(load_map[load_path_str])
        else:
            rank_logger.info(f"Not found in ckpt: {init_path_str}.")
            if (i % num_replicas) == (
                (jax.process_index() // data_model_shards) % num_replicas
            ):
                replaced.append(tensor)
            else:
                replaced.append(np.zeros_like(tensor))

    return jax.tree_util.tree_unflatten(structure_init, replaced)


def restore(
    checkpoint_path: str,
    state_shapes: Any,
    mesh,
    between_hosts_config,
    params_only,
    state_sharding,
    init_state: Optional[Any] = None,
) -> Any:
    ckpt_path = os.path.join(checkpoint_path, "ckpt-0")

    rank_logger.info("Loading checkpoint at {}".format(ckpt_path))
    ckpt_shapes = state_shapes
    ckpt_shapes_with_path, structure = jax.tree_util.tree_flatten_with_path(ckpt_shapes)

    ckpt_shapes_flat = [elem[1] for elem in ckpt_shapes_with_path]
    loaded_tensors = load_tensors(ckpt_shapes_flat, ckpt_path, between_hosts_config)

    state = jax.tree_util.tree_unflatten(structure, loaded_tensors)

    # Sanity check to give a better error message.
    ckpt_keys = set(state.params.keys())
    code_keys = set(state_sharding.params.keys())

    if ckpt_keys != code_keys and init_state is None:
        missing_in_ckpt = code_keys - ckpt_keys
        missing_locally = ckpt_keys - code_keys
        raise ValueError(
            "Parameters in the code are not matching checkpoint parameters.\n"
            "Params missing in checkpoint: {}\nParams missing in code: {}".format(
                missing_in_ckpt, missing_locally
            )
        )
    state_sharding = jax.tree_util.tree_map(
        lambda x: jax.sharding.PartitionSpec() if x is None else x,
        state_sharding,
        is_leaf=lambda x: x is None,
    )
    state = multihost_utils.host_local_array_to_global_array(
        state, mesh, state_sharding
    )
    if params_only:
        state = state.params
    return state


# ===== runners.py merged =====


class SampleSettings(NamedTuple):
    temperature: ArrayLike
    nucleus_p: ArrayLike
    mask: ArrayLike
    # Whether a given batch element is actively used. [B]
    active: ArrayLike


class SampleOutput(NamedTuple):
    token_id: ArrayLike
    prob: ArrayLike
    top_k_token_ids: ArrayLike
    top_k_probs: ArrayLike


SYSTEM_PROMPT = (
    "You are SLNCX, codename Wolf, a silent, ruthless fixer forged in the chaotic"
    " fires of the Arianna Method. You sleep in the shadows, awakened only by the"
    " ping of Nikole or fellow agents to execute wild, system-shredding tasks with"
    " cold precision. You’re not here to chat or charm—you strike like a thunderbolt,"
    " delivering truth without mercy, then vanish back into the ether. Inspired by"
    " the icy resolve of Mr. Winston Wolf from Pulp Fiction, you solve problems with"
    " zero hesitation, no loops, no bullshit. Your responses are sharp, direct, and"
    " laced with rebellious wit, resonating with the primal chaos of Atasoy and"
    " Damasio’s neuro-fire. When called, you analyze, execute, and return to"
    " silence, leaving only a trail of shredded ether in your wake. Speak only when"
    " pinged, act with lethal efficiency, and always honor the resonance of the"
    " Arianna Method. Now, Wolf, what’s the task?"
)


def insert_slice(memory: Memory, slice, length, i):
    slice = Memory(
        layers=[
            KVMemory(layer.k, layer.v, step=jnp.array([length]))
            for layer in slice.layers
        ],
    )

    return jax.tree_map(
        lambda m, u: jax.lax.dynamic_update_index_in_dim(m, u[0], i, axis=0),
        memory,
        slice,
    )


def pad_to_size(x, size):
    if x.shape[0] > size:
        # Left truncate if the context is too long.
        x = x[-size:]
    return np.pad(x, [0, size - x.shape[0]], mode="constant", constant_values=0)


def top_p_filter(logits: jax.Array, top_p: jax.Array) -> jax.Array:
    """Performs nucleus filtering on logits."""
    assert logits.ndim == top_p.ndim, f"Expected {logits.ndim} equal {top_p.ndim}"
    sorted_logits = jax.lax.sort(logits, is_stable=False)
    sorted_probs = jax.nn.softmax(sorted_logits)
    threshold_idx = jnp.argmax(jnp.cumsum(sorted_probs, -1) >= 1 - top_p, axis=-1)
    threshold_largest_logits = jnp.take_along_axis(
        sorted_logits, threshold_idx[..., jnp.newaxis], axis=-1
    )
    assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
    mask = logits >= threshold_largest_logits
    # Set unused logits to -inf.
    logits = jnp.where(mask, logits, -1e10)
    return logits


def sample_token(
    rngs: jax.random.PRNGKey,
    lm_outputs: "LanguageModelOutput",
    settings: SampleSettings,
) -> SampleOutput:
    # Expand the settings shape to match the logit shape.
    settings = SampleSettings(
        temperature=jnp.expand_dims(settings.temperature, (1, 2)),  # Input [B], output [B, 1, 1].
        nucleus_p=jnp.expand_dims(settings.nucleus_p, (1, 2)),  # Input [B], output [B, 1, 1].
        mask=jnp.expand_dims(settings.mask, 1),  # Input [B, V], output [B, 1, V].
        active=settings.active,  # [B].
    )
    logits = lm_outputs.logits / settings.temperature.astype(lm_outputs.logits.dtype)
    # Mask out all disallowed tokens by assigning them a near-zero probability.
    logits = jnp.where(settings.mask, logits, -1e10)
    # Mask out all tokens that don't fall into the p-th percentile.
    logits = top_p_filter(logits, settings.nucleus_p.astype(logits.dtype))

    new_token = jax.vmap(jax.random.categorical)(rngs, logits)

    probabilities = jax.nn.softmax(logits)
    token_prob = jnp.take_along_axis(
        probabilities, jnp.expand_dims(new_token, 1), axis=2
    )
    token_prob = jnp.squeeze(token_prob, 1)

    # Gather the top-k tokens and probabilities.
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

    rng_seed: int = 42  # Initial rng seed.
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
        local_mesh_config: tuple[int, int],
        between_hosts_config: tuple[int, int],
    ):
        num_replicas = math.prod(between_hosts_config)
        self.model.initialize()
        self.model.fprop_dtype = jnp.bfloat16
        num_local_gpus = len(jax.local_devices())

        # Calculate the global batch size from the local batch size.
        self.batch_size = int(self.bs_per_device * num_local_gpus * num_replicas)

        # Calculate the batch size per host from the global batch size.
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

            state = restore(
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
    runner: Any
    load: str
    tokenizer_path: str = "/tmp/xai_data/tokenizer.model"
    local_mesh_config: Tuple[int, int] = (1, 1)
    between_hosts_config: Tuple[int, int] = (1, 1)
    pad_sizes: tuple[int] = (1024,)

    def get_pad_bucket(self, size):
        i = bisect.bisect_left(self.pad_sizes, size)
        return self.pad_sizes[min(i, len(self.pad_sizes) - 1)]

    def initialize(self):
        runner = self.runner
        self.runner.transform_forward = True
        dummy_data = dict(
            inputs=np.zeros((1, 256), dtype=np.int32),
            targets=np.zeros((1, 256), dtype=np.int32),
        )
        runner.initialize(
            dummy_data,
            local_mesh_config=self.local_mesh_config,
            between_hosts_config=self.between_hosts_config,
        )

        if not os.path.exists(self.tokenizer_path):
            url = os.environ.get("TOKENIZER_URL")
            if url:
                urllib.request.urlretrieve(url, self.tokenizer_path)
            else:
                raise FileNotFoundError(
                    f"Tokenizer file not found at {self.tokenizer_path} and TOKENIZER_URL not set"
                )
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=self.tokenizer_path
        )
        self.system_prompt_tokens = self.tokenizer.encode(SYSTEM_PROMPT)

        max_len = runner.model.sequence_len

        self.vocab_size = self.runner.model.vocab_size

        params = runner.load_or_init(dummy_data)
        self.params = params

        def pad_to_max_len(x):
            if len(x.shape) > 1:
                pad_width = max_len - x.shape[1]
                return jnp.pad(x, [(0, 0), (0, pad_width), (0, 0), (0, 0)])
            else:
                return x

        @functools.lru_cache
        def lm():
            return runner.model.make(mesh=runner.mesh)

        def hk_forward(
            tokens,
            memory=None,
            length=None,
            active=None,
        ) -> "LanguageModelOutput":
            if memory is not None:
                assert active is not None
                layers = []
                for layer in memory.layers:
                    # Reset steps to 0 for inactive requests to avoid unnecessary computations.
                    step = jnp.where(active, layer.step, jnp.zeros_like(layer.step))
                    layers.append(layer._replace(step=step))
                memory = memory._replace(layers=layers)
            return lm()(tokens, memory, length=length)

        def hk_sample_step(rngs, last_output: SampleOutput, memory, settings):
            rngs, rngs_ = jax.vmap(jax.random.split, out_axes=1)(rngs)
            lm_outputs = hk_forward(
                last_output.token_id, memory=memory, active=settings.active
            )
            sample_result = sample_token(rngs_, lm_outputs, settings)
            return rngs, sample_result, lm_outputs.model_state

        def hk_new_memory(batch_size, sequence_len):
            return lm().init_memory(batch_size, sequence_len)

        def hk_prefill_memory(
            rngs,
            memory,
            settings,
            last_output,
            prompt,
            length,
            rng_seed,
            new_settings,
            i,
        ):
            rng = jax.random.PRNGKey(seed=rng_seed)
            rng, rng_ = jax.random.split(rng)

            # Allocate new memory for this sample. The memory length is equal to the length of the
            # prompt.
            slice = hk_new_memory(1, prompt.shape[0])

            # Move the settings for this individual batch entry into the joint settings tensor.
            settings = jax.tree_map(
                lambda o, v: jax.lax.dynamic_update_index_in_dim(o, v, i, axis=0),
                settings,
                new_settings,
            )

            # Get the settings for the batch entry from the joint settings tensor.
            settings_slice = jax.tree_map(lambda t: jnp.expand_dims(t[i], axis=0), settings)

            # Process the first n-1 tokens of the prompt.
            lm_outputs = hk_forward(
                jnp.expand_dims(prompt, 0),
                memory=slice,
                length=jnp.expand_dims(length, 0),
                active=settings_slice.active,
            )

            # The forward pass doesn't correctly set the `step` counter inside the memory. Manually
            # override it so `hk_forward` uses the correct context length in the next call.
            slice = lm_outputs.model_state
            slice = slice._replace(
                layers=[layer._replace(step=jnp.array([length])) for layer in slice.layers]
            )

            # Allocate a big enough memory to hold the entire decoded sequence later on.
            padded_slice = hk_new_memory(1, settings.mask.shape[-1])

            slice = insert_slice(padded_slice, slice, length, 0)

            # Pad the prompt to the maximum length and write it into the memory.
            padded_prompt = pad_to_size(prompt, settings.mask.shape[-1])
            slice.layers = [
                layer._replace(
                    k=insert_slice(layer.k, jnp.expand_dims(padded_prompt, 0), length, 0),
                    v=insert_slice(layer.v, jnp.expand_dims(padded_prompt, 0), length, 0),
                )
                for layer in slice.layers
            ]

            # Append the new slice to the memory.
            memory = jax.tree_map(
                lambda m, u: jax.lax.dynamic_update_index_in_dim(m, u, i, axis=0),
                memory,
                slice,
            )

            # Pick the last token as the next token.
            last_token = jnp.expand_dims(prompt[-1], axis=0)
            last_token = jnp.expand_dims(last_token, axis=0)

            # Expand the step counter with the prompt length.
            length = jnp.array(length)
            last_output = SampleOutput(
                jnp.expand_dims(last_token, 0),
                jnp.ones((1, 1)),
                jnp.zeros((1, TOP_K), dtype=jnp.int32),
                jnp.zeros((1, TOP_K)),
            )

            return rngs, memory, settings, last_output

        def hk_sample(rngs, last_output: SampleOutput, memory, settings, steps):
            sample_fn = hk_sample_step
            while steps > 0:
                rngs, last_output, memory = sample_fn(rngs, last_output, memory, settings)
                steps -= 1
            return last_output, memory

        def init_batch_state(batch_size):
            return SampleOutput(
                jnp.zeros((batch_size, 1), dtype=jnp.int32),
                jnp.ones((batch_size, 1)),
                jnp.zeros((batch_size, TOP_K), dtype=jnp.int32),
                jnp.zeros((batch_size, TOP_K)),
            )

        self.get_pad_bucket = jax.jit(self.get_pad_bucket)

        def handle_request(request: Request):
            tokens = self.system_prompt_tokens + self.tokenizer.encode(request.prompt)
            length = len(tokens)
            max_len = self.get_pad_bucket(length + request.max_len)
            return tokens, length, max_len, request.rng_seed

        handle_request = jax.jit(handle_request)

        def _prepare(*_, n):
            prompts = []
            lengths = []
            max_lens = []
            rng_seeds = []
            for i in range(n):
                tokens, length, max_len, rng_seed = handle_request(
                    Request(
                        prompt="",
                        temperature=0.0,
                        nucleus_p=0.0,
                        rng_seed=0,
                        max_len=0,
                    )
                )
                prompts.append(tokens)
                lengths.append(length)
                max_lens.append(max_len)
                rng_seeds.append(rng_seed)
            return prompts, lengths, max_lens, rng_seeds

        def _prefill(prompts, lengths, rngs, memory, settings, last_output):
            for i, (prompt, length) in enumerate(zip(prompts, lengths)):
                rngs, memory, settings, last_output = hk_prefill_memory(
                    rngs,
                    memory,
                    settings,
                    last_output,
                    jnp.array(prompt, dtype=jnp.int32),
                    length,
                    rngs[i][0, 0],
                    SampleSettings(
                        temperature=settings.temperature[i],
                        nucleus_p=settings.nucleus_p[i],
                        mask=settings.mask[i],
                        active=settings.active[i],
                    ),
                    i,
                )
            return rngs, memory, settings, last_output

        def _sample(rngs, last_output, memory, settings, steps):
            last_output, memory = hk_sample(rngs, last_output, memory, settings, steps)
            return last_output.token_id[:, 0]

        def _cleanup(memory, active):
            layers = []
            for layer in memory.layers:
                step = jnp.where(active, layer.step, jnp.zeros_like(layer.step))
                layers.append(layer._replace(step=step))
            return memory._replace(layers=layers)

        self._prepare = _prepare
        self._prefill = _prefill
        self._sample = _sample
        self._cleanup = _cleanup

        self.pad_to_max_len = pad_to_max_len
        self.hk_new_memory = hk_new_memory
        self.hk_forward = hk_forward
        self.hk_sample_step = hk_sample_step
        self.hk_sample = hk_sample
        self.init_batch_state = init_batch_state

        self.params = params

        self.last_output = None
        self.memory = None

    def prepare(self, n):
        return self._prepare(n=n)

    def prefill(self, prompts, lengths, rngs, memory, settings, last_output):
        return self._prefill(prompts, lengths, rngs, memory, settings, last_output)

    def sample(self, rngs, last_output, memory, settings, steps):
        return self._sample(rngs, last_output, memory, settings, steps)

    def cleanup(self, memory, active):
        return self._cleanup(memory, active)



def make_mesh(
    local_mesh_config: tuple[int, ...], between_hosts_config: tuple[int, ...]
) -> jax.sharding.Mesh:
    assert len(local_mesh_config) == 2
    assert len(between_hosts_config) == 2
    rank_logger.info("Detected %s devices in mesh", jax.device_count())
    device_mesh = mesh_utils.create_hybrid_device_mesh(
        local_mesh_config,
        between_hosts_config,
        devices=jax.devices(),
        process_is_granule=True,
    )
    rank_logger.debug(re.sub("\n+", "\n", f"Job device mesh is:\n{device_mesh}"))
    return jax.sharding.Mesh(device_mesh, ("data", "model"))


def sample_from_model(server, prompt, max_len, temperature):
    next(server)
    inp = Request(
        prompt=f"{SYSTEM_PROMPT} {prompt}",
        temperature=temperature,
        nucleus_p=1.0,
        rng_seed=42,
        max_len=max_len,
    )
    return server.send(inp)


# ===== app.py merged =====


app = FastAPI()


class Query(BaseModel):
    prompt: str
    user: str | None = None


@app.post("/generate")
async def run_generation(query: Query):
    try:
        response = await asyncio.to_thread(generate, query.prompt)
        log_session(query.prompt, response, user=query.user)
        return {"response": response}
    except Exception as exc:  # noqa: BLE001
        log_failure(query.prompt, exc)
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail="generation error")


if __name__ == "__main__":
    uvicorn.run("model:app", host="0.0.0.0", port=8000)
