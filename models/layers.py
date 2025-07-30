from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import model
from .attention import MHABlock
from .moe import MoELayer, Router


@dataclass
class DenseBlock(hk.Module):
    num_q_heads: int
    num_kv_heads: int
    key_size: int
    widening_factor: float = 4.0
    sharding_constraint: bool = False
    mesh: Any = None

    @hk.transparent
    def __call__(self, inputs: jax.Array) -> jax.Array:
        _, _, model_size = inputs.shape
        h_v = model.Linear(
            model.ffn_size(model_size, self.widening_factor),
            with_bias=False,
            mesh=self.mesh,
            sharding=P("data", "model"),
            name="linear_v",
        )(inputs)
        h_w1 = jax.nn.gelu(
            model.Linear(
                model.ffn_size(model_size, self.widening_factor),
                with_bias=False,
                mesh=self.mesh,
                sharding=P("data", "model"),
            )(inputs)
        )
        h_dense = model.Linear(
            model_size,
            with_bias=False,
            sharding=P("model", "data"),
            mesh=self.mesh,
            shard_axis=1,
        )(h_w1 * h_v)
        return h_dense


@dataclass
class DecoderOutput(NamedTuple):
    embeddings: jax.Array
    memory: Any


@dataclass
class DecoderLayer(hk.Module):
    num_q_heads: int
    num_kv_heads: int
    key_size: int
    num_layers: int
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

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        padding_mask: Optional[jax.Array],
        layer_memory: Optional[model.KVMemory],
    ) -> DecoderOutput:
        def layer_norm(x):
            return model.hk_rms_norm(x)

        sharding = P(self.data_axis, None, self.model_axis) if self.shard_activations else P(self.data_axis, None)
        h = model.with_sharding_constraint(inputs, sharding)

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
        h = model.with_sharding_constraint(h, sharding)

        def base_dense_block(x):
            return DenseBlock(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                widening_factor=self.widening_factor,
                sharding_constraint=False,
                mesh=self.mesh,
            )(x)

        if self.num_experts > 1:
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
        h = model.with_sharding_constraint(h, sharding)

        return DecoderOutput(embeddings=h, memory=attn_output.memory)

