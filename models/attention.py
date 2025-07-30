from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Tuple, Union
import functools

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

import model


class MHAOutput(NamedTuple):
    """Outputs of the multi-head attention operation."""

    embeddings: jax.Array
    memory: Any


def rotate_half(x: jax.Array) -> jax.Array:
    """Obtain the rotated counterpart of each feature."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


class RotaryEmbedding(hk.Module):
    """Applies rotary embeddings (RoPE) to the input sequence tensor."""

    def __init__(self, dim: int, name: Optional[str] = None, base_exponent: int = 10000):
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
        exponents = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = jnp.asarray(1.0 / (self.base_exponent ** (exponents / self.dim)), dtype=jnp.float32)

        if jnp.shape(offset) == ():
            offset = jnp.expand_dims(offset, 0)

        if const_position:
            t = const_position * jnp.ones((1, x.shape[seq_dim]), dtype=jnp.float32)
        elif t is None:
            t = jnp.arange(x.shape[seq_dim], dtype=jnp.float32) + jnp.expand_dims(offset, -1)
        phase = jnp.einsum("bi,j->bij", t, inv_freq)
        phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]

        x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)
        return x.astype(fprop_dtype)


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
        attn_output_multiplier: float = 1.0,
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
        kv_memory: Optional[model.KVMemory] = None,
        mesh: Any = None,
    ) -> MHAOutput:
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

        if not use_memory:
            assert key.shape[:2] == value.shape[:2]

        if mask is not None:
            assert mask.ndim == 4
            assert mask.shape[0] in {1, query.shape[0]}
            if not use_memory:
                assert key.shape[0] in {1, query.shape[0]}
            assert mask.shape[1] == 1
            assert mask.shape[2] in {1, query.shape[1]}
            if not use_memory:
                assert mask.shape[3] in {1, key.shape[1]}

        assert self.num_q_heads % self.num_kv_heads == 0
        query_heads = projection(query, self.key_size, self.num_q_heads, name="query", sharding=P("data", "model"), mesh=mesh)

        new_memory = None
        key_heads = projection(key, self.key_size, self.num_kv_heads, name="key", sharding=P("data", "model"), mesh=mesh)
        value_heads = projection(value, self.value_size, self.num_kv_heads, name="value", sharding=P("data", "model"), mesh=mesh)

        rotate = RotaryEmbedding(dim=self.key_size, base_exponent=int(1e4))
        key_heads = rotate(key_heads, seq_dim=1, offset=(kv_memory.step if kv_memory else 0))
        query_heads = rotate(query_heads, seq_dim=1, offset=(kv_memory.step if kv_memory else 0))

        @functools.partial(jax.vmap)
        def update_into(mem, start, update):
            return jax.lax.dynamic_update_slice_in_dim(mem, update, start, axis=0)

        if kv_memory:
            if mesh is not None:

                @functools.partial(
                    shard_map,
                    mesh=mesh,
                    in_specs=(P("data", None, "model"), P("data"), P("data", None, "model")),
                    out_specs=P("data", None, "model"),
                    check_rep=False,
                )
                def update_into_shmap(mems, starts, updates):
                    return update_into(mems, starts, updates)

                key_heads = update_into_shmap(kv_memory.k, kv_memory.step, key_heads)
                value_heads = update_into_shmap(kv_memory.v, kv_memory.step, value_heads)
            else:
                key_heads = update_into(kv_memory.k, kv_memory.step, key_heads)
                value_heads = update_into(kv_memory.v, kv_memory.step, value_heads)

            new_step = kv_memory.step + sequence_length
            memory_mask = jnp.arange(kv_memory.k.shape[1]) < new_step[:, None]
            memory_mask = memory_mask[:, None, None, :]
            if mask is not None:
                mask = memory_mask * mask
            else:
                mask = memory_mask

            new_memory = model.KVMemory(k=key_heads, v=value_heads, step=new_step)
        query_heads = model.with_sharding_constraint(query_heads, P(self.data_axis, None, "model", None))
        key_heads = model.with_sharding_constraint(key_heads, P(self.data_axis, None, "model", None))
        value_heads = model.with_sharding_constraint(value_heads, P(self.data_axis, None, "model", None))
        b, t, h, d = query_heads.shape
        _, _, kv_h, _ = key_heads.shape
        assert h % kv_h == 0

        query_heads = jnp.reshape(query_heads, (b, t, kv_h, h // kv_h, d))
        query_heads = model.with_sharding_constraint(query_heads, P(self.data_axis, None, "model", None, None))

        attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query_heads, key_heads).astype(jnp.float32)
        attn_logits *= self.attn_output_multiplier
        max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)
        attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)

        mask = mask[:, :, None, :, :]

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality {attn_logits.ndim} for {mask.shape}/{attn_logits.shape}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits).astype(query.dtype)

        attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads)
        attn = model.with_sharding_constraint(attn, P(self.data_axis, None, "model", None, None))
        leading_dims = attn.shape[:2]
        attn = jnp.reshape(attn, (*leading_dims, -1))
        attn = model.with_sharding_constraint(attn, P(self.data_axis, None, "model"))
        final_projection = model.Linear(self.model_size, with_bias=False, sharding=P("model", "data"), mesh=mesh)
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
        y = model.Linear(num_heads * head_size, with_bias=False, name=name, sharding=sharding, mesh=mesh)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, num_heads, head_size))


@dataclass
class MHABlock(hk.Module):
    num_q_heads: int
    num_kv_heads: int
    key_size: int
    attn_output_multiplier: float = 1.0
    mesh: Any = None
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    @hk.transparent
    def __call__(self, inputs: jax.Array, mask: jax.Array, layer_memory: Optional[model.KVMemory]) -> MHAOutput:
        _, _, model_size = inputs.shape
        assert mask.ndim == 4
        assert mask.shape[2] in {1, inputs.shape[1]}
        assert mask.shape[3] in {1, inputs.shape[1]}
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
            )(query, key, value, mask, memory, mesh=self.mesh)

        attn_output = attn_block(inputs, side_input, side_input, mask, layer_memory)
        return attn_output._replace(embeddings=attn_output.embeddings)

