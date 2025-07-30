from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import model


@dataclass
class Router(hk.Module):
    num_selected_experts: int
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"
    shard_activations: bool = False
    mesh: Any = None
    name: str = "router"

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("name", "router"))
        self.num_selected_experts = kwargs.get("num_selected_experts")
        self.data_axis = kwargs.get("data_axis", "data")
        self.model_axis = kwargs.get("model_axis", "model")
        self.shard_activations = kwargs.get("shard_activations", False)
        self.mesh = kwargs.get("mesh")

    def compute_routing_prob(self, inputs: jax.Array, padding_mask: Optional[jax.Array], num_experts: int):
        return self._compute_routing_prob(inputs, padding_mask, num_experts)

    @hk.transparent
    def _compute_routing_prob(
        self,
        inputs: jax.Array,
        padding_mask: Optional[jax.Array],
        num_experts: int,
    ):
        inputs = jax.lax.convert_element_type(inputs, jnp.float32)
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
        w = hk.get_parameter("w", [input_size, num_experts], jnp.float32, init=hk.initializers.Constant(0))
        if sharding:
            w = model.with_sharding_constraint(w, sharding)

        out = jnp.dot(x, w.astype(fprop_dtype))
        return out


@dataclass
class MoELayer(hk.Module):
    num_experts: int
    layer_fn: Callable
    router: Router
    mesh: Any = None
    shard_activations: bool = False
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"
    name: Optional[str] = "moe"

    @hk.transparent
    def _inference_call(self, inputs: jax.Array, padding_mask: Optional[jax.Array] = None):
        routing_probs, _, _ = self.router.compute_routing_prob(inputs, padding_mask, self.num_experts)
        expert_gate, expert_index = jax.lax.top_k(routing_probs, k=self.router.num_selected_experts)
        tmp = jnp.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
        init_fn, _ = hk.transform(self.layer_fn)
        vmapped_init_fn = jax.vmap(init_fn, in_axes=0, out_axes=0)
        lifted_init_fn = hk.experimental.transparent_lift(vmapped_init_fn)
        params = lifted_init_fn(jax.random.split(jax.random.PRNGKey(1), self.num_experts), jnp.zeros((self.num_experts, 1, 1, inputs.shape[-1])))
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

