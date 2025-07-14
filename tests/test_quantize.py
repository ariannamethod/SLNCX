import pathlib
import sys
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import jax.numpy as jnp
    from quantize import quantize_tensor, dequantize_tensor
except Exception as exc:  # pragma: no cover - env may lack JAX
    pytest.skip(f"JAX unavailable: {exc}", allow_module_level=True)


def test_quantize_tensor():
    t = jnp.array([0.0, 1.0, -1.0], dtype=jnp.float32)
    q = quantize_tensor(t)
    assert q.weight.dtype == jnp.int8
    assert q.scales.dtype == jnp.float32


def test_round_trip():
    t = jnp.array([-2.0, 0.5, 3.5], dtype=jnp.float32)
    q = quantize_tensor(t)
    t_deq = dequantize_tensor(q)
    assert jnp.allclose(t, t_deq, atol=0.5)
