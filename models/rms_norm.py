import jax
import jax.numpy as jnp
import equinox

class RMSNorm(equinox.Module):
    scale: jax.Array = equinox.field(static=False)
    def __init__(self, dim: int): 
        self.scale = jnp.zeros(dim)
    def __call__(self, x):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        normed_inputs = normed_inputs * (
            1 + self.scale
        )
        return normed_inputs.astype(dtype)