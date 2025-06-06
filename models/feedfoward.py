import jax
import jax.numpy as jnp
import equinox

class FeedForward(equinox.Module):
    gating_einsum: jax.Array = equinox.field(static=False)
    linear: jax.Array = equinox.field(static=False)
    
    def __init__(self, features: int, hidden_dim: int, rng: jax.Array = jax.random.PRNGKey(0)):
        rng, key = jax.random.split(rng)
        self.gating_einsum = jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,))(
            key,
            (2, features, hidden_dim),
        )
        rng, key = jax.random.split(rng)
        self.linear = jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1)(
            key,
            (hidden_dim, features),
        )
    def __call__(self, x: jax.Array):
        dtype = x.dtype
        ff_gate = jnp.dot(x, self.gating_einsum[0].astype(x.dtype))
        gate_value = jax.nn.gelu(ff_gate)
        ff1 = jnp.dot(x, self.gating_einsum[1].astype(x.dtype))
        activations = gate_value * ff1
        outputs = jnp.dot(activations, self.linear.astype(x.dtype))
        assert outputs.dtype == dtype
        return outputs