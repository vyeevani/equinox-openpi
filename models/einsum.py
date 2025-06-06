import jax
import jax.numpy as jnp
import equinox 

class Einsum(equinox.Module):
    w: jax.Array = equinox.field(static=False)
    eqn: str = equinox.field(static=True)
    def __init__(
        self,
        shape: tuple[int, ...],
        eqn: str,
        init_fn: jax.nn.initializers.Initializer = jax.nn.initializers.zeros,
        rng: jax.Array = jax.random.PRNGKey(0),
    ):
        self.eqn = eqn
        rng, key = jax.random.split(rng)
        self.w = init_fn(key, shape)
    
    def __call__(self, x: jax.Array):
        return jnp.einsum(self.eqn, x, self.w.astype(x.dtype))