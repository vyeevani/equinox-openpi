import jax
import jax.numpy as jnp
import equinox

class Embedder(equinox.Module):
    input_embedding: jax.Array = equinox.field(static=False)
    def __init__(self, vocab_size: int, embed_dim: int, rng: jax.Array):
        self.input_embedding = jax.nn.initializers.normal()(
            rng,
            (vocab_size, embed_dim),
        )
    def encode(self, x):
        x = self.input_embedding[(x,)]
        x *= jnp.sqrt(self.input_embedding.shape[1]).astype(x.dtype)
        return x
    def decode(self, x):
        return jnp.dot(x, self.input_embedding.T)