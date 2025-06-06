import dataclasses
import re
import math

import jax
import jax.numpy as jnp
import equinox 

@dataclasses.dataclass
class Config:
    rank: int
    alpha: float = 1.0
    rslora: bool = False
    axes: tuple[int, int] = (-2, -1)
    label: str = "L"
    init_fn: jax.nn.initializers.Initializer = jax.nn.initializers.normal(stddev=0.01)
    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank

class Einsum(equinox.Module):
    w: jax.Array = equinox.field(static=False)
    lora_a: jax.Array = equinox.field(static=False)
    lora_b: jax.Array = equinox.field(static=False)
    
    eqn: str = equinox.field(static=True)
    eqn_a: str = equinox.field(static=True)
    eqn_b: str = equinox.field(static=True)
    scaling_value: float = equinox.field(static=True)

    def __init__(
        self,
        shape: tuple[int, ...],
        eqn: str,
        config: Config,
        init_fn: jax.nn.initializers.Initializer = jax.nn.initializers.zeros,
        rng: jax.Array = jax.random.PRNGKey(0),
    ):  
        self.scaling_value = config.scaling_value
        self.eqn = eqn
        self.w = init_fn(rng, shape)
        
        # Make LoRA eqns
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()
        a_label, b_label = (rhs[x] for x in config.axes)
        a_rhs = rhs.replace(b_label, config.label)
        a_out = out.replace(b_label, config.label)
        self.eqn_a = f"{lhs},{a_rhs}->{a_out}"
        b_rhs = rhs.replace(a_label, config.label)
        self.eqn_b = f"{a_out},{b_rhs}->{out}"
        
        # Make LoRA weights
        shape_a, shape_b = list(shape), list(shape)
        shape_a[config.axes[1]] = config.rank
        shape_b[config.axes[0]] = config.rank
        rng, key = jax.random.split(rng)
        self.lora_a = config.init_fn(key, (shape[config.axes[0]], config.rank))
        rng, key = jax.random.split(rng)
        self.lora_b = config.init_fn(key, (config.rank, shape[config.axes[1]]))
    def __call__(self, x: jax.Array):
        result = jnp.einsum(self.eqn, x, self.w.astype(x.dtype))
        lora = jnp.einsum(self.eqn_a, x, self.lora_a.astype(x.dtype))
        lora = jnp.einsum(self.eqn_b, lora, self.lora_b.astype(x.dtype))
        result = result + lora * self.scaling_value
        return result
            
class FeedForward(equinox.Module):
    gating_einsum: jax.Array = equinox.field(static=False)
    linear: jax.Array = equinox.field(static=False)
    gating_einsum_lora_a: jax.Array = equinox.field(static=False)
    gating_einsum_lora_b: jax.Array = equinox.field(static=False)
    linear_lora_a: jax.Array = equinox.field(static=False)
    linear_lora_b: jax.Array = equinox.field(static=False)
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        config: Config,
        rng: jax.Array = jax.random.PRNGKey(0),
    ):
        rng, key = jax.random.split(rng)
        self.gating_einsum = jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,))(
            key,
            (2, features, hidden_dim),
        )
        self.linear = jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1)(
            key,
            (hidden_dim, features),
        )
        self.gating_einsum_lora_a = config.init_fn(
            key,
            (2, features, config.rank),
        )
        self.gating_einsum_lora_b = config.init_fn(
            key,
            (2, config.rank, hidden_dim),
        )
        self.linear_lora_a = config.init_fn(
            key,
            (hidden_dim, config.rank),
        )
        self.linear_lora_b = config.init_fn(
            key,
            (config.rank, features),
        )
    def __call__(self, x: jax.Array):
        dtype = x.dtype
        ff_gate = self._dot(
            x,
            self.gating_einsum[0],
            (self.gating_einsum_lora_a[0], self.gating_einsum_lora_b[0]),
        )
        gate_value = jax.nn.gelu(ff_gate)
        ff1 = self._dot(
            x,
            self.gating_einsum[1],
            (self.gating_einsum_lora_a[1], self.gating_einsum_lora_b[1]),
        )
        activations = gate_value * ff1
        outputs = self._dot(
            activations,
            self.linear,
            (self.linear_lora_a, self.linear_lora_b),
        )
        assert outputs.dtype == dtype
        return outputs
    def _dot(self, x: jax.Array, w: jax.Array, lora_weights: tuple[jax.Array, jax.Array] | None) -> jax.Array:
        return jnp.dot(x, w.astype(x.dtype)) + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))