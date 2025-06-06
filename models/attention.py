import jax
import jax.numpy as jnp
import equinox
from typing import Sequence
import einops
import jax.nn

from models.einsum import Einsum
import models.lora as lora

def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)

class Config:
    width: int = equinox.field(static=True)
    num_heads: int = equinox.field(static=True)
    num_kv_heads: int = equinox.field(static=True)
    head_dim: int = equinox.field(static=True)
    lora_configs: dict[str, lora.Config] = equinox.field(static=True)

class Attention(equinox.Module):
    configs: Sequence[Config] = equinox.field(static=True)
    q_einsum: Einsum = equinox.field(static=False)
    kv_einsum: Einsum = equinox.field(static=False)
    expert_q_einsums: Sequence[Einsum] = equinox.field(static=False)
    expert_kv_einsums: Sequence[Einsum] = equinox.field(static=False)
    
    attn_vec_einsum: Einsum = equinox.field(static=False)
    expert_attn_vec_einsums: Sequence[Einsum] = equinox.field(static=False)
    
    
    def __init__(self, configs: Sequence[Config]):
        assert all(config.head_dim == configs[0].head_dim for config in configs)
        assert all(config.num_heads == configs[0].num_heads for config in configs)
        assert all(config.num_kv_heads == configs[0].num_kv_heads for config in configs)
        self.configs = configs
        q_einsums = []
        kv_einsums = []
        for config in configs:
            if config.lora_configs.get("attn") is not None:
                q_einsums.append(lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    eqn="BTD,NDH->BTNH",
                    init_fn=jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    config=config.lora_configs.get("attn"),
                ))
                kv_einsums.append(lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    eqn="BTD,2NDH->2BTNH",
                    init_fn=jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    config=config.lora_configs.get("attn"),
                ))
            else:
                q_einsums.append(Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    eqn="BTD,NDH->BTNH",
                    init_fn=jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                ))
                kv_einsums.append(Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    eqn="BTD,2NDH->2BTNH",
                    init_fn=jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                ))
        self.q_einsum = q_einsums.pop(0)
        self.kv_einsum = kv_einsums.pop(0)
        self.expert_q_einsums = q_einsums
        self.expert_kv_einsums = kv_einsums
    
        attn_vec_einsums = []
        for config in configs:
            if config.lora_configs.get("attn") is not None:
                attn_vec_einsums.append(lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    eqn="BTNH,NHD->BTD",
                    init_fn=jax.nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    config=config.lora_configs.get("attn"),
                ))
            else:
                attn_vec_einsums.append(Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    eqn="BTNH,NHD->BTD",
                    init_fn=jax.nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                ))
        self.attn_vec_einsum = attn_vec_einsums.pop(0)
        self.expert_attn_vec_einsums = attn_vec_einsums
        
    def __call__(self, xs, positions, attn_mask, kv_cache):
        dtype = next(x.dtype for x in xs if x is not None)
        qkvs = []
        q_einsums = [self.q_einsum, *self.expert_q_einsums]
        kv_einsums = [self.kv_einsum, *self.expert_kv_einsums]
        for x, q_einsum, kv_einsum in zip(xs, q_einsums, kv_einsums, strict=True):
            if x is None:
                continue
            q = q_einsum(x)
            k, v = kv_einsum(x)
            qkvs.append((q, k, v))
            
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5
        k = _apply_rope(k, positions=positions)
        assert q.dtype == k.dtype == v.dtype == dtype
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
        
        out = []
        start = 0
        attn_vec_einsums = [self.attn_vec_einsum, *self.expert_attn_vec_einsums]
        for x, attn_vec_einsum in zip(xs, attn_vec_einsums, strict=True):
            if x is not None:
                end = start + x.shape[1]
                out.append(attn_vec_einsum(encoded[:, start:end]))
                start = end
            else:
                out.append(None)
        return out, (k, v)