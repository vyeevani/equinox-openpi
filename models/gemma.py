from typing import Sequence, TypeAlias, List
from typing_extensions import Self

import jax
import jax.numpy as jnp
import jaxtyping
import equinox 

import jax_utils
import equinox_utils

from models.attention import Attention
from models.feedfoward import FeedForward
import models.lora as lora
from models.rms_norm import RMSNorm
from models.embedder import Embedder

class Config(equinox.Module):
    width: int = equinox.field(static=True)
    depth: int = equinox.field(static=True)
    mlp_dim: int = equinox.field(static=True)
    num_heads: int = equinox.field(static=True)
    num_kv_heads: int = equinox.field(static=True)
    head_dim: int = equinox.field(static=True)
    lora_configs: dict[str, lora.Config] = equinox.field(static=True)
    
    @classmethod
    def get_variant(cls, variant: str) -> Self:
        """Returns config for specified gemma variant."""
        if variant == "dummy":
            return Config(
                width=64,
                depth=4,
                mlp_dim=128,
                num_heads=8,
                num_kv_heads=1,
                head_dim=16,
                lora_configs={},
            )
        if variant == "gemma_300m":
            # 311M params
            return Config(
                width=1024,
                depth=18,
                mlp_dim=4096,
                num_heads=8,
                num_kv_heads=1,
                head_dim=256,
                lora_configs={},
            )
        if variant == "gemma_2b":
            return Config(
                width=2048,
                depth=18,
                mlp_dim=16_384,
                num_heads=8,
                num_kv_heads=1,
                head_dim=256,
                lora_configs={},
            )
        if variant == "gemma_2b_lora":
            return Config(
                width=2048,
                depth=18,
                mlp_dim=16_384,
                num_heads=8,
                num_kv_heads=1,
                head_dim=256,
                lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
            )
        if variant == "gemma_300m_lora":
            # 311M params
            return Config(
                width=1024,
                depth=18,
                mlp_dim=4096,
                num_heads=8,
                num_kv_heads=1,
                head_dim=256,
                lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
            )
        raise ValueError(f"Unknown variant: {variant}")

class Block(equinox.Module):
    drop: equinox.Module = equinox.field(static=False)
    attn: Attention = equinox.field(static=False)
    pre_attention_norm: RMSNorm = equinox.field(static=False)
    expert_pre_attention_norms: Sequence[RMSNorm] = equinox.field(static=False)
    pre_ffw_norm: RMSNorm = equinox.field(static=False)
    expert_pre_ffw_norms: Sequence[RMSNorm] = equinox.field(static=False)
    mlp: FeedForward = equinox.field(static=False)
    expert_mlps: Sequence[FeedForward] = equinox.field(static=False)

    def __init__(
        self,
        configs: Sequence[Config],
        dropout: float = 0.0,
        rng: jax.Array = jax.random.PRNGKey(0),
    ):
        self.drop = equinox.nn.Dropout(dropout) if dropout else equinox.nn.Identity()
        self.attn = Attention(configs=configs) # duck typing the configs
        
        pre_attention_norms = []
        mlps = []
        pre_ffw_norms = []
        for config in configs:
            pre_attention_norms.append(RMSNorm(config.width))
            pre_ffw_norms.append(RMSNorm(config.width))
            rng, key = jax.random.split(rng)
            if config.lora_configs.get("ffn") is not None:
                mlps.append(lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    config=config.lora_configs.get("ffn"),
                    rng=key,
                ))
            else:
                mlps.append(FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    rng=key,
                ))
                
        self.pre_attention_norm = pre_attention_norms.pop(0)
        self.expert_pre_attention_norms = pre_attention_norms
        self.pre_ffw_norm = pre_ffw_norms.pop(0)
        self.expert_pre_ffw_norms = pre_ffw_norms
        self.mlp = mlps.pop(0)
        self.expert_mlps = mlps
        
    def __call__(self, xs, kv_cache, positions, attn_mask, rng: jax.Array = jax.random.PRNGKey(0)):
        pre_attn = []
        pre_attention_norms = [self.pre_attention_norm, *self.expert_pre_attention_norms]
        for x, pre_attention_norm in zip(xs, pre_attention_norms, strict=True):
            if x is not None:
                x = pre_attention_norm(x)
            pre_attn.append(x)
        
        post_attn, kv_cache = self.attn(pre_attn, positions, attn_mask, kv_cache)
        rng, key = jax.random.split(rng)
        post_attn = jax.tree.map(lambda x, key: self.drop(x, key=key), post_attn, jax_utils.random_tree_like(post_attn, key))
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)
        
        out = []
        mlps = [self.mlp, *self.expert_mlps]
        pre_ffw_norms = [self.pre_ffw_norm, *self.expert_pre_ffw_norms]
        for x, pre_ffw_norm, mlp in zip(xs, pre_ffw_norms, mlps, strict=True):
            if x is not None:
                x = pre_ffw_norm(x)
                x = mlp(x)
            out.append(x)
        rng, key = jax.random.split(rng)
        out = jax.tree.map(lambda x, key: self.drop(x, key=key), out, jax_utils.random_tree_like(out, key))
        xs = jax.tree.map(lambda x, y: x + y, xs, out)
        return xs, kv_cache
    
KVCache: TypeAlias = tuple[jaxtyping.Float[jaxtyping.Array, "l b _t _k _h"], jaxtyping.Float[jaxtyping.Array, "l b _t _v _h"]]
PALIGEMMA_VOCAB_SIZE = 257_152

class Module(equinox.Module):
    configs: Sequence[Config] = equinox.field(static=True)
    embed_dtype: str = equinox.field(static=True)
    embedder: Embedder = equinox.field(static=False)
    layers: Block = equinox.field(static=False)
    final_norm: RMSNorm = equinox.field(static=False)
    expert_final_norms: Sequence[RMSNorm] = equinox.field(static=False)
    
    def __init__(self, configs: Sequence[Config], embed_dtype: str, dropout: float = 0.0, rng: jax.Array = jax.random.PRNGKey(0)):
        self.configs = configs
        self.embed_dtype = embed_dtype
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=configs[0].width,
            rng=rng,
        )
        rng, key = jax.random.split(rng)
        self.layers = equinox.filter_vmap(lambda key: Block(configs=configs, dropout=dropout, rng=key))(jax.random.split(key, configs[0].depth))
        final_norms = [RMSNorm(config.width) for config in configs]
        self.final_norm = final_norms.pop(0)
        self.expert_final_norms = final_norms
    
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[jaxtyping.Float[jaxtyping.Array, "b _t _d"] | None],
        positions: jaxtyping.Int[jaxtyping.Array, "b t"],
        mask: jaxtyping.Bool[jaxtyping.Array, "b t s"],
        kv_cache: KVCache | None = None,
    ) -> tuple[Sequence[jaxtyping.Float[jaxtyping.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda x: x.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        def body(embedded, layer_and_kv_cache):
            layer, kv_cache = layer_and_kv_cache
            embedded, kv_cache = layer(embedded, kv_cache, positions, mask)
            return embedded, kv_cache
        embedded, kv_cache = equinox_utils.scan(body, embedded, (self.layers, kv_cache))
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
        final_norms = [self.final_norm, *self.expert_final_norms]
        return [f(e) if e is not None else e for f, e in zip(final_norms, embedded, strict=True)], kv_cache

    def embed(self, tokens: jaxtyping.Int[jaxtyping.Array, "b t"]) -> jaxtyping.Float[jaxtyping.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

def load(
    params,
    configs: Sequence[Config] = [Config.get_variant("gemma_2b")],
    dtype: str = "float32",
    dropout: float = 0.0,
) -> Module:
    model = Module(
        configs=configs,
        embed_dtype=dtype,
        dropout=dropout,
        rng=jax.random.PRNGKey(0),
    )
    def get_model_params(model: Module) -> List[jax.Array]:
        return (
            model.embedder.input_embedding,
            model.final_norm.scale,
            model.layers.attn.attn_vec_einsum.w,
            *[model.layers.attn.expert_attn_vec_einsums[i].w for i in range(len(configs) - 1)],
            model.layers.attn.kv_einsum.w,
            *[model.layers.attn.expert_kv_einsums[i].w for i in range(len(configs) - 1)],
            model.layers.attn.q_einsum.w,
            *[model.layers.attn.expert_q_einsums[i].w for i in range(len(configs) - 1)],
            model.layers.mlp.gating_einsum,
            *[model.layers.expert_mlps[i].gating_einsum for i in range(len(configs) - 1)],
            model.layers.mlp.linear,
            *[model.layers.expert_mlps[i].linear for i in range(len(configs) - 1)],
            model.layers.pre_attention_norm.scale,
            *[model.layers.expert_pre_attention_norms[i].scale for i in range(len(configs) - 1)],
            model.layers.pre_ffw_norm.scale,
            *[model.layers.expert_pre_ffw_norms[i].scale for i in range(len(configs) - 1)],
        )
    model: Module = equinox.tree_at(
        where=get_model_params,
        pytree=model,
        replace=tuple([jax.numpy.asarray(param) for param in [
            params["embedder"]["input_embedding"],
            params["final_norm"]["scale"],
            params["layers"]["attn"]["attn_vec_einsum"]["w"],
            *[params["layers"]["attn"][f"attn_vec_einsum_{i}"]["w"] for i in range(1, len(configs))],
            params["layers"]["attn"]["kv_einsum"]["w"],
            *[params["layers"]["attn"][f"kv_einsum_{i}"]["w"] for i in range(1, len(configs))],
            params["layers"]["attn"]["q_einsum"]["w"],
            *[params["layers"]["attn"][f"q_einsum_{i}"]["w"] for i in range(1, len(configs))],
            params["layers"]["mlp"]["gating_einsum"],
            *[params["layers"][f"mlp_{i}"]["gating_einsum"] for i in range(1, len(configs))],
            params["layers"]["mlp"]["linear"],
            *[params["layers"][f"mlp_{i}"]["linear"] for i in range(1, len(configs))],
            params["layers"]["pre_attention_norm"]["scale"],
            *[params["layers"][f"pre_attention_norm_{i}"]["scale"] for i in range(1, len(configs))],
            params["layers"]["pre_ffw_norm"]["scale"],
            *[params["layers"][f"pre_ffw_norm_{i}"]["scale"] for i in range(1, len(configs))],
        ]]),
    )
    return model