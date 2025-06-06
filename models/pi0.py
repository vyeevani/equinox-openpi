from typing import Sequence
from typing_extensions import Self

import jax
import equinox
import einops

from models import gemma, siglip

class Config(equinox.Module):
    llm_config: gemma.Config = equinox.field(static=True)
    action_expert_config: gemma.Config = equinox.field(static=True)
    img_config: siglip.Config = equinox.field(static=True)
    action_dim: int = equinox.field(static=True)
    action_horizon: int = equinox.field(static=True)
    max_token_len: int = equinox.field(static=True)
    dtype: str = equinox.field(static=True)
    
    @classmethod
    def default(cls) -> Self:
        return cls(
            llm_config = gemma.Config.get_variant("gemma_2b"),
            action_expert_config = gemma.Config.get_variant("gemma_300m"),
            img_config = siglip.Config.get_variant("So400m/14"),
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            dtype = "float32"
        )

class Pi0(equinox.Module):
    llm: gemma.Module
    img: siglip.Module
    state_proj: equinox.nn.Linear
    action_in_proj: equinox.nn.Linear
    action_time_mlp_in: equinox.nn.Linear
    action_time_mlp_out: equinox.nn.Linear
    action_out_proj: equinox.nn.Linear
    
    def __init__(self, config: Config, rng: jax.Array, dtype: str = "float32", dropout: float = 0.0):
        self.llm = gemma.Module(configs = [config.llm_config, config.action_expert_config], rng = rng, embed_dtype=dtype, dropout=dropout)
        self.img = siglip.Module(config = config.img_config, rng = rng, dtype = dtype, dropout=dropout)
        rng, key = jax.random.split(rng)
        self.state_proj = equinox.nn.Linear(in_features = config.action_dim, out_features = config.action_expert_config.width, key = key)
        rng, key = jax.random.split(rng)
        self.action_in_proj = equinox.nn.Linear(in_features = config.action_dim, out_features = config.action_expert_config.width, key = key)
        rng, key = jax.random.split(rng)
        self.action_time_mlp_in = equinox.nn.Linear(in_features = 2 * config.action_expert_config.width, out_features = config.action_expert_config.width, key = key)
        rng, key = jax.random.split(rng)
        self.action_time_mlp_out = equinox.nn.Linear(in_features = config.action_expert_config.width, out_features = config.action_expert_config.width, key = key)
        rng, key = jax.random.split(rng)
        self.action_out_proj = equinox.nn.Linear(in_features = config.action_expert_config.width, out_features = config.action_dim, key = key)
        
    def __call__(self, language_tokens: jax.Array, image: jax.Array, state: jax.Array, noisy_actions: jax.Array, timestep: jax.Array) -> jax.Array:
        pass
    
def load(params, config: Config = Config.default(), dtype: str = "float32", dropout: float = 0.0) -> Pi0:
    pi0_model = Pi0(config, rng = jax.random.PRNGKey(0), dtype = dtype, dropout = dropout)
    gemma_model = gemma.load(params["PaliGemma"]["llm"], dtype = dtype, dropout = dropout)
    img_model = siglip.load(params["PaliGemma"]["img"], dtype = dtype, dropout = dropout)
    
    def where_replace(x: Pi0):
        return (
            x.llm,
            x.img,
            x.action_in_proj.bias,
            x.action_in_proj.weight,
            x.action_out_proj.bias,
            x.action_out_proj.weight,
            x.action_time_mlp_in.bias,
            x.action_time_mlp_in.weight,
            x.action_time_mlp_out.bias,
            x.action_time_mlp_out.weight,
            x.state_proj.bias,
            x.state_proj.weight,
        )
    
    params_to_replace = [
        gemma_model,
        img_model,
        pi0_model.action_in_proj.bias,
        pi0_model.action_in_proj.weight,
        pi0_model.action_out_proj.bias,
        pi0_model.action_out_proj.weight,
        pi0_model.action_time_mlp_in.bias,
        pi0_model.action_time_mlp_in.weight,
        pi0_model.action_time_mlp_out.bias,
        pi0_model.action_time_mlp_out.weight,
        pi0_model.state_proj.bias,
        pi0_model.state_proj.weight,
    ]
        
    pi0_model = equinox.tree_at(
        where = where_replace,
        pytree = pi0_model,
        replace = params_to_replace
    )
    
    return pi0_model