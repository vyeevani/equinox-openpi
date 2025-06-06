from typing import Sequence
from typing_extensions import Self

import jax
import equinox

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
            img_config = siglip.Config.default(),
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
    
    def __init__(self, config: Config):
        self.llm = gemma.Module(configs = [config.llm_config, config.action_expert_config])
        self.img = siglip.Module(config = config.img_config)
        self.state_proj = equinox.nn.Linear(in_features = config.action_dim, out_features = config.action_expert_config.width)
        self.action_in_proj = equinox.nn.Linear(in_features = config.action_dim, out_features = config.action_expert_config.width)
        self.action_time_mlp_in = equinox.nn.Linear(in_features = 2 * config.action_expert_config.width, out_features = config.action_expert_config.width)
        self.action_time_mlp_out = equinox.nn.Linear(in_features = config.action_expert_config.width, out_features = config.action_expert_config.width)
        self.action_out_proj = equinox.nn.Linear(in_features = config.action_expert_config.width, out_features = config.action_dim)
        
    def __call__(self, language_tokens: jax.Array, image: jax.Array, state: jax.Array, noisy_actions: jax.Array, timestep: jax.Array) -> jax.Array:
        pass
    
# def load(llm_params, img_params, config: Config = Config.default()) -> Pi0:
#     llm = gemma.load(llm_params)
#     img = siglip.load(img_params)
#     return Pi0(config)