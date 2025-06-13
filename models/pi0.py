import dataclasses
from typing import Sequence
from typing_extensions import Self

import jax
import jax.numpy as jnp
import jaxtyping
import equinox
import einops

from models import gemma, siglip

def make_attn_mask(input_mask: jaxtyping.Bool[jax.Array, "t"], mask_ar: jaxtyping.Bool[jax.Array, "t"]) -> jaxtyping.Bool[jax.Array, "t t"]:
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[N] true if its part of the input, false if padding.
      mask_ar: bool[N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    input_mask = einops.rearrange(input_mask, "t -> 1 t")
    mask_ar = einops.rearrange(mask_ar, "t -> 1 t")
    
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    mask = jnp.logical_and(attn_mask, valid_mask)
    
    return einops.rearrange(mask, "1 ta tb -> ta tb")

def posemb_sincos(
    pos: jax.Array, embedding_dim: int, min_period: float, max_period: float
) -> jax.Array:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    pos = jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
    return pos

@dataclasses.dataclass
class Config:
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
    action_horizon: int = equinox.field(static=True)
    
    llm: gemma.Module = equinox.field(static=False)
    img: siglip.Module = equinox.field(static=False)
    state_proj: equinox.nn.Linear = equinox.field(static=False)
    action_in_proj: equinox.nn.Linear = equinox.field(static=False)
    action_time_mlp_in: equinox.nn.Linear = equinox.field(static=False)
    action_time_mlp_out: equinox.nn.Linear = equinox.field(static=False)
    action_out_proj: equinox.nn.Linear = equinox.field(static=False)
    
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
        
        self.action_horizon = config.action_horizon
        
    def cache(
        self,
        language_token_ids: jax.Array,
        image: jax.Array,
        key: jax.Array,
    ):
        rng = key
        language_tokens = einops.rearrange(self.llm.embed(einops.rearrange(language_token_ids, "t -> 1 t")), "1 t d -> t d")
        language_ar_mask = jnp.array([False] * language_tokens.shape[0])
        
        rng, key = jax.random.split(rng)
        image_tokens, _ = self.img(image, key=key)
        image_ar_mask = jnp.array([False] * image_tokens.shape[0])
        
        tokens, _ = einops.pack([image_tokens, language_tokens], "* d")
        ar_mask, _ = einops.pack([image_ar_mask, language_ar_mask], "*")
        valid_mask = jnp.ones_like(ar_mask)
        attn_mask = make_attn_mask(valid_mask, ar_mask)
        positions = jnp.cumsum(valid_mask) - 1
        
        tokens = einops.rearrange(tokens, "t d -> 1 t d")
        attn_mask = einops.rearrange(attn_mask, "ta tb -> 1 ta tb")
        positions = einops.rearrange(positions, "t -> 1 t")
        _, kv_cache = self.llm([tokens, None], mask=attn_mask, positions=positions)
        return kv_cache, ar_mask
        
    def __call__(
        self, 
        context_cache: jax.Array,
        context_ar_mask: jax.Array,
        state: jax.Array, 
        actions: jax.Array,
        timestep: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        state_token = self.state_proj(state)
        state_ar_mask = jnp.array([False])
        
        action_tokens = equinox.filter_vmap(self.action_in_proj)(actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = einops.repeat(time_emb, "a d -> (t a) d", t=self.action_horizon)
        action_time_tokens, _ = einops.pack([action_tokens, time_emb], "t *")
        action_time_tokens = equinox.filter_vmap(self.action_time_mlp_in)(action_time_tokens)
        action_time_tokens = equinox.filter_vmap(jax.nn.swish)(action_time_tokens)
        action_time_tokens = equinox.filter_vmap(self.action_time_mlp_out)(action_time_tokens)
        action_time_ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        
        output_tokens, output_token_shapes = einops.pack([einops.rearrange(state_token, "d -> 1 d"), action_time_tokens], "* d")
        output_ar_mask, _ = einops.pack([state_ar_mask, action_time_ar_mask], "*")
        output_valid_mask = jnp.ones_like(output_ar_mask)
        output_attn_mask = make_attn_mask(output_valid_mask, output_ar_mask)
        
        context_attn_mask = einops.repeat(jnp.ones_like(context_ar_mask), "a -> b a", b=output_tokens.shape[0])
        
        attn_mask, _ = einops.pack([context_attn_mask, output_attn_mask], "a *")
        positions = jnp.sum(jnp.ones_like(context_ar_mask)) + jnp.cumsum(jnp.ones_like(output_ar_mask)) - 1

        output_tokens = einops.rearrange(output_tokens, "t d -> 1 t d")
        attn_mask = einops.rearrange(attn_mask, "ta tb -> 1 ta tb")
        positions = einops.rearrange(positions, "t -> 1 t")
        (_, output_tokens), _ = self.llm([None, output_tokens], mask=attn_mask, positions=positions, kv_cache=context_cache)
        output_tokens = einops.rearrange(output_tokens, "1 t d -> t d")
        _, action_tokens = einops.unpack(output_tokens, output_token_shapes, "* d")
        actions = equinox.filter_vmap(self.action_out_proj)(action_tokens)
        return actions
    
    def sample_actions(
        self,
        language_token_ids: jax.Array,
        image: jax.Array,
        state: jax.Array,
        key: jax.Array,
        sample_steps: int = 10,
        cfg_scale: float = 1.0
    ) -> jax.Array:
        rng = key
        rng, key = jax.random.split(rng)
        gc_context_cache, gc_context_ar_mask = self.cache(language_token_ids, image, key)
        null_context_cache, null_context_ar_mask = self.cache(jnp.array([1]), image, key)
        noisy_actions = jax.random.normal(rng, (self.action_horizon, state.shape[-1]))
        dt = jax.numpy.array([1 / sample_steps])
        def body(carry, t):
            noisy_actions, rng = carry
            time = 1.0 - (dt * t)
            rng, key = jax.random.split(rng)
            gc_action_flow = self(gc_context_cache, gc_context_ar_mask, state, noisy_actions, time, key)
            rng, key = jax.random.split(rng)
            null_action_flow = self(null_context_cache, null_context_ar_mask, state, noisy_actions, time, key)
            deviation = jax.numpy.linalg.norm(gc_action_flow - null_action_flow)
            action_flow = null_action_flow + cfg_scale * (gc_action_flow - null_action_flow)
            noisy_actions = noisy_actions - (dt * action_flow)
            return (noisy_actions, rng), deviation
        (noisy_actions, _), deviations = jax.lax.scan(body, (noisy_actions, rng), jnp.arange(sample_steps))
        deviation = jax.numpy.mean(deviations)
        return noisy_actions, deviation
    
def sample_actions(
    positive_model: Pi0,
    negative_model: Pi0,
    language_token_ids: jax.Array,
    image: jax.Array,
    state: jax.Array,
    key: jax.Array,
    sample_steps: int,
    cfg_scale: float,
) -> jax.Array:
    rng = key
    rng, key = jax.random.split(rng)
    positive_context_cache, positive_context_ar_mask = positive_model.cache(language_token_ids, image, key)
    negative_context_cache, negative_context_ar_mask = negative_model.cache(language_token_ids, image, key)
    assert positive_model.action_horizon == negative_model.action_horizon, "Positive and negative models must have the same action horizon"
    noisy_actions = jax.random.normal(rng, (positive_model.action_horizon, state.shape[-1]))
    dt = jax.numpy.array([1 / sample_steps])
    def body(carry, t):
        noisy_actions, rng = carry
        time = 1.0 - (dt * t)
        rng, key = jax.random.split(rng)
        positive_action_flow = positive_model(positive_context_cache, positive_context_ar_mask, state, noisy_actions, time, key)
        rng, key = jax.random.split(rng)
        negative_action_flow = negative_model(negative_context_cache, negative_context_ar_mask, state, noisy_actions, time, key)
        deviation = jax.numpy.linalg.norm(positive_action_flow - negative_action_flow)
        action_flow = negative_action_flow + cfg_scale * (positive_action_flow - negative_action_flow)
        noisy_actions = noisy_actions - (dt * action_flow)
        return (noisy_actions, rng), deviation
    (noisy_actions, _), deviations = jax.lax.scan(body, (noisy_actions, rng), jnp.arange(sample_steps + 1))
    deviation = jax.numpy.mean(deviations)
    return noisy_actions, deviation
    
def load(params, config: Config = Config.default(), dtype: str = "float32", dropout: float = 0.0) -> Pi0:
    pi0_model = Pi0(config, rng = jax.random.PRNGKey(0), dtype = dtype, dropout = dropout)
    gemma_model = gemma.load(params["PaliGemma"]["llm"], configs=[config.llm_config, config.action_expert_config], dtype = dtype, dropout = dropout)
    img_model = siglip.load(params["PaliGemma"]["img"], config=config.img_config, dtype = dtype, dropout = dropout)
    
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
    
    params_to_replace = (
        gemma_model,
        img_model,
    ) + tuple([jax.numpy.array(x) for x in [
        params["action_in_proj"]["bias"],
        einops.rearrange(params["action_in_proj"]["kernel"], "d_in d_out -> d_out d_in"),
        params["action_out_proj"]["bias"],
        einops.rearrange(params["action_out_proj"]["kernel"], "d_in d_out -> d_out d_in"),
        params["action_time_mlp_in"]["bias"],
        einops.rearrange(params["action_time_mlp_in"]["kernel"], "d_in d_out -> d_out d_in"),
        params["action_time_mlp_out"]["bias"],
        einops.rearrange(params["action_time_mlp_out"]["kernel"], "d_in d_out -> d_out d_in"),
        params["state_proj"]["bias"],
        einops.rearrange(params["state_proj"]["kernel"], "d_in d_out -> d_out d_in"),
    ]])
     
    replaced_pi0_model = equinox.tree_at(
        where = where_replace,
        pytree = pi0_model,
        replace = params_to_replace
    )
    return replaced_pi0_model