from typing import Sequence
from typing_extensions import Self
import jax
import jax.numpy as jnp
import equinox
import einops

import equinox_utils

def linear_init(linear: equinox.nn.Linear, rng: jax.Array, kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.lecun_normal(), bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros, dtype: str = "float32") -> equinox.nn.Linear:
    rng, key = jax.random.split(rng)
    weight = kernel_init(key, linear.weight.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    bias = bias_init(key, linear.bias.shape, dtype=dtype)
    linear = equinox.tree_at(
        where=lambda x: (x.weight, x.bias),
        pytree=linear,
        replace=(weight, bias)
    )
    return linear

def conv_init(conv: equinox.nn.Conv, rng: jax.Array, kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.lecun_normal(), bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros, dtype: str = "float32") -> equinox.nn.Conv:
    rng, key = jax.random.split(rng)
    weight = kernel_init(key, conv.weight.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    bias = bias_init(key, conv.bias.shape, dtype=dtype)
    conv = equinox.tree_at(
        where=lambda x: (x.weight, x.bias),
        pytree=conv,
        replace=(weight, bias)
    )
    return conv
    
def mha_init(attn: equinox.nn.MultiheadAttention, rng: jax.Array, kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.lecun_normal(), bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros, dtype: str = "float32") -> equinox.nn.MultiheadAttention:
    rng, key = jax.random.split(rng)
    query_proj_weight = kernel_init(key, attn.query_proj.weight.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    query_proj_bias = bias_init(key, attn.query_proj.bias.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    key_proj_weight = kernel_init(key, attn.key_proj.weight.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    key_proj_bias = bias_init(key, attn.key_proj.bias.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    value_proj_weight = kernel_init(key, attn.value_proj.weight.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    value_proj_bias = bias_init(key, attn.value_proj.bias.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    output_proj_weight = kernel_init(key, attn.output_proj.weight.shape, dtype=dtype)
    rng, key = jax.random.split(rng)
    output_proj_bias = bias_init(key, attn.output_proj.bias.shape, dtype=dtype)
    attn = equinox.tree_at(
        where=lambda x: (
            x.query_proj.weight,
            x.query_proj.bias,
            x.key_proj.weight,
            x.key_proj.bias,
            x.value_proj.weight,
            x.value_proj.bias,
            x.output_proj.weight,
            x.output_proj.bias,
        ),
        pytree=attn,
        replace=(
            query_proj_weight,
            query_proj_bias,
            key_proj_weight,
            key_proj_bias,
            value_proj_weight,
            value_proj_bias,
            output_proj_weight,
            output_proj_bias,
        )
    )
    return attn
    

class MlpBlock(equinox.Module):
    layer_1: equinox.nn.Linear = equinox.field(static=False)
    dropout: equinox.nn.Dropout = equinox.field(static=False)
    layer_2: equinox.nn.Linear = equinox.field(static=False)
    
    def __init__(self, input_dim: int, rng: jax.Array, output_dim: int, mlp_dim: int | None = None, dropout: float = 0.0, dtype: str = "float32"):
        hidden_dim = mlp_dim or 4 * input_dim
        kernel_init = jax.nn.initializers.xavier_uniform()
        bias_init = jax.nn.initializers.normal(stddev=1e-6)
        self.layer_1 = linear_init(equinox.nn.Linear(input_dim, hidden_dim, dtype=dtype, key=jax.random.key(0)), rng, kernel_init=kernel_init, bias_init=bias_init, dtype=dtype)
        self.dropout = equinox.nn.Dropout(dropout)
        self.layer_2 = linear_init(equinox.nn.Linear(hidden_dim, output_dim, dtype=dtype, key=jax.random.key(0)), rng, kernel_init=kernel_init, bias_init=bias_init, dtype=dtype)
    def __call__(self, x: jax.Array, key: jax.Array) -> jax.Array:
        assert x.shape[0] == self.layer_2.out_features, f"Input dimension {x.shape[0]} does not match output dimension {self.layer_2.out_features}"
        rng = key
        rng, key = jax.random.split(rng)
        x = self.layer_1(x, key=key)
        x = jax.nn.gelu(x)
        rng, key = jax.random.split(rng)
        x = self.dropout(x, key=key)
        rng, key = jax.random.split(rng)
        x = self.layer_2(x, key=key)
        return x

class Encoder1DBlock(equinox.Module):
    layer_norm_sa: equinox.nn.LayerNorm = equinox.field(static=False)
    sa: equinox.nn.MultiheadAttention = equinox.field(static=False)
    dropout_sa: equinox.nn.Dropout = equinox.field(static=False)
    layer_norm_mlp: equinox.nn.LayerNorm = equinox.field(static=False)
    mlp: MlpBlock = equinox.field(static=False)
    dropout_mlp: equinox.nn.Dropout = equinox.field(static=False)
    def __init__(self, dim: int, rng: jax.Array, num_heads: int = 12, mlp_dim: int | None = None, dropout: float = 0.0, dtype: str = "float32", ):
        self.layer_norm_sa = equinox.nn.LayerNorm(dim, dtype=dtype)
        self.sa = mha_init(
            equinox.nn.MultiheadAttention(
                num_heads=num_heads,
                query_size=dim,
                dtype=dtype,
                use_key_bias=True,
                use_value_bias=True,
                use_query_bias=True,
                use_output_bias=True,
                key=jax.random.key(0),
            ), 
            rng, 
            kernel_init=jax.nn.initializers.xavier_uniform(), 
            bias_init=jax.nn.initializers.normal(stddev=1e-6), 
            dtype=dtype
        )
        self.dropout_sa = equinox.nn.Dropout(dropout)
        
        self.layer_norm_mlp = equinox.nn.LayerNorm(dim, dtype=dtype)
        self.mlp = MlpBlock(dim, rng, dim, mlp_dim=mlp_dim, dropout=dropout, dtype=dtype)
        self.dropout_mlp = equinox.nn.Dropout(dropout)
    def __call__(self, x: jax.Array, key: jax.Array) -> jax.Array:
        rng = key
        rng, key = jax.random.split(rng)
        out = {}
        y = equinox.filter_vmap(self.layer_norm_sa)(x)
        out["sa_ln"] = y
        rng, key = jax.random.split(rng)
        y = self.sa(y, y, y, key=key)
        out["sa"] = y
        rng, key = jax.random.split(rng)
        y = self.dropout_sa(y, key=key)
        x = x + y
        out["+sa"] = x
        
        y = equinox.filter_vmap(self.layer_norm_mlp)(x)
        rng, key = jax.random.split(rng)
        y = equinox.filter_vmap(self.mlp)(y, jax.random.split(key, y.shape[0]))
        out["mlp"] = y
        rng, key = jax.random.split(rng)
        y = self.dropout_mlp(y, key=key)
        x = x + y
        out["+mlp"] = x
        return x, out

class Encoder(equinox.Module):
    encoder_blocks: Encoder1DBlock = equinox.field(static=False)
    encoder_norm: equinox.nn.LayerNorm = equinox.field(static=False)
    depth: int = equinox.field(static=True)
    def __init__(self, dim: int, rng: jax.Array, depth: int, mlp_dim: int | None = None, num_heads: int = 12, dropout: float = 0.0, dtype: str = "float32"):
        rng, key = jax.random.split(rng)
        self.encoder_blocks = equinox.filter_vmap(lambda key: Encoder1DBlock(dim, key, num_heads, mlp_dim, dropout, dtype))(jax.random.split(key, depth))
        self.encoder_norm = equinox.nn.LayerNorm(dim, dtype=dtype)
        self.depth = depth
    def __call__(self, x, key: jax.Array):
        rng = key
        out = {}
        def body(x_and_rng, encoder):
            x, rng = x_and_rng
            rng, key = jax.random.split(rng)
            x, out = encoder(x, key=key)
            return (x, rng), out
        rng, key = jax.random.split(rng)
        (x, _), scan_out = equinox_utils.scan(body, (x, key), self.encoder_blocks)
        for lyr in range(self.depth):
            out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        out["pre_ln"] = x
        x = equinox.filter_vmap(self.encoder_norm)(x)
        return x, out
    
class Module(equinox.Module):
    embedding: equinox.nn.Conv = equinox.field(static=False)
    posemb: jax.Array = equinox.field(static=False)
    Transformer: Encoder = equinox.field(static=False)
    head: equinox.nn.Linear = equinox.field(static=False)
    dropout: equinox.nn.Dropout = equinox.field(static=False)
    dtype_mm: str = equinox.field(static=True)
    
    def __init__(
        self, 
        image_height: int,
        image_width: int,
        image_channels: int,
        rng: jax.Array, 
        num_classes: int | None = None, 
        patch_size: Sequence[int] = (16, 16),
        width: int = 768,
        depth: int = 12,
        mlp_dim: int | None = None,
        num_heads: int = 12,
        dropout: float = 0.0,
        dtype: str = "float32",
    ):
        rng, key = jax.random.split(rng)
        self.embedding = conv_init(equinox.nn.Conv(
            num_spatial_dims=2,
            in_channels=image_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            padding="VALID",
            dtype=dtype,
            key=jax.random.key(0),
        ), rng, dtype=dtype)
        rng, key = jax.random.split(rng)
        self.posemb = jax.nn.initializers.normal(stddev=1 / jnp.sqrt(width))(key, (int(image_height * image_width / (patch_size[0] * patch_size[1])), width), dtype=jnp.float32)
        rng, key = jax.random.split(rng)
        self.Transformer = Encoder(
            dim=width,
            rng=key,
            depth=depth,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout=dropout,
            dtype=dtype,
        )
        head = equinox.nn.Linear(width, num_classes, dtype=dtype, key=jax.random.key(0))
        self.head = linear_init(head, rng, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros, dtype=dtype)
        self.dropout = equinox.nn.Dropout(dropout)
        self.dtype_mm = dtype
    def __call__(self, image: jax.Array, key: jax.Array) -> jax.Array:
        x = einops.rearrange(image, "h w c -> c h w")
        rng = key
        rng, key = jax.random.split(rng)
        out = {}
        x = self.embedding(x, key=key)
        x = einops.rearrange(x, "c h w -> h w c")
        out["stem"] = x
        x = einops.rearrange(x, "h w c -> (h w) c")
        
        x = x + self.posemb
        out["with_posemb"] = x
        rng, key = jax.random.split(rng)
        x = self.dropout(x, key=key)
        x = x.astype(self.dtype_mm)
        rng, key = jax.random.split(rng)
        x, out["encoder"] = self.Transformer(x, key=key)
        out["encoded"] = x
        x = equinox.filter_vmap(self.head)(x)
        return x, out
    
def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
    
class Config(equinox.Module):
    variant: str = equinox.field(static=True)
    image_height: int = equinox.field(static=True)
    image_width: int = equinox.field(static=True)
    image_channels: int = equinox.field(static=True)
    num_classes: int = equinox.field(static=True)
    dtype: str = equinox.field(static=True)
    
    @classmethod
    def default(cls, ) -> Self:
        return cls(
            variant="So400m/14",
            image_height=224,
            image_width=224,
            image_channels=3,
            num_classes=2048,
            dtype="float32"
        )

def load(
    img_params,
    config: Config = Config.default(),
) -> Module:
    params = decode_variant(config.variant)
    model = Module(
        image_height=config.image_height,
        image_width=config.image_width,
        image_channels=config.image_channels,
        rng=jax.random.PRNGKey(0),
        num_classes=config.num_classes,
        **params,
        dtype=config.dtype
    )
    
    # flax -> equinox: all kernels have their dimensions swapped because of the different conventions
    replace_params = tuple([jax.numpy.asarray(param) for param in [
        img_params["Transformer"]["encoder_norm"]["bias"],
        img_params["Transformer"]["encoder_norm"]["scale"],
        
        img_params["Transformer"]["encoderblock"]["LayerNorm_0"]["bias"],
        img_params["Transformer"]["encoderblock"]["LayerNorm_0"]["scale"],
        img_params["Transformer"]["encoderblock"]["LayerNorm_1"]["bias"],
        img_params["Transformer"]["encoderblock"]["LayerNorm_1"]["scale"],
        
        img_params["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_0"]["bias"],
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_0"]["kernel"], "a b c -> a c b"),
        img_params["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_1"]["bias"],
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MlpBlock_0"]["Dense_1"]["kernel"], "a b c -> a c b"),
        
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["query"]["bias"], "a b c -> a (b c)"),
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["query"]["kernel"], "a b c d -> a (c d) b"),
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["key"]["bias"], "a b c -> a (b c)"),
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["key"]["kernel"], "a b c d -> a (c d) b"),
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["value"]["bias"], "a b c -> a (b c)"),
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["value"]["kernel"], "a b c d -> a (c d) b"),
        
        img_params["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["out"]["bias"],
        einops.rearrange(img_params["Transformer"]["encoderblock"]["MultiHeadDotProductAttention_0"]["out"]["kernel"], "a b c d -> a d (b c)"),
        
        einops.rearrange(img_params["embedding"]["bias"], "d -> d 1 1"),
        einops.rearrange(img_params["embedding"]["kernel"], "a b c d -> d c a b"),
        
        img_params["head"]["bias"],
        einops.rearrange(img_params["head"]["kernel"], "a b -> b a"),
        einops.rearrange(img_params["pos_embedding"], "1 a b -> a b"),
    ]])
    
    def where_replace(x: Module):
        return (
            x.Transformer.encoder_norm.bias,
            x.Transformer.encoder_norm.weight,
            
            x.Transformer.encoder_blocks.layer_norm_sa.bias,
            x.Transformer.encoder_blocks.layer_norm_sa.weight,
            x.Transformer.encoder_blocks.layer_norm_mlp.bias,
            x.Transformer.encoder_blocks.layer_norm_mlp.weight,
            
            x.Transformer.encoder_blocks.mlp.layer_1.bias,
            x.Transformer.encoder_blocks.mlp.layer_1.weight,
            x.Transformer.encoder_blocks.mlp.layer_2.bias,
            x.Transformer.encoder_blocks.mlp.layer_2.weight,
            
            x.Transformer.encoder_blocks.sa.query_proj.bias,
            x.Transformer.encoder_blocks.sa.query_proj.weight,
            x.Transformer.encoder_blocks.sa.key_proj.bias,
            x.Transformer.encoder_blocks.sa.key_proj.weight,
            x.Transformer.encoder_blocks.sa.value_proj.bias,
            x.Transformer.encoder_blocks.sa.value_proj.weight,
            
            x.Transformer.encoder_blocks.sa.output_proj.bias,
            x.Transformer.encoder_blocks.sa.output_proj.weight,
            
            x.embedding.bias,
            x.embedding.weight,
            
            x.head.bias,
            x.head.weight,
            x.posemb,
        )
    model = equinox.tree_at(where=where_replace, pytree=model, replace=replace_params)
    return model