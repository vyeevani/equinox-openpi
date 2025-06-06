from typing import List
import urllib.request
import pathlib
from PIL import Image

import jax
import jax.numpy as jnp
import jaxtyping
import equinox
import numpy as np
import flax
import einops

import models.siglip as siglip
import models.gemma as gemma
from tokenizer import Tokenizer

def load_siglip(img_params) -> siglip.Module:
    params = siglip.decode_variant("So400m/14")
    model = siglip.Module(
        image_height=224,
        image_width=224,
        image_channels=3,
        rng=jax.random.PRNGKey(0),
        num_classes=2048,
        **params,
        dtype="float32"
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
    
    def where_replace(x: siglip.Module):
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

def load_gemma(llm_params) -> gemma.Module:
    config = gemma.Config(
        width=2048,
        depth=18,
        mlp_dim=16_384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        lora_configs={},
    )

    model = gemma.Module(
        configs=[config],
        embed_dtype="bfloat16",
        # embed_dtype="float32",
        dropout=0.0,
        rng=jax.random.PRNGKey(0),
    )

    def get_model_params(model: gemma.Module) -> List[jax.Array]:
        return (
            model.embedder.input_embedding,
            model.final_norm.scale,
            model.layers.attn.attn_vec_einsum.w,
            model.layers.attn.kv_einsum.w,
            model.layers.attn.q_einsum.w,
            model.layers.mlp.gating_einsum,
            model.layers.mlp.linear,
            model.layers.pre_attention_norm.scale,
            model.layers.pre_ffw_norm.scale,
        )
    model = equinox.tree_at(
        where=get_model_params,
        pytree=model,
        replace=tuple([jax.numpy.asarray(param) for param in [
            llm_params["embedder"]["input_embedding"],
            llm_params["final_norm"]["scale"],
            llm_params["layers"]["attn"]["attn_vec_einsum"]["w"],
            llm_params["layers"]["attn"]["kv_einsum"]["w"],
            llm_params["layers"]["attn"]["q_einsum"]["w"],
            llm_params["layers"]["mlp"]["gating_einsum"],
            llm_params["layers"]["mlp"]["linear"],
            llm_params["layers"]["pre_attention_norm"]["scale"],
            llm_params["layers"]["pre_ffw_norm"]["scale"],
        ]]),
    )
    return model

def load_model():
    model_path = pathlib.Path("pt_224.npz")
    if not model_path.exists():
        with urllib.request.urlopen("https://storage.googleapis.com/vertex-model-garden-paligemma-us/paligemma/pt_224.npz") as f:
            with open(model_path, "wb") as out:
                out.write(f.read())
    with open(model_path, "rb") as f:
        model_params = dict(np.load(f, allow_pickle=True))
    params = flax.traverse_util.unflatten_dict(model_params, sep="/")["params"]
    img_params = params["img"]
    llm_params = params["llm"]
    
    siglip_model = load_siglip(img_params)
    gemma_model = load_gemma(llm_params)
    
    return siglip_model, gemma_model

def make_attn_mask(input_mask: jaxtyping.Bool[jax.Array, "b t"], mask_ar: jaxtyping.Bool[jax.Array, "b t"]) -> jaxtyping.Bool[jax.Array, "b t t"]:
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
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

if __name__ == "__main__":
    import requests
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = np.array(Image.open(requests.get(url, stream=True).raw))
    Image.fromarray(image).save("car.png")
    image = jax.image.resize(image, (224, 224, 3), "bilinear")
    Image.fromarray(np.array(image.astype(np.uint8))).save("processed_car.png")
    prompt = "What is this a photo of?"
    
    siglip_model, gemma_model = load_model()
    siglip_model = equinox.nn.inference_mode(siglip_model)
    gemma_model = equinox.nn.inference_mode(gemma_model)
    tokenizer = Tokenizer()
    
    image = image.astype(jnp.float32) / 255.0 * 2.0 - 1.0
    image_tokens, _ = siglip_model(image, jax.random.key(0))
    image_mask = jnp.ones((image_tokens.shape[0],), dtype=jnp.bool_)
    
    prompt_token_ids, prompt_mask = tokenizer.tokenize(prompt)
    prompt_tokens = gemma_model.embed(prompt_token_ids)
    
    prefix_tokens, _ = einops.pack([image_tokens, prompt_tokens], "* d")
    prefix_mask, _ = einops.pack([image_mask, prompt_mask], "*")
    len_prefix_tokens = jnp.sum(prefix_mask)
    positions = einops.rearrange(jnp.arange(prefix_tokens.shape[0]), "... -> 1 ...")
    prefix_tokens = einops.rearrange(prefix_tokens, "... -> 1 ...")
    prefix_tokens = jax.numpy.array(prefix_tokens)
    prefix_mask = einops.rearrange(jax.numpy.array(prefix_mask), "... -> 1 ...")
    
    suffix_token_ids = []
    
    for i in range(10):
        attn_mask = make_attn_mask(jax.numpy.ones_like(prefix_mask), prefix_mask == False)
        suffix_tokens_logits, _ = gemma_model([prefix_tokens], positions, attn_mask)
        new_suffix_token_logit = suffix_tokens_logits[0][0, len_prefix_tokens - 1 + i, :]
        new_suffix_token_logits = gemma_model.embedder.decode(new_suffix_token_logit)
        new_suffix_token_id = jnp.argmax(new_suffix_token_logits)
        new_suffix_token = gemma_model.embed(jnp.array(new_suffix_token_id))
        if new_suffix_token_id == tokenizer._tokenizer.eos_id():  # EOS token
            break
        suffix_token_ids.append(new_suffix_token_id)
        prefix_tokens = prefix_tokens.at[0, len_prefix_tokens + i].set(new_suffix_token)
        prefix_mask = prefix_mask.at[0, len_prefix_tokens + i].set(True)
        print(tokenizer.detokenize(jnp.array(suffix_token_ids)))
