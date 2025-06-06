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
    
    siglip_model = siglip.load(img_params)
    gemma_model = gemma.load(llm_params)
    
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
