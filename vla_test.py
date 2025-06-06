import orbax.checkpoint
import jax
import jax.numpy as jnp
import equinox

import models.pi0 as pi0
from tokenizer import Tokenizer

prompt = "Pick up the banana"
tokenizer = Tokenizer()
language_token_ids, language_mask = tokenizer.tokenize(prompt)
language_token_ids = language_token_ids[language_mask]

checkpointer = orbax.checkpoint.PyTreeCheckpointer()
params = checkpointer.restore("/Users/vineethyeevani/Documents/equinox_openpi/pi0_base/params")["params"]
pi0_model = pi0.load(params)

actions = pi0_model.sample_actions_with_cache(
    language_token_ids = language_token_ids,
    image = jnp.ones((224, 224, 3)),
    state = jnp.ones((32,)),
    sample_steps = 10,
    key = jax.random.PRNGKey(0),
)