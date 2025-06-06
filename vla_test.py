import orbax.checkpoint
import jax
import equinox
import models.pi0 as pi0

checkpointer = orbax.checkpoint.PyTreeCheckpointer()
params = checkpointer.restore("/Users/vineethyeevani/Documents/equinox_openpi/pi0_base/params")["params"]
equinox.tree_pprint(params)
pi0_model = pi0.load(params)
equinox.tree_pprint(pi0_model)