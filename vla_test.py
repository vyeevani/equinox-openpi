import orbax.checkpoint
import jax
import equinox

# Create a checkpointer
checkpointer = orbax.checkpoint.PyTreeCheckpointer()

# Restore the checkpoint directly
params = checkpointer.restore("/Users/vineethyeevani/Documents/equinox_openpi/pi0_base/params")

print("Successfully loaded checkpoint")
equinox.tree_pprint(params)