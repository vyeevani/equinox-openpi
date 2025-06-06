import orbax.checkpoint
import jax
import equinox

checkpointer = orbax.checkpoint.PyTreeCheckpointer()
params = checkpointer.restore("/Users/vineethyeevani/Documents/equinox_openpi/pi0_base/params")["params"]["PaliGemma"]

equinox.tree_pprint(params)

import models.gemma as gemma

model = gemma.load(configs=[gemma.Config.get_variant("gemma_2b"), gemma.Config.get_variant("gemma_300m")], params= params["llm"])
equinox.tree_pprint(model)