"""Release only the objects created by failed notebook preflight, not runtime."""
import gc
import torch

for name in ('net','discriminator','bank','wrappers','checkpoint','images',
             'features','logits','probability','loss','state'):
    globals().pop(name,None)
gc.collect()
torch.cuda.empty_cache()
print('PREFLIGHT_TRANSIENT_OBJECTS_RELEASED',flush=True)
