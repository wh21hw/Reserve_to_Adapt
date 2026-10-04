"""One targeted regression check for the new memory-bounded proposal interface."""
import importlib.util
import sys
import numpy as np

spec=importlib.util.spec_from_file_location('streaming_capacity',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rng=np.random.RandomState(11)
anchors=np.array([[0.,0.,0.],[2.,0.,0.]])
x=np.concatenate([rng.normal(anchors[0],.07,(23,3)),rng.normal(anchors[1],.07,(19,3)),
    rng.normal([0.,2.,0.],.07,(17,3)),rng.normal([2.,2.,0.],.07,(13,3))])
settings=dict(penalty=.4,prior_strength=np.array([23.,19.]),reference_samples=42.,
    birth_order='before_update',birth_penalty=.1)
full=module.fit_robust_capacity(x,anchors,**settings)
blocked=module.fit_robust_capacity(x,anchors,proposal_block_size=7,**settings)
assert full['converged'] and blocked['converged']
assert full['K']==blocked['K'] and np.array_equal(full['assignments'],blocked['assignments'])
assert np.array_equal(full['counts'],blocked['counts'])
assert np.allclose(full['centers'],blocked['centers'],rtol=0,atol=1e-10)
assert len(full['history'])==len(blocked['history'])
assert np.allclose([r['objective'] for r in full['history']],
    [r['objective'] for r in blocked['history']],rtol=0,atol=1e-10)
print('STREAMING_CAPACITY_CHECK_OK',dict(N=len(x),block_size=7,K=full['K'],
    steps=len(full['history'])-1,same_assignments=True),flush=True)
