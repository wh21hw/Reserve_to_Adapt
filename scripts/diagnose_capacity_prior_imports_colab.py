import json
from pathlib import Path
import sys

report=dict(module_paths={name:getattr(sys.modules.get(name),'__file__',None)
    for name in ('networks','pilot_state','capacity_prior','utilities','centroid')},
    classifier_type_module=net[1].__class__.__module__,
    classifier_buffers=list(net[1]._buffers),
    prior_shape=list(net[1].capacity_log_prior.shape),
    saved_prior_present='1.capacity_log_prior' in net.state_dict(),
    optimizer_steps=[wrapper.global_step for wrapper in wrappers],
    interpretation='Investigate notebook module cache; no adaptation experiment launched')
with Path('/content/capacity-prior-import-failure-v1.json').open('x') as stream:
    json.dump(report,stream,indent=2)
print('PRIOR_PREFLIGHT_IMPORT_DIAGNOSIS',json.dumps(report),flush=True)
