"""Apply the frozen capacity rule to OfficeHome without target labels."""
import json
from pathlib import Path
import numpy as np
from source_precision_capacity import estimate_capacity

root = Path('/content/imp-runs/source-precision-officehome-v1/seed1')
output = root/'capacity.json'
if output.exists():
    raise FileExistsError('Preserve previous inference results')
features = np.load(root/'source/features.npz')
result, settings = estimate_capacity(features['source'], features['source_labels'], features['target'])
report = dict(task='OfficeHome Pr->Rw', seed=1, settings=settings,
              K=result['K'], counts=result['counts'].tolist(),
              noise_count=result['noise_count'], converged=result['converged'],
              history=result['history'], semantic_unknown_count=None,
              source_prior='C25 supervised 3 epochs; shared by both RTA arms',
              target_labels_used=False)
output.write_text(json.dumps(report, indent=2))
print('OFFICEHOME_CAPACITY', json.dumps(report), flush=True)
