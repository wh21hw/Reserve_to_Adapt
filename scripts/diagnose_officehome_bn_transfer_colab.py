"""Read existing fixed-K checkpoints' BN buffers; no inference or reevaluation."""
import json
from pathlib import Path
import numpy as np
import torch

base=Path('/content/imp-runs')
roots={'ordinary':base/'officehome-capacity-10e-v1',
       'frozen_source':base/'officehome-frozenbn-capacity-10e-v1'}
output=base/'officehome-frozenbn-capacity-10e-v1/bn-transfer-diagnosis.json'
if output.exists():raise FileExistsError('Preserve prior diagnosis')
report=dict(purpose='BN-buffer drift at matched fixed K4, source prior to final10',
    checkpoint_inference=False,target_labels_used=False,arms={})
for name,root in roots.items():
    launch=json.loads((root/'fixed4/launch.json').read_text())
    if launch['K']!=4 or launch['seed']!=1 or launch['epochs']!=10:
        raise ValueError('Expected matched fixed K4, seed1, ten-epoch arms')
    prior_path=Path(launch['source_prior'])
    final_path=root/'fixed4/officehome-pr2rw_seed1/last.pt'
    prior=torch.load(str(prior_path),map_location='cpu')['model']
    final=torch.load(str(final_path),map_location='cpu')['model']
    keys=[key for key in prior if key.startswith('0.') and key.endswith('running_mean')]
    if not keys:raise ValueError('Missing encoder BN buffers')
    shifts=[];ratios=[];batches=[]
    for key in keys:
        variance=key[:-len('running_mean')]+'running_var'
        counter=key[:-len('running_mean')]+'num_batches_tracked'
        shifts.extend(((final[key]-prior[key])/torch.sqrt(prior[variance]+1e-5)).abs().numpy().tolist())
        ratios.extend(torch.log((final[variance]+1e-5)/(prior[variance]+1e-5)).abs().numpy().tolist())
        batches.append(int(final[counter]-prior[counter]))
    report['arms'][name]=dict(source_prior=str(prior_path),final=str(final_path),encoder_bn_buffer_paths=len(keys),
        absolute_mean_shift_in_prior_std_quantiles=np.quantile(shifts,[.5,.9,.99]).tolist(),
        absolute_log_variance_ratio_quantiles=np.quantile(ratios,[.5,.9,.99]).tolist(),
        additional_training_batch_counter_range=[min(batches),max(batches)])
    del prior,final
report['caveat']='Quantiles are over state-dict buffer paths, which may include aliases, not independent BN layers. Buffers changing is expected in ordinary RTA; it does not prove BN drift caused the HOS difference. No new train/eval, loss or K change.'
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('BN_TRANSFER_BUFFER_DIAGNOSIS',json.dumps(report),flush=True)
