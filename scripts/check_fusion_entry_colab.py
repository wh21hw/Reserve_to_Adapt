"""One targeted build/real-feature interface check; no training."""
import os
import subprocess

env = dict(os.environ, KONLY_SOURCE_PRIOR='build-only', FUSION_EPOCHS='70', FUSION_BUILD_ONLY='1')
subprocess.run(['/content/rta-py38/bin/python', '/content/train_fusion_imp_rta_entry.py'], env=env, check=True)
code = '''import torch
torch.set_num_threads(2)
from fusion_imp_rta import initial_structure
result, settings = initial_structure('/content/imp-runs/fusion-imp-rta-v1/seed1/source/features.npz', 10)
print('INITIAL_FUSION_K', result['candidate_count'], 'finite', bool(torch.isfinite(result['candidate_prototypes']).all()), flush=True)
'''
subprocess.run(['/content/rta-py38/bin/python', '-c', code], cwd='/content', check=True)
