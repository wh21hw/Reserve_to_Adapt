"""Pinned real-image backward-only check using seed2's own inputs."""
import hashlib
from pathlib import Path
import subprocess
import sys

worker = Path('/content/preflight_matched_structure_colab.py')
base = worker.read_bytes()
assert hashlib.sha256(base).hexdigest() == '3354097dbfea0c631f0b75b3ab8c8519c5f24d256cffdd2b432cb773b97bd44f'
source = base.decode()
replacements = {
    'office31-a2w_seed1/warmup-complete.pt': 'office31-a2w_seed2/warmup-complete.pt',
    'ff88bdc9d49a1a8c151baa8b4ca166549a2723195882ec37fe5056e0d91f1eb1': 'e16e895e1bdaec17f14dd4eda7cc81f8b05ad1c1c6d76145bbd3088aa1d2fea0',
    'multitask-a2w-seed1-warm-features-v1/features.npz': 'multitask-a2w-seed2-warm-features-v1/features.npz',
    'a9d1a40aeb863b72aafefaed965d22d8eb3b3099516aee117f1897bdb4fb5459': '37972fed3fb89f8418e38186b050e3f525ab25f6515faf2a38a00c6172889f14',
    'matched-warm-candidates-v1/original.npz': 'matched-warm-candidates-seed2-v1/original.npz',
    '3d4b9fe2833e89f97863db3f718bcf00dd4899d1221c80393e1668c44883721b': '16db9d30eb56d3899c148b1482a3da7c4a1800cdf1769073e3fa4781d00f1e8b',
    '/content/matched-structure-realbatch-v1.json': '/content/matched-structure-realbatch-seed2-isolated-v2.json',
    "centers = torch.from_numpy(proposal['candidates'][indices]).cuda()": "centers = torch.from_numpy(proposal['candidates'][indices])",
    "torch.as_tensor(mass[indices], device='cuda')": 'torch.as_tensor(mass[indices])',
    'networks.CLS(2048, 12)).cuda()': 'networks.CLS(2048, 12))',
    'torch.nn.Linear(256, 10 + len(indices), bias=False).cuda()': 'torch.nn.Linear(256, 10 + len(indices), bias=False)',
    'net[1].main[1][2] = net[1].fc': 'net[1].main[1][2] = net[1].fc\nnet = net.cuda()\ncenters, prior = centers.cuda(), prior.cuda()',
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
result = subprocess.run([sys.executable, '-u', '-c', source], capture_output=True, text=True)
print(result.stdout, flush=True)
print(result.stderr, flush=True)
assert result.returncode == 0
