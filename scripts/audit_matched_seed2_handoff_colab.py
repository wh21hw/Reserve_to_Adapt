"""Reuse the pinned CPU handoff audit with seed2's independently verified inputs."""
import hashlib
from pathlib import Path
import subprocess
import sys

worker = Path('/content/audit_matched_real_handoff.py')
base = worker.read_bytes()
assert hashlib.sha256(base).hexdigest() == '34322e6c7068507bc0a9c03143a818c9eac85b8f992ecd50c26c405723a71290'
source = base.decode()
replacements = {
    'office31-a2w_seed1/warmup-complete.pt': 'office31-a2w_seed2/warmup-complete.pt',
    'ff88bdc9d49a1a8c151baa8b4ca166549a2723195882ec37fe5056e0d91f1eb1': 'e16e895e1bdaec17f14dd4eda7cc81f8b05ad1c1c6d76145bbd3088aa1d2fea0',
    'matched-warm-candidates-v1/original.npz': 'matched-warm-candidates-seed2-v1/original.npz',
    '3d4b9fe2833e89f97863db3f718bcf00dd4899d1221c80393e1668c44883721b': '16db9d30eb56d3899c148b1482a3da7c4a1800cdf1769073e3fa4781d00f1e8b',
    '/content/matched-real-handoff-v1.json': '/content/matched-real-handoff-seed2-v1.json',
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
for name, expected in {
    'matched_handoff.py': '789727cc8b498aa39aead54446f01bc8969c9f79935b4e68014b370abd87cbd8',
    'prototype_structure.py': '4f7ed0fa3bce8f2a2cea88d964917c3f69e333ee6b5257aa5e218deb04f964d8',
}.items():
    assert hashlib.sha256((Path('/content') / name).read_bytes()).hexdigest() == expected
# Kernel sys.modules can retain a different experiment's imports. A fresh child
# keeps the strict import boundary without restarting or altering the runtime.
anchor = 'from utilities import OptimWithSheduler, inverseDecaySheduler'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "\nimport utilities\nassert Path(networks.__file__).resolve() == code / 'networks.py'\nassert Path(utilities.__file__).resolve() == code / 'utilities.py'")
result = subprocess.run([sys.executable, '-u', '-c', source], capture_output=True, text=True)
print(result.stdout, flush=True)
print(result.stderr, flush=True)
assert result.returncode == 0
