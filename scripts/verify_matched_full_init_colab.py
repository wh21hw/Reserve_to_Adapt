"""Both full-budget arm initialization checks; no directory creation or SGD."""
import hashlib
from pathlib import Path
import subprocess
import sys
for filename, expected in {
    'matched_full_budget.py': 'e764da6a387160c5c98d906c1e9aed0dbd1afafb2e70458a443fe072c80852e3',
    'train_matched_full_a2w_seed1.py': '0d351d40b13bbe021754c1271dfea33de3788cc64fca7c9ccca81c1829ef3826',
}.items():
    assert hashlib.sha256((Path('/content') / filename).read_bytes()).hexdigest() == expected
for arm in ('structure_off', 'structure_on'):
    result = subprocess.run([sys.executable, '-u', '/content/train_matched_full_a2w_seed1.py',
                             '--arm', arm, '--verify-init'], capture_output=True, text=True)
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    assert result.returncode == 0
    assert f'FULL_INIT_VERIFIED {arm} [56.0, 56.0, 56.0] 112.0' in result.stdout
print('BOTH_FULL_INITIALIZATIONS_PASS; 0 SGD', flush=True)
