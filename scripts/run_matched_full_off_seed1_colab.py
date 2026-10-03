"""ONE full-budget arm, no automatic next experiment or retry."""
import hashlib
from pathlib import Path
import subprocess
import sys
for filename, expected in {
    'matched_full_budget.py': 'e764da6a387160c5c98d906c1e9aed0dbd1afafb2e70458a443fe072c80852e3',
    'train_matched_full_a2w_seed1.py': '0d351d40b13bbe021754c1271dfea33de3788cc64fca7c9ccca81c1829ef3826',
}.items():
    assert hashlib.sha256((Path('/content') / filename).read_bytes()).hexdigest() == expected
with Path('/content/matched-full-a2w-seed1-off-console-v1.log').open('x') as stream:
    process = subprocess.Popen([sys.executable, '-u', '/content/train_matched_full_a2w_seed1.py',
                                '--arm', 'structure_off'], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError(f'Full arm failed ({status}); preserve evidence, no automatic retry')
