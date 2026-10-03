"""Archive fixed warmup checkpoint and audit files; exclude oracle-best weights."""
import hashlib
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/rta-space-warmup-l4-v1')
archive = Path('/content/rta-space-warmup-l4-checkpoint-v1.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for name in ['audit.json', 'console.log', 'a2w_seed1/config.json',
                 'a2w_seed1/history.jsonl', 'a2w_seed1/last.pt']:
        bundle.write(root/name, name)
print('RTA_CHECKPOINT_ARCHIVE', archive.stat().st_size,
      hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
