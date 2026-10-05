"""Persist the small remaining A2W artifacts before user-requested shutdown."""
from pathlib import Path
import shutil

root = Path('/content/imp-runs/a2w-unknown-ce-10e-v1')
drive = Path('/content/drive/MyDrive/OSDA/runs/a2w-unknown-ce-10e-v1')
required = ['a2w-unknown-ce-10e-v1-results.zip', 'source-final.pt', 'control-last.pt', 'off-last.pt']
if any(not (drive/name).is_file() for name in required):
    raise FileNotFoundError('Earlier durable results missing; preserve runtime and report')
pairs = [(root/'source'/name, drive/('source-'+name))
         for name in ('features.npz','summary.json','config.json','history.jsonl')]
pairs += [(root/arm/'office31-a2w_seed3/best.pt',drive/(arm+'-best.pt')) for arm in ('control','off')]
for original,destination in pairs:
    if not original.is_file():
        raise FileNotFoundError(str(original))
    if destination.exists():
        raise FileExistsError('Do not overwrite a previous shutdown backup')
for original,destination in pairs:
    shutil.copyfile(original,destination)
    print('PERSISTED',destination.name,destination.stat().st_size,flush=True)
print('A2W_SHUTDOWN_BACKUP_COMPLETE',flush=True)
