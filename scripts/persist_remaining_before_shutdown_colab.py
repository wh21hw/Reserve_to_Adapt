"""Preserve computed models/caches before user-requested runtime deletion.

Only explicit experiment folders; no dataset duplicates, hashes, or model eval.
"""
from pathlib import Path
import shutil

base = Path('/content/imp-runs')
drive = Path('/content/drive/MyDrive/OSDA/runs')
visda = base/'visda-frozenbn-capacity-10e-v1'
destination = drive/visda.name
files = [visda/'source'/name for name in ('source-final.pt','features.npz')]
files += [visda/arm/'visda-synthetic2real_seed1'/name
          for arm in ('fixed2','estimated') for name in ('best.pt','last.pt')]
files += [visda/'fixed2-final10-geometry-v1/features.npz']
# Keep ordinary results alongside the caches, without reproducing checkpoints.
files += [path for path in visda.rglob('*') if path.is_file() and path.suffix in ('.json','.jsonl','.log')]
for original in files:
    if not original.is_file():
        raise FileNotFoundError('Computed artifact missing: '+str(original))
for original in files:
    target = destination/original.relative_to(visda)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError('Preserve existing backup: '+str(target))
    shutil.copyfile(original,target)
    print('PERSISTED_VISDA',str(target.relative_to(destination)),target.stat().st_size,flush=True)
proxy = base/'source-frozenbn-block0to4-v1'
proxy_destination = drive/proxy.name
if proxy.exists():
    extra = [proxy/'source-frozenbn/source'/name for name in ('source-final.pt','features.npz')]
    extra += [path for path in proxy.rglob('*') if path.is_file() and path.suffix in ('.json','.jsonl','.log')]
    for original in extra:
        if not original.is_file():
            raise FileNotFoundError(str(original))
        target = proxy_destination/original.relative_to(proxy)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise FileExistsError(str(target))
        shutil.copyfile(original,target)
    print('SOURCE_PROXY_CACHE_PERSISTED',flush=True)
print('ALL_NEEDED_RUNTIME_RESULTS_PERSISTED',flush=True)
