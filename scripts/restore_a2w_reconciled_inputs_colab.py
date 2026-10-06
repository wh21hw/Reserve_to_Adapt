"""Restore the approved K8 pair from existing Drive caches; no retraining."""
from pathlib import Path
import shutil
import tarfile
import zipfile

def unpack_tar(archive, destination):
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive) as bundle:
        for item in bundle.getmembers():
            path = Path(item.name)
            if path.is_absolute() or '..' in path.parts or not (item.isfile() or item.isdir()):
                raise ValueError('Unsafe archive member: '+item.name)
        bundle.extractall(destination)

drive = Path('/content/drive/MyDrive/OSDA')
saved = drive/'runs/a2w-unknown-ce-10e-v1'
if not (saved/'source-final.pt').is_file():
    raise FileNotFoundError('Drive must be mounted; require the existing shared source3 prior')
unpack_tar('/content/rta-baseline-code-v2.tar.gz',Path('/content/rta-legacy-l4-bridge-v1'))
support = Path('/content/a2w-reconciled-support-v1')
unpack_tar('/content/a2w-reconciled-support-v1.tar.gz',support)
for path in support.rglob('*.py'):
    destination = Path('/content')/path.name
    if destination.exists():
        raise FileExistsError('Do not overwrite a runtime entry: '+str(destination))
    shutil.copyfile(path,destination)
data = Path('/content/osda-office31-a2w-v1')
archive = Path('/content/office31_images.tar')
shutil.copyfile(drive/'datasets/office31_images.tar',archive)
unpack_tar(archive,data)
for name in ('amazon_0-9_train_all.txt','webcam_0-9_20-30_test.txt'):
    shutil.copyfile(Path('/content/rta-legacy-l4-bridge-v1/data')/name,data/name)
    rows = [line.rsplit(None,1)[0] for line in (data/name).read_text().splitlines() if line.strip()]
    if any(not (data/name).is_file() for name in rows):
        raise FileNotFoundError('Restored dataset image path missing')
weights = Path('/content/osda-datasets')
weights.mkdir(exist_ok=True)
shutil.copyfile(drive/'pretrained/resnet50-19c8e357.pth',weights/'resnet50-19c8e357.pth')
reference = Path('/content/imp-runs/a2w-unknown-ce-10e-v1')
reference.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(saved/'a2w-unknown-ce-10e-v1-results.zip') as bundle:
    for name in ('summary.json','control/launch.json'):
        path = reference/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with bundle.open(name) as original,path.open('xb') as destination:
            shutil.copyfileobj(original,destination)
source = reference/'source'
source.mkdir()
shutil.copyfile(saved/'source-final.pt',source/'source-final.pt')
for name in ('features.npz','summary.json','config.json','history.jsonl'):
    shutil.copyfile(saved/('source-'+name),source/name)
artifact = reference/'identity-reconciliation-probe-v1'
artifact.mkdir()
for name in ('settings.json','clusters.npz'):
    shutil.copyfile(saved/'identity-reconciliation-probe-v1'/name,artifact/name)
print('A2W_K8_INPUTS_RESTORED_FROM_DRIVE; no source training or target-label inference',flush=True)
