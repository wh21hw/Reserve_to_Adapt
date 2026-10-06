"""Restore this bounded pair from durable Drive caches, not public downloads."""
from pathlib import Path
import shutil
import tarfile


def unpack(archive,destination):
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive) as bundle:
        for item in bundle.getmembers():
            path = Path(item.name)
            if path.is_absolute() or '..' in path.parts or not (item.isfile() or item.isdir()):
                raise ValueError('Unsafe archive member: '+item.name)
        bundle.extractall(destination)


drive = Path('/content/drive/MyDrive/OSDA')
prior = drive/'runs/a2w-unknown-ce-10e-v1/source-final.pt'
if not prior.is_file():
    raise FileNotFoundError('Require mounted existing C10/source3 prior')
unpack('/content/rta-baseline-code-v2.tar.gz',Path('/content/rta-legacy-l4-bridge-v1'))
support = Path('/content/a2w-capacity-refresh-support-v1')
unpack('/content/a2w-capacity-refresh-support-v1.tar.gz',support)
for original in support.rglob('*.py'):
    destination = Path('/content')/original.name
    if destination.exists():
        raise FileExistsError('Do not overwrite runtime code: '+str(destination))
    shutil.copyfile(original,destination)
archive = Path('/content/office31_images.tar')
shutil.copyfile(drive/'datasets/office31_images.tar',archive)
data = Path('/content/osda-office31-a2w-v1')
unpack(archive,data)
for name in ('amazon_0-9_train_all.txt','webcam_0-9_20-30_test.txt'):
    shutil.copyfile(Path('/content/rta-legacy-l4-bridge-v1/data')/name,data/name)
weights = Path('/content/osda-datasets')
weights.mkdir(exist_ok=True)
shutil.copyfile(drive/'pretrained/resnet50-19c8e357.pth',weights/'resnet50-19c8e357.pth')
source = Path('/content/imp-runs/a2w-capacity-refresh-20e-v1/source')
source.mkdir(parents=True,exist_ok=False)
shutil.copyfile(prior,source/'source-final.pt')
shutil.copyfile(drive/'runs/a2w-unknown-ce-10e-v1/source-summary.json',source/'summary.json')
print('A2W_CAPACITY_REFRESH_INPUTS_RESTORED; shared source prior, no source training',flush=True)
