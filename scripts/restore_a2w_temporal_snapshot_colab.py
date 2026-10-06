"""Restore only code/dataset/weights needed for one cached-state diagnosis."""
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
checkpoint = drive/'runs/a2w-reconciled-identity-10e-v1/argmax-last.pt'
if not checkpoint.is_file():
    raise FileNotFoundError('Mount Drive and restore the existing final10 checkpoint')
code = Path('/content/rta-legacy-l4-bridge-v1')
unpack('/content/rta-baseline-code-v2.tar.gz',code)
support = Path('/content/a2w-temporal-support-v1')
unpack('/content/a2w-temporal-support-v1.tar.gz',support)
for original in support.rglob('*.py'):
    destination = Path('/content')/original.name
    if destination.exists():
        raise FileExistsError(str(destination))
    shutil.copyfile(original,destination)
archive = Path('/content/office31_images.tar')
shutil.copyfile(drive/'datasets/office31_images.tar',archive)
data = Path('/content/osda-office31-a2w-v1')
unpack(archive,data)
for name in ('amazon_0-9_train_all.txt','webcam_0-9_20-30_test.txt'):
    shutil.copyfile(code/'data'/name,data/name)
weights = Path('/content/osda-datasets')
weights.mkdir(exist_ok=True)
shutil.copyfile(drive/'pretrained/resnet50-19c8e357.pth',weights/'resnet50-19c8e357.pth')
print('A2W_TEMPORAL_INPUTS_RESTORED; existing Drive archives and final10 model only',flush=True)
