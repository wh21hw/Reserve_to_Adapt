"""Restore frozen code/cached data and preserve previous small results."""
from pathlib import Path
import shutil
import tarfile
import zipfile

drive=Path('/content/drive/MyDrive/OSDA')
code=Path('/content/legacy-code-oct8')
code.mkdir()
with tarfile.open('/content/online-imp-code-v1.tar.gz') as archive:
    for item in archive.getmembers():
        path=Path(item.name)
        if path.is_absolute() or '..' in path.parts or not(item.isfile() or item.isdir()):
            raise ValueError('Unsafe code member')
    archive.extractall(code)
base=Path('/content/rta-legacy-l4-bridge-v1')
shutil.copytree(code/'experiments/rta_l4_control_v1',base)
networks=base/'networks.py'
networks.write_text(networks.read_text().replace(', weights_only=False',''))
data=Path('/content/osda-office31-a2w-v1');data.mkdir()
with tarfile.open(drive/'datasets/office31_images.tar') as archive:
    for item in archive.getmembers():
        path=Path(item.name)
        if path.is_absolute() or '..' in path.parts or not(item.isfile() or item.isdir()):
            raise ValueError('Unsafe data member')
    archive.extractall(data)
for name in ('amazon_0-9_train_all.txt','webcam_0-9_20-30_test.txt'):
    shutil.copyfile(code/'data'/name,data/name)
weights=Path('/content/osda-datasets');weights.mkdir()
shutil.copyfile(drive/'pretrained/resnet50-19c8e357.pth',weights/'resnet50-19c8e357.pth')
recovered=Path('/content/legacy-recovered-oct8');recovered.mkdir()
for run in ('legacy-imp-seed-screen-v1','legacy-imp-param-screen-v1'):
    source=drive/'runs'/run
    if source.is_dir():
        for item in source.rglob('*'):
            if item.is_file() and item.suffix!='.pt':
                dest=recovered/run/item.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(item,dest)
with zipfile.ZipFile('/content/legacy-recovered-oct8.zip','x',zipfile.ZIP_DEFLATED) as archive:
    for item in recovered.rglob('*'):
        if item.is_file():archive.write(item,item.relative_to(recovered))
print('LEGACY_CACHED_INPUTS_READY; previous smallresults recovered',flush=True)
