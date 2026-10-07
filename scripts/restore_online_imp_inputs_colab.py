"""Restore cachedOffice31/sourceprior, no repeated public data downloads."""
from pathlib import Path
import shutil
import tarfile

drive = Path('/content/drive/MyDrive/OSDA')
bundle = Path('/content/online-imp-code-v1.tar.gz')
with tarfile.open(bundle) as archive:
    archive.extractall('/content/online-imp-code-v1')
code = Path('/content/online-imp-code-v1')
base = Path('/content/rta-legacy-l4-bridge-v1')
if base.exists():
    raise FileExistsError('Do not replace existing runtime baseline')
shutil.copytree(code/'experiments/rta_l4_control_v1',base)
# The frozen local control has a torch2.x load flag; legacytorch1.7 does not
# accept it. Removing only this keyword leaves trustedcheckpoint loading intact.
networks = base/'networks.py'
networks.write_text(networks.read_text().replace(', weights_only=False', ''))
(base/'data').mkdir()
for filename in ('amazon_0-9_train_all.txt','webcam_0-9_20-30_test.txt'):
    shutil.copyfile(code/'data'/filename,base/'data'/filename)
for item in list(code.glob('*.py'))+list((code/'scripts').glob('*.py')):
    shutil.copyfile(item,Path('/content')/item.name)
data = Path('/content/osda-office31-a2w-v1')
data.mkdir()
with tarfile.open(drive/'datasets/office31_images.tar') as archive:
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
            raise ValueError('Unsafe cached dataset member')
    archive.extractall(data)
for filename in ('amazon_0-9_train_all.txt','webcam_0-9_20-30_test.txt'):
    shutil.copyfile(base/'data'/filename,data/filename)
weights = Path('/content/osda-datasets')
weights.mkdir()
shutil.copyfile(drive/'pretrained/resnet50-19c8e357.pth',weights/'resnet50-19c8e357.pth')
prior = Path('/content/online-source')
prior.mkdir()
shutil.copyfile(drive/'runs/a2w-unknown-ce-10e-v1/source-final.pt',prior/'source-final.pt')
print('ONLINE_IMP_INPUTS_RESTORED; cacheddata and sharedsource3, no retraining',flush=True)
