"""Extract already uploaded data/code safely; no hashes or model tests."""
from pathlib import Path
import shutil
import tarfile


def extract(archive_path, destination):
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member: '+member.name)
        archive.extractall(destination)


extract('/content/rta-baseline-code-v2.tar.gz', Path('/content/rta-legacy-l4-bridge-v1'))
extract('/content/office31_images.tar', Path('/content/osda-datasets'))
shutil.copyfile('/content/resnet50-19c8e357.pth', '/content/osda-datasets/resnet50-19c8e357.pth')
for name in ('amazon_0-9_train_all.txt', 'webcam_0-9_20-30_test.txt'):
    shutil.copyfile('/content/rta-legacy-l4-bridge-v1/data/'+name, '/content/'+name)
print('LEGACY_IMP_INPUTS_READY', flush=True)
