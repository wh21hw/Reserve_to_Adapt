"""Hash/schema/finite checks of own-seed exported inputs, never train."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,choices=(2,3),required=True)
parser.add_argument('--archive-sha256',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/'pipeline-results'
path=root/f'multitask-a2w-seed{args.seed}-warm-features-v1.zip'
assert hashlib.sha256(path.read_bytes()).hexdigest()==args.archive_sha256
with zipfile.ZipFile(path) as archive:
    assert archive.testzip() is None
    manifest=json.loads(archive.read('manifest.json'))
    assert manifest['seed']==args.seed and manifest['epoch']==4
    assert manifest['target_labels_in_features'] is False and manifest['optimizer_steps_executed']==0
    expected={2:'e16e895e1bdaec17f14dd4eda7cc81f8b05ad1c1c6d76145bbd3088aa1d2fea0',
              3:'f74f473b8d5ae58bac64916db8dc5ad153ba2b54c5e4688227c460a96956152b'}
    assert manifest['checkpoint_sha256']==expected[args.seed]
    for name,digest in manifest['files'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    data=np.load(io.BytesIO(archive.read('features.npz')),allow_pickle=False)
    assert set(data.files)=={'source','target','source_labels','source_logits','target_logits'}
    assert data['source'].shape==(958,256) and data['target'].shape==(564,256)
    assert data['source_logits'].shape==(958,12) and data['target_logits'].shape==(564,12)
    assert all(np.isfinite(data[name]).all() for name in data.files)
    assert np.array_equal(np.unique(data['source_labels']),np.arange(10))
print('OWN_SEED_WARM_ARCHIVE_VERIFIED',args.seed,manifest['files']['features.npz'])
