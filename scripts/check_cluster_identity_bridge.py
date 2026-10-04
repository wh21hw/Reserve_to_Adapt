"""One targeted CPU check: identity/index transport, permutation and entry hooks."""
from pathlib import Path
import tempfile
import numpy as np
import torch
from cluster_identity_bridge import (ClusterIdentityBridge, IndexedDataset,
                                     IndexedDomainBus, patch_identity_entry)

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    paths = ['a.png', 'b.png', 'c.png', 'd.png']
    (root/'target.txt').write_text('\n'.join(p+' 999' for p in paths))
    np.savez(root/'clusters.npz', assignments=np.array([2, 3, 0, -1]), target_paths=np.array(paths))
    bridge = ClusterIdentityBridge(root/'clusters.npz', root/'target.txt', 2, 2)
    features = np.array([[1., 0.], [0., 1.], [.5, .5], [.3, .2]])
    head = torch.tensor([[.2, .3], [.1, .2], [0., 1.], [1., 0.]])
    assert bridge.align(features, np.arange(4), head) == [3, 2]
    labels = bridge.labels(np.arange(4), torch.tensor([2, 3, 3, 2]))
    assert labels.tolist() == [3, 2, 3, 2]
    assert bridge.report()['cluster_labels_used'] == 2
    assert bridge.report()['pseudo_slot_counts'] == [2, 2]
    base = torch.utils.data.TensorDataset(torch.arange(4), torch.arange(4))
    source = torch.utils.data.DataLoader(base, batch_size=2)
    target = torch.utils.data.DataLoader(IndexedDataset(base), batch_size=2, shuffle=False)
    bus = IndexedDomainBus([source, target], iter_num=3)
    observed = [batch[1][2].tolist() for batch in bus]
    assert observed == [[0, 1], [2, 3], [0, 1]]
raw = Path('/content/rta-legacy-l4-bridge-v1/main.py').read_text()
patched = patch_identity_entry(raw)
compile(patched, '<identity patched legacy>', 'exec')
assert patched.count('target_indices)) in enumerate(customgenearator)') == 2
print('CLUSTER_IDENTITY_INTERFACE_CHECK_OK: CPU only; no training/evaluation')
