"""Fixed unlabeled cluster identities -> existing RTA unknown slots.

Keeps candidate selection, head initialization, loss and prediction unchanged.
Known/noise cluster members retain the original unknown-slot argmax fallback.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from domain_bus import DomainBus


class IndexedDomainBus(DomainBus):
    """Same cycling and ordering, preserving optional third sample-index field."""
    def get_samples(self):
        batches = []
        for i in range(len(self.domainloaders)):
            try:
                batch = next(self.domainiters[i])
            except StopIteration:
                self.domainiters[i] = iter(self.domainloaders[i])
                batch = next(self.domainiters[i])
            batches.append(batch)
        self.current_iter += 1
        return batches


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


class ClusterIdentityBridge:
    def __init__(self, artifact, target_list, known, slots):
        self.known, self.slots = known, slots
        paths = [line.rsplit(None, 1)[0] for line in open(target_list) if line.strip()]
        with np.load(artifact, allow_pickle=False) as data:
            self.assignments = data['assignments'].astype(np.int64)
            expected = data['target_paths'].tolist()
        if paths != expected or self.assignments.shape != (len(paths),):
            raise ValueError('Cluster artifact does not match target path order')
        if not np.array_equal(np.unique(self.assignments[self.assignments >= known]),
                              np.arange(known, known + slots)):
            raise ValueError('Expected every inferred unknown cluster, with unchanged K')
        if (self.assignments < -1).any():
            raise ValueError('Invalid cluster ID')
        self.mapping = None
        self.reset_counts()

    def reset_counts(self):
        self.selected = 0
        self.overridden = 0
        self.slot_counts = np.zeros(self.slots, dtype=np.int64)

    def align(self, features, indices, head):
        features = np.asarray(features)
        ids = self.assignments[np.asarray(indices, dtype=np.int64).reshape(-1)]
        centers = []
        for cluster in range(self.known, self.known + self.slots):
            rows = features[ids == cluster]
            if not len(rows):
                raise RuntimeError('Warm-end batches missed an unknown cluster; no silent mapping')
            centers.append(rows.mean(0))
        centers = np.stack(centers)
        weights = head.detach().cpu().numpy()[self.known:]
        if weights.shape != centers.shape or not np.isfinite(centers).all():
            raise ValueError('Cluster/head dimensions or finite values conflict')
        centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        weights = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-8)
        rows, columns = linear_sum_assignment(1 - centers.dot(weights.T))
        self.mapping = np.empty(self.slots, dtype=np.int64)
        self.mapping[rows] = columns + self.known
        return self.mapping.tolist()

    def labels(self, sample_indices, fallback):
        if self.mapping is None:
            return fallback
        ids = self.assignments[np.asarray(sample_indices, dtype=np.int64).reshape(-1)]
        mask = ids >= self.known
        labels = fallback.detach().clone()
        labels[torch.as_tensor(mask, device=labels.device)] = torch.as_tensor(
            self.mapping[ids[mask] - self.known], device=labels.device, dtype=labels.dtype)
        self.selected += len(ids)
        self.overridden += int(mask.sum())
        self.slot_counts += np.bincount(labels.detach().cpu().numpy() - self.known,
                                       minlength=self.slots)
        return labels

    def report(self):
        return dict(selected=self.selected, cluster_labels_used=self.overridden,
                    fallback=self.selected - self.overridden,
                    pseudo_slot_counts=self.slot_counts.tolist(),
                    cluster_to_slot=None if self.mapping is None else self.mapping.tolist())


def patch_identity_entry(source):
    """Apply only identity transport to the already task-adapted trainer source."""
    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError('Unexpected identity bridge anchor: ' + old[:80])
        source = source.replace(old, new)
    replace('from domain_bus import DomainBus',
            'from cluster_identity_bridge import IndexedDomainBus as DomainBus')
    replace('ds1 = CustomDataset(images,labels,img_transformer=transform,is_train=True)',
        '''from cluster_identity_bridge import IndexedDataset, ClusterIdentityBridge
cluster_bridge = ClusterIdentityBridge(os.environ['RTA_CLUSTER_LABELS'], args.target,
    args.shared_classes, args.all_classes-args.shared_classes)
ds1 = IndexedDataset(CustomDataset(images,labels,img_transformer=transform,is_train=True))''')
    replace("['pred_s','pred_t','label_s', 'kl','fss','ftt']",
            "['pred_s','pred_t','label_s', 'kl','fss','ftt','cluster_sample_ids']")
    replace('(im_target, label_target)) in enumerate(customgenearator):',
            '(im_target, label_target, target_indices)) in enumerate(customgenearator):', count=2)
    replace('            im_target = im_target.cuda()\n            \n            _, feature_source, fc_source',
            '''            im_target = im_target.cuda()
            cluster_sample_ids = target_indices.cpu().numpy()
            
            _, feature_source, fc_source''', count=2)
    replace('                pseudo_index=pseudo_index + args.shared_classes',
        '''                pseudo_index=pseudo_index + args.shared_classes
                pseudo_index = cluster_bridge.labels(
                    target_indices[r.view(-1).detach().cpu()].cpu().numpy(), pseudo_index)''')
    replace('    if epoch<=30:\n', '''    if epoch == warmiter:
        mapping = cluster_bridge.align(ProbRecorder['ftt'], ProbRecorder['cluster_sample_ids'], net[1].fc.weight)
        print('CLUSTER_IDENTITY_ALIGNED', mapping, flush=True)
    row = dict(epoch=epoch+1, **cluster_bridge.report())
    with open(os.path.join(args.log_dir, 'cluster-identity-history.jsonl'), 'a') as stream:
        stream.write(json.dumps(row)+'\\n')
    print('CLUSTER_IDENTITY_USAGE', row, flush=True)
    cluster_bridge.reset_counts()
    if epoch<=30:
''')
    return source
