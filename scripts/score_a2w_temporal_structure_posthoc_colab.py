"""Evaluate already persisted partitions; never refit K or select parameters.

Target truth is read ONLY here, after both label-blind artifacts are saved.
This is cluster diagnosis, not a new RTA accuracy/checkpoint evaluation.
"""
import json
from pathlib import Path
import shutil
import numpy as np


def composition(ids, truth):
    known = truth < 10
    occupied = np.unique(ids[ids >= 0])
    clusters = []
    for cluster in occupied:
        rows = ids == cluster
        semantic, counts = np.unique(truth[rows], return_counts=True)
        clusters.append(dict(identity=int(cluster), size=int(rows.sum()),
            known_rows=int((rows & known).sum()), unknown_rows=int((rows & ~known).sum()),
            semantic_counts={int(label):int(count) for label,count in zip(semantic,counts)}))
    return dict(known_as_unknown_fraction=float((ids[known] >= 10).mean()),
        unknown_as_unknown_fraction=float((ids[~known] >= 10).mean()),
        unknown_absorbed_by_known_fraction=float(((ids[~known] >= 0)&(ids[~known] < 10)).mean()),
        known_identity_accuracy=float((ids[known] == truth[known]).mean()),
        known_noise_fraction=float((ids[known] < 0).mean()),
        unknown_noise_fraction=float((ids[~known] < 0).mean()),
        occupied_clusters=len(occupied), clusters=clusters)


def main():
    drive = Path('/content/drive/MyDrive/OSDA/runs')
    current = drive/'a2w-temporal-capacity-v1'
    report = json.loads((current/'summary.json').read_text())
    if report['target_labels_used']:
        raise ValueError('Capacity must be frozen before posthoc evaluation')
    output = Path('/content/imp-runs/a2w-temporal-capacity-posthoc-v1')
    if output.exists():
        raise FileExistsError('Keep the existing report; no repeat fitting')
    with np.load(current/'clusters.npz', allow_pickle=False) as data:
        paths = data['target_paths']
        current_raw = data['assignments']
        current_matched = data['matched_assignments'] if 'matched_assignments' in data else None
    previous = drive/'a2w-unknown-ce-10e-v1'
    partitions = dict(final10_raw=current_raw)
    if current_matched is not None:
        partitions['final10_matched'] = current_matched
    for name,folder in [('source3_raw','imp-structure-probe-v1'),
                        ('source3_matched','identity-reconciliation-probe-v1')]:
        with np.load(previous/folder/'clusters.npz', allow_pickle=False) as data:
            if not np.array_equal(data['target_paths'],paths):
                raise ValueError('Saved partitions have different sample ordering')
            partitions[name] = data['assignments']
    rows = [line.rsplit(None,1) for line in Path(
        '/content/osda-office31-a2w-v1/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
    if not np.array_equal(np.asarray([row[0] for row in rows]), paths):
        raise ValueError('Evaluation truth ordering differs')
    truth = np.asarray([int(row[1]) for row in rows])
    result = dict(task='Office31 A->W', new_training=False, new_image_forward=False,
        target_labels_posthoc_only=True, capacity_artifact=str(current),
        partitions={name:composition(ids,truth) for name,ids in partitions.items()},
        caveat='No target-fitted K, threshold or hyperparameter; descriptive on this already observed task/seed. '
               'Cluster detection is not RTA accuracy and does not establish stale-label causality.')
    output.mkdir(parents=True)
    (output/'composition.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    saved = drive/'a2w-temporal-capacity-posthoc-v1'
    saved.mkdir(exist_ok=False)
    shutil.copyfile(output/'composition.json', saved/'composition.json')
    compact = {name:{key:value for key,value in item.items() if key != 'clusters'}
               for name,item in result['partitions'].items()}
    print('A2W_TEMPORAL_POSTHOC_COMPLETE', json.dumps(compact), flush=True)


if __name__ == '__main__':
    main()
