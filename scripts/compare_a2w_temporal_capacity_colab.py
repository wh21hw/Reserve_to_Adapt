"""Apply the frozen existing estimator once to final10 cached features.

Reuse already fitted source3 artifacts; no repeat source inference, GPU, target
truth, GMM/threshold scan, checkpoint selection, or downstream training.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0,'/content')
from source_precision_capacity import estimate_source_cost_capacity
from prototype_identity_reconciliation import reconcile_known_identities
from temporal_partition_report import compare_partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous',default='/content/drive/MyDrive/OSDA/runs/a2w-unknown-ce-10e-v1')
    parser.add_argument('--current',default='/content/drive/MyDrive/OSDA/runs/a2w-current-relation-snapshot-v1')
    parser.add_argument('--output',default='/content/imp-runs/a2w-temporal-capacity-v1')
    args = parser.parse_args()
    previous,current,output = map(Path,(args.previous,args.current,args.output))
    if output.exists():
        raise FileExistsError('Preserve existing comparison; do not rerun fitting')
    manifest = json.loads((current/'manifest.json').read_text())
    if not manifest['complete'] or manifest['target_labels_used'] or (manifest['epoch'],manifest['C'],manifest['K']) != (10,10,8):
        raise ValueError('Require the declared final10 label-free snapshot')
    old_capacity = json.loads((previous/'imp-structure-probe-v1/capacity.json').read_text())
    old_identity = json.loads((previous/'identity-reconciliation-probe-v1/settings.json').read_text())
    with np.load(previous/'imp-structure-probe-v1/clusters.npz',allow_pickle=False) as data:
        old_raw,paths = data['assignments'],data['target_paths']
    with np.load(previous/'identity-reconciliation-probe-v1/clusters.npz',allow_pickle=False) as data:
        old_matched = data['assignments']
        if not np.array_equal(data['target_paths'],paths):
            raise ValueError('Previous artifact ordering differs')
    with np.load(current/'features.npz',allow_pickle=False) as data:
        source,labels,target,logits = data['source'],data['source_labels'],data['target'],data['target_logits']
        if not np.array_equal(data['target_paths'],paths):
            raise ValueError('Current artifact ordering differs')
    if source.shape != (958,256) or target.shape != (564,256) or logits.shape != (564,18):
        raise ValueError('Unexpected complete snapshot shapes')
    result,settings = estimate_source_cost_capacity(source,labels,target,proposal_block_size=64)
    # Numeric radii/costs recalibrate from the current source, by the same rule.
    # They are not target-selected constants and are NOT held numerically equal.
    for key in ('version','calibration_seed','source_prior_counts','source_rows','target_rows',
                'prior_mass_mode','birth_order','proposal_block_size'):
        if settings[key] != old_capacity['settings'][key]:
            raise ValueError('Estimator rule/input protocol differs: '+key)
    occupied = len(np.unique(result['assignments'][result['assignments']>=0]))
    matched = None
    if occupied >= 10:
        matched = reconcile_known_identities(result['assignments'],logits[:,:10],10)
    report = dict(task='Office31 A->W',target_labels_used=False,new_training=False,new_model_forward=False,
        previous=dict(raw_K=old_capacity['K'],matched_K=old_identity['K'],settings=old_capacity['settings']),
        current=dict(raw_K=result['K'],matched_K=None if matched is None else matched['K'],
            reconciliation_status='insufficient occupied clusters for all-known assumption' if matched is None else 'available',
            counts=result['counts'].tolist(),noise_count=result['noise_count'],settings=settings),
        raw_partition_drift=compare_partitions(old_raw,result['assignments'],10),
        matched_partition_drift=None if matched is None else compare_partitions(old_matched,matched['assignments'],10),
        caveat='Temporal state diagnosis, not proof of semantic count, accuracy gain or stale-label causality. Representation, classifier and source-derived numeric calibration all evolve; fixed rule, not fixed numeric lambda. Never force K1 or infer success from lower K.')
    output.mkdir(parents=True)
    arrays = dict(centers=result['centers'],assignments=result['assignments'],target_paths=paths)
    if matched is not None:
        arrays['matched_assignments'] = matched['assignments']
        report['current']['cluster_to_identity'] = matched['cluster_to_identity']
    np.savez_compressed(output/'clusters.npz',**arrays)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    saved = Path('/content/drive/MyDrive/OSDA/runs/a2w-temporal-capacity-v1')
    if not saved.parent.is_dir():
        raise RuntimeError('Drive missing; local comparison preserved')
    saved.mkdir(exist_ok=False)
    for filename in ('clusters.npz','summary.json'):
        shutil.copyfile(output/filename,saved/filename)
    print('A2W_TEMPORAL_CAPACITY_COMPLETE',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
