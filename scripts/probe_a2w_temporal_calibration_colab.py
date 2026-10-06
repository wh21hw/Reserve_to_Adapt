"""One calibration-block counterfactual on final10 cached features.

Use source3 radius/birth cost, with current anchors/features/logits. No target
truth, parameter grid, new forward, training or promotion based on performance.
"""
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0,'/content')
from robust_capacity import fit_robust_capacity
from prototype_identity_reconciliation import reconcile_known_identities
from temporal_partition_report import compare_partitions


def main():
    drive = Path('/content/drive/MyDrive/OSDA/runs')
    temporal = drive/'a2w-temporal-capacity-v1'
    baseline = json.loads((temporal/'summary.json').read_text())
    old, current = baseline['previous']['settings'], baseline['current']['settings']
    with np.load(drive/'a2w-current-relation-snapshot-v1/features.npz',allow_pickle=False) as data:
        source, target = data['source'].astype(np.float64), data['target'].astype(np.float64)
        labels, logits, paths = data['source_labels'],data['target_logits'],data['target_paths']
    with np.load(temporal/'clusters.npz',allow_pickle=False) as data:
        reference = data['assignments']
    output = Path('/content/imp-runs/a2w-temporal-calibration-v1')
    if output.exists():
        raise FileExistsError('Keep existing counterfactual')
    rng = np.random.RandomState(2026)
    rows = []
    for c in range(10):
        ids = np.flatnonzero(labels == c)
        rng.shuffle(ids)
        rows.extend(ids[:max(1,int(.7*len(ids)))])
    rows = np.asarray(rows)
    anchors = np.stack([source[rows][labels[rows] == c].mean(0) for c in range(10)])
    counts = np.asarray([(labels[rows] == c).sum() for c in range(10)])
    if counts.tolist() != old['source_prior_counts'] or counts.tolist() != current['source_prior_counts']:
        raise ValueError('Source split protocol changed')
    result = fit_robust_capacity(target,anchors,old['lambda_radius'],prior_strength=counts,
        reference_samples=old['reference_samples'],birth_order='before_update',
        birth_penalty=old['birth_cost'],proposal_block_size=64)
    if not result['converged']:
        raise RuntimeError('Unconverged control; preserve evidence')
    occupied = len(np.unique(result['assignments'][result['assignments'] >= 0]))
    matched = reconcile_known_identities(result['assignments'],logits[:,:10],10) if occupied >= 10 else None
    report = dict(target_labels_used=False,new_training=False,new_forward=False,
        factor='Source-calibrated numeric radius/birth-cost block only',
        current_recalibrated=dict(raw_K=baseline['current']['raw_K'],matched_K=baseline['current']['matched_K'],
            lambda_radius=current['lambda_radius'],birth_cost=current['birth_cost']),
        current_with_source3_calibration=dict(raw_K=result['K'],matched_K=None if matched is None else matched['K'],
            lambda_radius=old['lambda_radius'],birth_cost=old['birth_cost'],counts=result['counts'].tolist()),
        partition_drift=compare_partitions(reference,result['assignments'],10),
        caveat='Descriptive calibration counterfactual, not a new tuned method. '
               'No claim that stale numeric calibration is preferable or that K equals true class count.')
    output.mkdir(parents=True)
    arrays = dict(assignments=result['assignments'],centers=result['centers'],target_paths=paths)
    if matched is not None:
        arrays['matched_assignments'] = matched['assignments']
    np.savez_compressed(output/'clusters.npz',**arrays)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    saved = drive/'a2w-temporal-calibration-v1'
    saved.mkdir(exist_ok=False)
    for name in ('clusters.npz','summary.json'):
        shutil.copyfile(output/name,saved/name)
    print('A2W_TEMPORAL_CALIBRATION_COMPLETE',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
