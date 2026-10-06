"""Use a single saved final70/K5 cache; no model, labels, training or refitting."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--boundary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--drive-output', required=True)
    args = parser.parse_args()
    cache = Path(args.cache)
    manifest = json.loads((cache/'manifest.json').read_text())
    if (not manifest['complete'] or manifest['target_labels_used'] or
            (manifest['epoch'], manifest['C'], manifest['K']) != (70,10,5)):
        raise ValueError('Require the declared complete final70/K5 label-blind cache')
    output = Path(args.output)
    durable = Path(args.drive_output)
    if output.exists() or durable.exists():
        raise FileExistsError('Preserve existing diagnostic results')
    with np.load(cache/'features.npz', allow_pickle=False) as data:
        target_logits = data['target_logits']
        source_logits = data['source_logits']
        target_paths = data['target_paths']
    boundary = Path(args.boundary)
    estimate = json.loads((boundary/'estimate.json').read_text())
    mapping = estimate['head']['row_mapping']
    if (estimate['target_labels_used'] or estimate['inferred_K'] != 5 or
            mapping != [[10,10],[11,11]] or estimate['head']['new_random_rows'] != 3):
        raise ValueError('Unexpected expansion correspondence')
    with np.load(boundary/'snapshot.npz', allow_pickle=False) as data:
        if not np.array_equal(target_paths, data['target_paths']):
            raise ValueError('Final and boundary sample order differ')
        assignments = data['matched_assignments']
    report = dict(checkpoint=manifest['checkpoint'], epoch=70, C=10, K=5,
        new_rows=[12,13,14], retained_rows=[10,11], target_labels_used=False,
        accuracy_evaluated=False, new_training=False, K_refitted=False,
        splits={}, caveat='One frozen center-crop eval snapshot, not the historical '
        'training augmentation/BN/gate stream. No claim that an unused eval row '
        'never had gradients or positive pseudo-labels during training.')
    for split, logits in (('target',target_logits), ('source',source_logits)):
        if logits.shape[1] != 15 or not np.isfinite(logits).all():
            raise ValueError('Require finite full C10+K5 logits')
        full = logits.argmax(1)
        unknown_only = logits[:,10:].argmax(1)+10
        pruned = logits[:,:12].argmax(1)
        report['splits'][split] = dict(rows=len(logits),
            full_head_winners=np.bincount(full,minlength=15).tolist(),
            unknown_only_winners=np.bincount(unknown_only-10,minlength=5).tolist(),
            predicted_unknown_rows=int((full >= 10).sum()),
            new_rows_full_winners=int((full >= 12).sum()),
            new_rows_unknown_only_winners=int((unknown_only >= 12).sum()),
            deleting_new_rows_changes_any_prediction=int((full != pruned).sum()),
            deleting_new_rows_changes_binary_rejection=int(((full >= 10)!=(pruned >= 10)).sum()),
            new_row_max_logit_margin_over_retained_unknown=(
                logits[:,12:]-logits[:,10:12].max(1,keepdims=True)).max(0).tolist())
    # Frozen epoch10 candidate IDs are only a descriptive cross-tab, not labels.
    full = target_logits.argmax(1)
    table = []
    for candidate in range(10,15):
        mask = assignments == candidate
        table.append(dict(candidate_id=candidate,rows=int(mask.sum()),
            final_known_predictions=int((full[mask] < 10).sum()),
            final_unknown_slot_counts=np.bincount(full[mask][full[mask]>=10]-10,minlength=5).tolist()))
    report['epoch10_candidate_to_final_head'] = table
    report['hypothesis_tests'] = dict(
        H1_unused_new_rows_falsified=report['splits']['target']['new_rows_unknown_only_winners']>0,
        H2_new_rows_have_final_functional_use_falsified=report['splits']['target']['new_rows_full_winners']==0)
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    durable.mkdir(parents=True)
    shutil.copyfile(output/'summary.json',durable/'summary.json')
    print('NEW_SLOT_USAGE_DIAGNOSTIC_COMPLETE',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
