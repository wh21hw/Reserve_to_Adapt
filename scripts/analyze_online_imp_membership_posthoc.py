"""Posthoc semantic annotation of saved epoch10-start memberships only.

Never imported by training; does not select thresholds, K or configurations.
No model evaluation. It cannot reveal actual per-sample training exposure.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--archive',required=True)
parser.add_argument('--target-list',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--arms',nargs=2,default=['self_label','structure_label'])
args=parser.parse_args()
rows=[line.rsplit(None,1) for line in Path(args.target_list).read_text().splitlines() if line.strip()]
paths=[row[0] for row in rows]
truth=np.asarray([int(row[1]) for row in rows])
is_known=truth<10
report=dict(stage='Start of epoch10, not final10 network',target_truth_used='Posthoc annotation only',
    actual_selected_sample_exposure_available=False,arms={})
with zipfile.ZipFile(args.archive) as bundle:
    for arm in args.arms:
        with np.load(io.BytesIO(bundle.read(arm+'/office31-a2w_seed3/current-structure.npz'))) as data:
            if data['target_paths'].tolist()!=paths:
                raise ValueError('Snapshot and posthoc target paths differ')
            assignment=data['assignments']; reliable=data['reliable']
            support_mask=data['target_support_mask'] if 'target_support_mask' in data.files else None
            teacher_space=str(data['teacher_space'].item()) if 'teacher_space' in data.files else 'bottleneck'
        clusters=[]
        for j in range(len(reliable)):
            mask=assignment==10+j
            clusters.append(dict(slot=10+j,members=int(mask.sum()),known_members=int((mask&is_known).sum()),
                unknown_members=int((mask&~is_known).sum()),label_eligible=bool(reliable[j]),
                virtual_eligible=bool(reliable[j]) if teacher_space=='bottleneck' else None,
                semantic_counts={str(c):int((mask&(truth==c)).sum()) for c in np.unique(truth[mask])}))
        candidate=assignment>=10
        eligible=np.zeros(len(assignment),dtype=bool)
        eligible[candidate]=reliable[assignment[candidate]-10]
        report['arms'][arm]=dict(clusters=clusters,
            teacher_space=teacher_space,
            known_candidate_fraction=float(candidate[is_known].mean()),
            unknown_candidate_fraction=float(candidate[~is_known].mean()),
            known_reliable_candidate_fraction=float(eligible[is_known].mean()),
            unknown_reliable_candidate_fraction=float(eligible[~is_known].mean()),
            known_retained_identity_accuracy=float((assignment[is_known]==truth[is_known]).mean()))
        if support_mask is not None:
            report['arms'][arm]['calibration_support']=dict(count=int(support_mask.sum()),
                known_count=int((support_mask&is_known).sum()),unknown_count=int((support_mask&~is_known).sum()),
                unknown_fraction=float((support_mask&~is_known).sum()/max(int(support_mask.sum()),1)),
                true_known_coverage=float(support_mask[is_known].mean()))
        exposure_files=[name for name in bundle.namelist() if name.startswith(arm+'/office31-a2w_seed3/sample-exposure-')]
        if exposure_files:
            selected=np.zeros(len(truth),dtype=np.int64)
            overridden=np.zeros(len(truth),dtype=np.int64)
            final_exposure=None
            for name in sorted(exposure_files):
                with np.load(io.BytesIO(bundle.read(name))) as exposure:
                    if exposure['target_paths'].tolist()!=paths:raise ValueError('Exposure target paths differ')
                    if not bool(exposure['unknown_ce_active']):continue
                    selected+=exposure['selected'];overridden+=exposure['overridden']
                    final_exposure={key:exposure[key].copy() for key in ('selected','overridden')}
            report['arms'][arm]['active_exposure']=dict(
                selected_total=int(selected.sum()),selected_known=int(selected[is_known].sum()),
                selected_unknown=int(selected[~is_known].sum()),
                overridden_total=int(overridden.sum()),overridden_known=int(overridden[is_known].sum()),
                overridden_unknown=int(overridden[~is_known].sum()),
                unique_unknown_selected=int((selected[~is_known]>0).sum()),
                unique_unknown_overridden=int((overridden[~is_known]>0).sum()),
                unknown_class_override_counts={str(c):int(overridden[truth==c].sum()) for c in np.unique(truth[~is_known])},
                actual_image_seen_counts_available=False)
            if final_exposure is not None:
                report['arms'][arm]['epoch10_exposure']=dict(
                    unknown_label_eligible_members=int((eligible&~is_known).sum()),
                    unknown_eligible_without_ce=int((eligible&~is_known&(final_exposure['selected']==0)).sum()),
                    known_overrides=int(final_exposure['overridden'][is_known].sum()),
                    unknown_overrides=int(final_exposure['overridden'][~is_known].sum()))
Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({arm:{k:v for k,v in values.items() if k!='clusters'} for arm,values in report['arms'].items()},indent=2))
