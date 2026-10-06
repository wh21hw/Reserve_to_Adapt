"""One source-calibrated capacity update; cluster labels never train RTA.

The fixed arm performs the identical extraction/inference, but does not resize.
Only state correspondence, not head direction initialization, uses assignments.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from alternating_konly import _FrozenImages, resize_unknown_head
from source_precision_capacity import estimate_source_cost_capacity
from prototype_identity_reconciliation import reconcile_known_identities


@torch.no_grad()
def apply_inferred_capacity(cls, optimizer, args, assignments, inferred_K,
                            old_probabilities, apply):
    """Use a frozen candidate assignment only to transport head/SGD state."""
    old_K = cls.fc.out_features - args.shared_classes
    if args.all_classes != cls.fc.out_features:
        raise ValueError('Live classifier/args dimensions differ')
    if not apply:
        return dict(old_K=old_K, new_K=old_K, inferred_K=inferred_K,
                    changed=False, applied=False)
    if inferred_K is None or inferred_K <= 0:
        raise RuntimeError('Undefined/zero inferred capacity; do not force K1')
    ids = np.asarray(assignments)
    if ids.shape != (len(old_probabilities),) or ids.dtype.kind not in 'iu':
        raise ValueError('Assignment/probability rows differ')
    if (ids < -1).any() or (ids >= args.shared_classes+inferred_K).any():
        raise ValueError('Inferred identities outside proposed head')
    membership = torch.zeros(len(ids),args.shared_classes+inferred_K,
                             dtype=torch.float64)
    rows = np.flatnonzero(ids >= 0)
    membership[torch.from_numpy(rows),torch.from_numpy(ids[rows])] = 1.
    report = resize_unknown_head(cls,optimizer,args.shared_classes,membership,
                                 old_probabilities)
    args.all_classes = args.shared_classes+inferred_K
    report.update(applied=True,inferred_K=inferred_K,
                  assignments_used_as_training_labels=False)
    return report


@torch.no_grad()
def refresh_boundary(net, cls, optimizer_wrapper, args, completed_epochs, apply):
    if completed_epochs != 10:
        raise ValueError('This declared experiment refreshes only after epoch10')
    output = Path(args.log_dir)/'capacity-after-010'
    output.mkdir(exist_ok=False)
    source_rows = [line.rsplit(None,1) for line in Path(args.source).read_text().splitlines() if line.strip()]
    source_names = [row[0] for row in source_rows]
    source_labels = np.asarray([int(row[1]) for row in source_rows],dtype=np.int64)
    # Discard target semantic column at the reader boundary.
    target_names = [line.rsplit(None,1)[0] for line in Path(args.target).read_text().splitlines() if line.strip()]
    if (len(source_names),len(target_names),args.shared_classes) != (958,564,10):
        raise ValueError('Require the complete declared Office31 A2W inputs')
    was_training = net.training
    device = cls.fc.weight.device
    net.eval()
    try:
        arrays = {}
        for offset,(split,names) in enumerate((('source',source_names),('target',target_names))):
            loader = DataLoader(_FrozenImages(names,args.data_dir),batch_size=64,
                shuffle=False,num_workers=4,pin_memory=device.type=='cuda',
                generator=torch.Generator().manual_seed(3102026+offset))
            features,logits,probabilities = [],[],[]
            for images in loader:
                _,feature,logit,probability = net(images.to(device))
                if not all(torch.isfinite(value).all() for value in (feature,logit,probability)):
                    raise RuntimeError('Nonfinite current representation/head output')
                features.append(feature.cpu().numpy())
                logits.append(logit.cpu().numpy())
                probabilities.append(probability.cpu())
            arrays[split] = np.concatenate(features)
            arrays[split+'_logits'] = np.concatenate(logits)
            if split == 'target':
                old_probabilities = torch.cat(probabilities)
    finally:
        net.train(was_training)
    result,settings = estimate_source_cost_capacity(arrays['source'],source_labels,
        arrays['target'],proposal_block_size=64)
    occupied = len(np.unique(result['assignments'][result['assignments'] >= 0]))
    matched = reconcile_known_identities(result['assignments'],arrays['target_logits'][:,:10],10) if occupied >= 10 else None
    inferred_K = None if matched is None else matched['K']
    report = dict(completed_epochs=10,raw_K=result['K'],inferred_K=inferred_K,
        apply_requested=apply,applied=False,settings=settings,occupied_clusters=occupied,
        noise_count=result['noise_count'],target_labels_used=False,
        additional_training_objective=False,prediction_change=False,
        Q=args.virtual_clusters,source_and_target_same_current_model=True)
    if matched is not None:
        report['cluster_to_identity'] = matched['cluster_to_identity']
    arrays.update(source_labels=source_labels,target_paths=np.asarray(target_names),
        assignments=result['assignments'],centers=result['centers'])
    if matched is not None:
        arrays['matched_assignments'] = matched['assignments']
    # Save inference evidence before a potentially undefined resize, too.
    np.savez_compressed(output/'snapshot.npz',**arrays)
    (output/'estimate.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    head = apply_inferred_capacity(cls,optimizer_wrapper.optimizer,args,
        None if matched is None else matched['assignments'],inferred_K,
        old_probabilities,apply)
    report['head'] = head
    report['applied'] = head['applied']
    (output/'estimate.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    config_path = Path(args.log_dir)/'config.json'
    config = json.loads(config_path.read_text())
    config.setdefault('initial_all_classes',config['all_classes'])
    config.update(all_classes=args.all_classes,inferred_K_after10=inferred_K,
                  capacity_update_applied=apply)
    config_path.write_text(json.dumps(config,indent=2,allow_nan=False))
    print('CAPACITY_BOUNDARY_COMPLETE',json.dumps(report,allow_nan=False),flush=True)
    return report
