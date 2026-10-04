"""Frozen v1 K-estimation protocol; only the returned count enters K-only RTA.

Source labels calibrate anchors, scales and empirical prior precision. No target
labels accepted. K is modeling capacity, not a certified semantic class count.
"""
import numpy as np
from robust_capacity import fit_robust_capacity


def estimate_capacity(source, source_labels, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    labels = np.asarray(source_labels)
    if source.ndim != 2 or labels.shape != (len(source),) or labels.dtype.kind not in 'iu':
        raise ValueError('Expected source feature matrix and integer label vector')
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(len(classes))) or len(classes) == 0:
        raise ValueError('Source classes must be contiguous 0..C-1')
    rng = np.random.RandomState(2026)
    calibration, validation = [], []
    for c in classes:
        indices = np.flatnonzero(labels == c)
        if len(indices) < 2:
            raise ValueError('At least two source examples required per class')
        rng.shuffle(indices)
        cut = max(1, int(.7*len(indices)))
        calibration.extend(indices[:cut])
        validation.extend(indices[cut:])
    calibration, validation = np.asarray(calibration), np.asarray(validation)
    anchors = np.stack([source[calibration][labels[calibration] == c].mean(0) for c in classes])
    counts = np.asarray([(labels[calibration] == c).sum() for c in classes])
    residual = source[calibration]-anchors[labels[calibration]]
    penalty = max(float(np.quantile((residual**2).sum(1), .99)), 1e-8)
    result = fit_robust_capacity(target, anchors, penalty, prior_strength=counts,
                                 reference_samples=float(counts.mean()), birth_order='before_update')
    if not result['converged']:
        raise RuntimeError('Capacity inference did not converge; do not silently pass count to RTA')
    settings = dict(version='source-precision-capacity-v1', C=len(classes), K=result['K'],
                    calibration_fraction=.7, calibration_seed=2026,
                    calibration_rows=len(calibration), validation_rows=len(validation),
                    source_rows=len(source), target_rows=len(target), penalty=penalty,
                    quantile=.99, source_prior_counts=counts.tolist(),
                    reference_samples=float(counts.mean()), known_centers_fixed=False,
                    birth_order='before_update', prediction_change=False,
                    classifier_prototype_initialization=False, target_labels_used=False,
                    semantic_unknown_count=None,
                    zero_capacity_policy='return 0 explicitly; RTA caller must not force K1',
                    caveat='empirical power-weighted objective, not full DP posterior')
    return result, settings


def estimate_source_cost_capacity(source, source_labels, target, proposal_block_size=None):
    """Previously declared source-calibrated birth cost; target labels not accepted."""
    source=np.asarray(source,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64)
    labels=np.asarray(source_labels)
    if proposal_block_size is not None and (isinstance(proposal_block_size,bool)
            or not isinstance(proposal_block_size,(int,np.integer)) or proposal_block_size<1):
        raise ValueError('proposal_block_size must be a positive integer or None')
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1] or labels.shape!=(len(source),) or labels.dtype.kind not in 'iu':
        raise ValueError('Expected matching feature matrices and integer source labels')
    classes=np.unique(labels)
    if not np.array_equal(classes,np.arange(len(classes))) or not len(classes):
        raise ValueError('Source classes must be contiguous')
    rng=np.random.RandomState(2026);train,calibration=[],[]
    for c in classes:
        ids=np.flatnonzero(labels==c);rng.shuffle(ids)
        cut=max(1,int(.7*len(ids)));middle=cut+(len(ids)-cut)//2
        if middle<=cut:raise ValueError('Insufficient source calibration examples')
        train.extend(ids[:cut]);calibration.extend(ids[cut:middle])
    train,calibration=map(np.asarray,(train,calibration))
    anchors=np.stack([source[train][labels[train]==c].mean(0) for c in classes])
    counts=np.array([(labels[train]==c).sum() for c in classes])
    radius=max(float(np.quantile(((source[train]-anchors[labels[train]])**2).sum(1),.99)),1e-8)
    reference=float(counts.mean())
    def distance(left,right):
        return np.maximum((left*left).sum(1)[:,None]+(right*right).sum(1)[None]-2*left.dot(right.T),0.)
    known=source[calibration];residual=np.minimum(distance(known,anchors).min(1),radius)
    if proposal_block_size is None:
        gain=reference/len(known)*np.maximum(residual[:,None]-distance(known,known),0.).sum(0)
    else:
        gain=np.empty(len(known),dtype=np.float64)
        for start in range(0,len(known),proposal_block_size):
            end=min(start+proposal_block_size,len(known))
            gain[start:end]=reference/len(known)*np.maximum(
                residual[:,None]-distance(known,known[start:end]),0.).sum(0)
    cost=max(float(gain.max())*(1+1e-6),1e-8)
    result=fit_robust_capacity(target,anchors,radius,prior_strength=counts,reference_samples=reference,
        birth_order='before_update',birth_penalty=cost,proposal_block_size=proposal_block_size)
    if not result['converged']:raise RuntimeError('Unconverged capacity estimate')
    settings=dict(version='source-calibrated-birth-cost-v1',C=len(classes),K=result['K'],
        calibration_seed=2026,anchor_rows=len(train),birth_cost_rows=len(calibration),
        source_rows=len(source),target_rows=len(target),lambda_radius=radius,birth_cost=cost,
        source_prior_counts=counts.tolist(),reference_samples=reference,known_centers_fixed=False,
        birth_order='before_update',target_labels_used=False,semantic_unknown_count=None,
        proposal_block_size=proposal_block_size,
        zero_capacity_policy='Return zero; never force K1',
        caveat='Same source maximum initial birth-gain formula as prior controls; empirical objective, not full DP posterior')
    return result,settings
