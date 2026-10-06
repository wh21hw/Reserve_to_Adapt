"""Label-free partition drift; unknown IDs may permute between snapshots."""
import numpy as np


def _adjusted_rand(left,right):
    _,a = np.unique(left,return_inverse=True)
    _,b = np.unique(right,return_inverse=True)
    table = np.zeros((a.max()+1,b.max()+1),dtype=np.int64)
    np.add.at(table,(a,b),1)
    choose2 = lambda values: int(np.sum(values*(values-1)//2))
    intersection = choose2(table)
    pairs_a,pairs_b = choose2(table.sum(1)),choose2(table.sum(0))
    total = len(left)*(len(left)-1)//2
    expected = pairs_a*pairs_b/total
    maximum = (pairs_a+pairs_b)/2
    return 1.0 if maximum == expected else float((intersection-expected)/(maximum-expected))


def compare_partitions(previous,current,known_classes):
    """No semantic target truth input. Noise is -1, known IDs are 0..C-1."""
    before,after = np.asarray(previous),np.asarray(current)
    if (before.ndim != 1 or before.shape != after.shape or not len(before)
            or before.dtype.kind not in 'iu' or after.dtype.kind not in 'iu'
            or (before < -1).any() or (after < -1).any()
            or not isinstance(known_classes,int) or known_classes < 1):
        raise ValueError('Expected aligned integer cluster vectors and positive C')
    assigned = (before >= 0)&(after >= 0)
    both_known = assigned&(before < known_classes)&(after < known_classes)
    return dict(rows=len(before),jointly_assigned_rows=int(assigned.sum()),
        jointly_assigned_fraction=float(assigned.mean()),
        adjusted_rand_on_jointly_assigned=_adjusted_rand(before[assigned],after[assigned]) if assigned.sum()>=2 else None,
        noise_status_changed_fraction=float(((before == -1)!=(after == -1)).mean()),
        unknown_candidate_status_changed_fraction=float(((before >= known_classes)!=(after >= known_classes)).mean()),
        known_identity_agreement_on_both_known=float((before[both_known]==after[both_known]).mean()) if both_known.any() else None,
        target_labels_used=False,
        interpretation='ARI ignores ID permutations; high drift is not evidence of worse semantics or a cause of accuracy loss. Noise excluded from ARI and coverage reported separately.')
