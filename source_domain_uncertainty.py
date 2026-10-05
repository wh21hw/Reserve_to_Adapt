"""Proposed source-style predictive prior, not certified target shift variance.

For isotropic within-class feature variance rho_c, source mean uncertainty is
rho_c/n_c. Add class-specific style displacement variance xi_c before inverting:
kappa_c = rho_c / (rho_c/n_c + xi_c).
The common class-balanced displacement is removed because shared delta already
models it. Style views are paired source images, not target images or labels.
"""
import numpy as np


def calibrate_style_prior(source, labels, styled_source):
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(labels)
    views = np.asarray(styled_source, dtype=np.float64)
    if (x.ndim != 2 or y.shape != (len(x),) or y.dtype.kind not in 'iu'
            or views.ndim != 3 or views.shape[1:] != x.shape or not len(views)
            or not np.isfinite(x).all() or not np.isfinite(views).all()):
        raise ValueError('Expected finite paired source views [styles,N,D] and integer source labels')
    classes = np.unique(y)
    if not len(classes) or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError('Source identities must be contiguous')
    counts = np.array([(y == c).sum() for c in classes])
    if (counts < 2).any():
        raise ValueError('Within-class variance requires at least two source images')
    means = np.stack([x[y == c].mean(0) for c in classes])
    rho = np.array([x[y == c].var(0, ddof=1).mean() for c in classes])
    rho = np.maximum(rho, 1e-8)
    styled_means = np.stack([np.stack([v[y == c].mean(0) for c in classes]) for v in views])
    displacements = styled_means-means[None]
    common = displacements.mean(1)
    residual = displacements-common[:,None]
    xi = (residual*residual).mean(axis=(0,2))
    kappa = rho/(rho/counts+xi)
    return dict(source_prior_precision=kappa.tolist(), source_counts=counts.tolist(),
        within_class_variance=rho.tolist(), style_domain_variance=xi.tolist(),
        common_style_shifts=common.tolist(), styles=len(views), classes=len(classes),
        target_data_used=False,
        caveat='Source style perturbations are an assumed surrogate for domain changes, not evidence that target shift covariance is identified. Same-class random effects remain an isotropic approximation. No target threshold fitting.')
