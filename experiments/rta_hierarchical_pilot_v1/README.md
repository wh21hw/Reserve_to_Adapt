# Hierarchical unknown pilot

Office31 A→W, seed1, fixed warm epoch4, two adaptation epochs4–5.
Three arms: original two-slot directions, candidate top2 directions, candidate18.
Unknown components are latent, NOT semantic classes. Fixed conditional weights
pi=1/K; semantic unknown logit=logsumexp(raw_unknown+log(pi))+log(2).

Source CE, target selected-unknown CE, entropy, virtual CE and prediction use
known10+ONE unknown semantic outputs. Conditional latent allocation is retained
for diagnosis, without adding a separate structural loss or DP posterior update.
Released relation KL/GMM/sample selection, adversary, virtual templates and loss
coefficients remain. Fixed warm handoff is controlled, not uninterrupted resume.

Mixture priors are checkpoint buffers; strict replay needs this model class.
New warm loading permits ONLY the unknown_log_weights missing key. Optimizer
handoff retains known/non-head momentum and resets unknown momentum in all arms.
Workers run in fresh interpreters and assert module paths to avoid notebook cache.

Output/loss/shared-logit gradients are invariant to splitting a component and its
prior mass; independent duplicated-parameter SGD trajectories are NOT asserted
invariant. This is a finite discriminative mixture, not semantic count recovery
or full Bayesian DPMM. Keep all outcomes; do not choose capacity from target HOS.
Three-task full matched baseline/IMP experiments remain required.
