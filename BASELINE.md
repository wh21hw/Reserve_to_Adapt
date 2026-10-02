# RTA official-code baseline

Upstream: PRIS-CV/Reserve_to_Adapt, commit 8a4cd890b341a0dd4ed7c239533a1b6a9583fe8b.

No IMP changes. All upstream training, clustering and evaluation logic retained:
70 epochs, batch 64, source learning rate 5e-5, classifier/discriminator 5e-4,
SGD momentum 0.9, weight decay 5e-4, 10 known outputs + 2 unknown outputs,
20 target clusters, 800 Faiss iterations, warmiter 3.

Minimal changes: pretrained path via RTA_MODEL_PATH; unused TensorFlow logger
imports deferred; RTA_SEED; streaming logs; per-epoch metrics and checkpoints.
Evaluation uses no_grad to avoid retaining a graph between epochs on T4.
Neither evaluation predictions nor training gradients are changed by this.

Office-31 A->W uses the original data lists (known 0-9, unknown 20-30).
Paper Table I: OS*=92.2%, UNK=93.8%, HOS=93.0%.
Six-task average: OS*=93.4%, UNK=95.1%, HOS=94.2 +/- 0.8%.
These are separate targets: A->W is not the six-task average.

The official implementation reports the epoch with best target HOS. Record
both this best result and the final epoch; best-target selection uses target
labels and must not silently become the selection policy for new methods.

Paper section IV-A names loss coefficients 0.01/0.3/0.4, but the released
code uses CE + 0.01 virtual CE + 0.3 adversarial + entropy + unknown CE.
The first baseline follows released code without silently changing weights.

The runtime uses a T4 rather than the README's 3090 Ti. Python 3.8 and
PyTorch 1.7.1+cu110 / torchvision 0.8.2+cu110 are matched. Direct scientific
dependencies follow upstream requirements; unused notebook/cloud packages
are not installed. Bitwise identity across GPU types is not promised.
