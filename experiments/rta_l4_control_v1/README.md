# RTA L4 warmup control snapshot

This is an isolated snapshot of the audited released-code baseline, not a new
algorithm. Upstream base: `8a4cd890b341a0dd4ed7c239533a1b6a9583fe8b`.
Existing baseline logging/seed/path/no_grad changes are retained.

Additional changes: `RTA_EPOCHS` limits epochs (default remains 70); the pinned
legacy ImageNet weight is hash-checked and loaded with `weights_only=False`.
No loss, clustering, matching, or unknown-head algorithm change was made.

The four-epoch L4 run uses torch 2.11.0+cu130, torchvision 0.26.0+cu130,
numpy 2.1.3 and faiss-cpu 1.12.0, unlike the historical T4 baseline.
Its fixed final checkpoint is used for diagnostics; `best.pt` is not an input.
This is not a complete 70-epoch reproduction or a controlled loss ablation
against the earlier three-epoch source-only diagnostic.

See `IMP_EXPERIMENTS.md` for run IDs, artifact hashes, limitations and next gates.
Data, pretrained weights and generated artifacts are deliberately excluded from
the Git code snapshot. Run using `scripts/run_rta_l4_warmup_control.py` after
input/environment preflight; the runtime copy lives under `/content/rta-l4-control-v1`.
