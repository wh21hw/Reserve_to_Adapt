# Published-code multi-task baseline v1 (not yet trained)

Copied from frozen `rta_l4_control_v1`; root user training files are untouched.
Loss coefficients, preprocessing, optimizer, warmup and mixture schedule remain
the published-code control, not an equation-faithful paper reimplementation.

Changes: explicit task protocol; source remapping; target training labels replaced
by a constant unknown sentinel; original labels retained only for evaluation;
macro-per-original-class UNK; normalized image root; no output overwrite.
`--virtual-clusters` is mandatory: Office31 published code used 20, while the
OfficeHome/VisDA exact published configuration has not been established. Do not
silently present a selected cross-dataset count as the author's configuration.

VisDA requires verified `--class-map` JSON (original IDs to names) and explicit
`--accept-unresolved-visda-resnet`. The latter permits a controlled ResNet-50
comparison, not a claim to reproduce the VGGNet Table III result. VGG is rejected.

Required launch inputs: --task, --source, --target, --data_dir, --log_dir,
--virtual-clusters, and RTA_MODEL_PATH. Seed/RTA_EPOCHS retain the old interface.
Use a fresh subprocess on Colab to avoid cached imports.

Syntax/stdlib protocol tests are not a GPU data-loader or training validation.
Do not launch full experiments until actual batch forward/backward checks pass.
