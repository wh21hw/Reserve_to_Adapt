# Colab pipeline

Code: GitHub branch `codex/colab-pipeline`. This code-only branch contains the local
IMP research edits and Colab portability changes. Original full training defaults
remain 70 epochs; the pipeline test uses K-means, 5 epochs and 5 batches per epoch.
It crosses the warmup boundary and verifies training, evaluation and checkpoint loading.
These smoke-test scores are not paper reproduction results.

Data: Google Drive `OSDA/datasets/office31_images.tar` and `data-manifest.json`.
Weights: `OSDA/pretrained/resnet50-19c8e357.pth`.
Outputs: `OSDA/runs/a2w-pipeline-seed1/`.
Generate archives locally with `powershell -File scripts/pack-data.ps1`.

CLI is installed in WSL at `/home/wanghao21/.local/opt/murphy-colab-cli`.
Node is installed at `/home/wanghao21/.local/opt/osda-node/bin/node`.
Use `node --use-env-proxy dist/index.js` from the CLI directory, or the `colab`
launcher in `~/.local/bin` if configured.

Local CLI adjustments: OAuth timeout is 15 minutes; background OAuth inherits
Node's proxy flag; upload chunks are 5 MiB with concurrency 2. These changes fix
browser callback timing, proxy authentication and excessive upload memory use.
If initial kernel creation fails while the VM is starting, restart the local CLI
daemon and reattach to the existing VM rather than allocating another instance.

When the shared Drive API client hits quota, upload inputs with `colab fs upload`
to `/content/osda-upload`, execute `scripts/mount_drive.py`, then execute
`scripts/persist_data.py`. This writes and verifies the files on mounted Drive.

1. `colab auth login`, then `colab drive login` (human browser authorization).
2. Create the Drive folders and upload the archive, manifest and weights.
3. `colab runtime available`, then `colab runtime create --accelerator T4`.
4. Mount Drive by running `from google.colab import drive; drive.mount('/content/drive')`
   through `colab exec`. Complete the browser consent when prompted.
5. Execute `scripts/remote_pipeline.py` with `colab exec -e ENDPOINT --background -f FILE`.
6. Inspect `colab exec attach ID --tail 30`. Success prints `PIPELINE_SUCCESS`.
7. Retrieve `metrics.json`, `environment.json` and logs; retain checkpoints on Drive.

The CLI file execution sends one file's content, not the whole repository. The remote
pipeline explicitly clones the code-only GitHub branch and verifies data SHA-256.
Pin `OSDA_REVISION` to a release branch/tag for subsequent reproducible runs; every
run records the resolved Git commit and runtime versions in `environment.json`.

IMP currently can collapse to fewer than 11 clusters. It fails explicitly when no
unknown prototype remains. Its cluster count does not determine the fixed classifier
output width. Warmup classifier reinitialization uses a fixed-size K-means clustering;
this compatibility choice must be reviewed before claiming an IMP algorithm result.
