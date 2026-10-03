"""Run one source-calibrated estimate in the selected legacy environment."""
import os
import subprocess

command = ['/content/rta-py38/bin/python', '-u', '/content/estimate_k_source_prior.py',
           '--source-run', '/content/imp-runs/konly-source-prior-v2/seed3',
           '--output', '/content/imp-runs/konly-estimate-v1/seed3-mobile']
status = subprocess.call(command, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
if status:
    raise RuntimeError('K estimate failed; preserve the output')
