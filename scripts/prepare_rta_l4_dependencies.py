"""Install only missing FAISS; do not upgrade the verified torch/numpy stack."""
import importlib.util
import json
import subprocess
import sys
import numpy as np
import torch

before = dict(torch=torch.__version__, numpy=np.__version__)
if importlib.util.find_spec('faiss') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'faiss-cpu==1.12.0'], check=True)
import faiss
assert before == dict(torch=torch.__version__, numpy=np.__version__)
data = np.random.default_rng(1).normal(size=(64, 8)).astype('float32')
model = faiss.Kmeans(8, 4, niter=2, min_points_per_centroid=1, gpu=False)
model.train(data)
assert model.centroids.shape == (4, 8) and np.isfinite(model.centroids).all()
print('FAISS_PREFLIGHT_PASS', json.dumps(dict(before, faiss=faiss.__version__)), flush=True)
