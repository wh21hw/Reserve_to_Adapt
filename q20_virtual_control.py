"""Virtual module ablation: Q20 K-means on the same frozen full feature sets.

Not a bit-for-bit original RTA reproduction: original uses augmented batch banks.
"""
import json
from pathlib import Path
import faiss
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader
from alternating_konly import _FrozenImages


@torch.no_grad()
def refresh_q20_virtual(net,args,completed_epochs):
    rows = [row.rsplit(None,1) for row in Path(args.source).read_text().splitlines() if row.strip()]
    labels = np.asarray([int(row[1]) for row in rows])
    target_names = [row.rsplit(None,1)[0] for row in Path(args.target).read_text().splitlines() if row.strip()]
    device = next(net.parameters()).device
    modes = [(module,module.training) for module in net.modules()]
    net.eval()
    try:
        def extract(names,offset):
            loader = DataLoader(_FrozenImages(names,args.data_dir),batch_size=args.batch_size,
                shuffle=False,num_workers=4,pin_memory=True,
                generator=torch.Generator().manual_seed(completed_epochs+offset))
            return torch.cat([net(images.to(device))[1].cpu() for images in loader]).numpy()
        source = extract([row[0] for row in rows],100)
        target = extract(target_names,200)
    finally:
        for module,training in modes:
            module.training = training
    known = np.stack([source[labels==c].mean(0) for c in range(args.shared_classes)])
    clustering = faiss.Kmeans(256,20,niter=800,verbose=False,min_points_per_centroid=1,gpu=False)
    clustering.train(target)
    centers = clustering.centroids
    _,matched = linear_sum_assignment(np.linalg.norm(known[:,None]-centers[None],axis=-1))
    candidates = centers[[index for index in range(20) if index not in matched]]
    report = dict(completed_epochs=completed_epochs,Q=20,V=len(candidates),
                  K=args.all_classes-args.shared_classes,stage='Q20 virtual module control',
                  head_resized=False,target_labels_used=False,
                  representation='full current eval center-crop source/target features')
    with (Path(args.log_dir)/'fusion-history.jsonl').open('a') as stream:
        stream.write(json.dumps(report)+'\n')
    print('Q20_VIRTUAL',json.dumps(report),flush=True)
    return torch.from_numpy(candidates.copy()).to(device)
