"""Seed2/3 own-cache label-free inference; CPU only and no SGD."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings
import zipfile
import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,filename,sha):
    path=Path('/content')/filename
    assert digest(path)==sha
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(2,3),required=True)
    parser.add_argument('--feature-sha256',required=True)
    args=parser.parse_args()
    assert len(args.feature_sha256)==64 and all(c in '0123456789abcdef' for c in args.feature_sha256)
    torch.set_num_threads(2)
    imp=load('own_seed_imp','matched_source_anchored_imp.py','ec38a017ae2b91755499e136d309fc2af10eff802aa7a114b12fa2cd51943a4f')
    relation=load('own_seed_relation','matched_relation_gate.py','1556c60c5536b508e7f631fdd0dcac434a43749d99e08850eba2292bb26de3da')
    feature_root=Path(f'/content/imp-runs/multitask-a2w-seed{args.seed}-warm-features-v1')
    feature_path=feature_root/'features.npz'
    manifest=json.loads((feature_root/'manifest.json').read_text())
    assert manifest['seed']==args.seed and manifest['epoch']==4 and manifest['target_labels_in_features'] is False
    expected_warm={2:'e16e895e1bdaec17f14dd4eda7cc81f8b05ad1c1c6d76145bbd3088aa1d2fea0',
                   3:'f74f473b8d5ae58bac64916db8dc5ad153ba2b54c5e4688227c460a96956152b'}
    assert manifest['checkpoint_sha256']==expected_warm[args.seed]
    assert manifest['files']['features.npz']==args.feature_sha256==digest(feature_path)
    data=np.load(feature_path,allow_pickle=False)
    assert set(data.files)=={'source','target','source_labels','source_logits','target_logits'}
    assert all(np.isfinite(data[k]).all() for k in data.files)
    source,target=torch.from_numpy(data['source']),torch.from_numpy(data['target'])
    labels=torch.from_numpy(data['source_labels'])
    assert source.shape==(958,256) and target.shape==(564,256)
    assert torch.equal(torch.unique(labels),torch.arange(10))
    centers=torch.stack([source[labels==c].mean(0) for c in range(10)])
    residual=source-centers[labels]
    threshold=max(float(np.quantile(residual.square().sum(1).numpy(),.99)),1e-8)
    variance=max(float(residual.square().mean()),1e-8)
    scores=relation.source_relation_scores(torch.from_numpy(data['source_logits']),labels,
                                          torch.from_numpy(data['target_logits']),10)['scores'].numpy()
    mixture=BayesianGaussianMixture(n_components=4,max_iter=800,random_state=args.seed)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always');mixture.fit(scores[:,None])
    known=int(mixture.means_.argmin())
    compatibility=torch.from_numpy(mixture.predict_proba(scores[:,None])[:,known]).to(target.dtype)
    output=Path(f'/content/imp-runs/matched-warm-candidates-seed{args.seed}-v1')
    assert not output.exists(),'Refusing overwrite'
    output.mkdir(parents=True)
    report=dict(seed=args.seed,feature_sha256=args.feature_sha256,warm_checkpoint_sha256=expected_warm[args.seed],
                threshold=threshold,variance=variance,prior_strength=5,steps=5,capacity=100,
                birth_unknown_min=.5,support_min=5,gate_seed=args.seed,gate_converged=bool(mixture.converged_),
                gate_means=mixture.means_.ravel().tolist(),warnings=[str(m.message) for m in messages],
                semantic_unknown_count=None,optimizer_steps=0,target_labels_used=False,trials=[],
                caveat='In-sample source calibration; subset audit inherits full gate; not a DP posterior or semantic count')
    orders=[('original',np.arange(len(target))),('reverse',np.arange(len(target)-1,-1,-1)),
            ('shuffle1',np.random.default_rng(1).permutation(len(target)))]
    orders += [(f'subset80-seed{s}',np.random.default_rng(s).permutation(len(target))[:int(.8*len(target))]) for s in (1,2,3)]
    original=None
    for name,order in orders:
        model=imp.SourceAnchoredIMP(threshold,variance,prior_strength=5,steps=5,max_prototypes=100)
        result=model.fit(target[order.copy()],centers,compatibility[order.copy()],birth_strategy='farthest')
        prototypes=torch.cat([result['known_prototypes'],result['candidate_prototypes']])
        responsibility=model._assign(target,prototypes,10,compatibility)
        mass=responsibility.sum(0)[10:]
        equal=original is None or torch.equal(original,responsibility)
        if original is None:original=responsibility.clone()
        row=dict(name=name,fitting_samples=len(order),candidate_count=result['candidate_count'],
                 supported_components=int((mass>=5).sum()),masses=mass.tolist(),permutation_equal=equal,
                 finite=bool(torch.isfinite(responsibility).all()),
                 normalized=bool(torch.allclose(responsibility.sum(1),torch.ones(len(target)),atol=1e-5)))
        assert row['finite'] and row['normalized']
        if len(order)==len(target):assert equal,'Permutation changed candidate inference'
        report['trials'].append(row)
        np.savez_compressed(output/f'{name}.npz',candidates=result['candidate_prototypes'].numpy(),
                            known=result['known_prototypes'].numpy(),responsibilities=responsibility.numpy(),
                            known_compatibility=compatibility.numpy(),fit_indices=order)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('OWN_SEED_CANDIDATES',json.dumps(row),flush=True)
    # This preserves all proposals, but downstream uses original only; no result selection.
    archive=Path(f'/content/matched-warm-candidates-seed{args.seed}-v1.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
        for path in output.iterdir():stream.write(path,path.name)
        stream.write(__file__,'candidate-worker.py')
    print('OWN_SEED_CANDIDATE_AUDIT_PASS',archive.name,digest(archive),flush=True)

if __name__=='__main__':main()
