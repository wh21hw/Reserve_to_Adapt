"""Readout diagnostic of recorded final probabilities, not a new trained model."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--archive',required=True)
parser.add_argument('--arms',nargs=2,required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
report=dict(purpose='Individual vs collective unknown competition, no alternative rule fitted or accuracy optimization',
    targets_used='Evaluation annotations only',arms={})
with zipfile.ZipFile(args.archive) as bundle:
    for arm in args.arms:
        with np.load(io.BytesIO(bundle.read(arm+'/office31-a2w_seed3/final-target-predictions.npz'))) as data:
            p=data['probabilities'];truth=data['truth'];pred=data['predictions'];epoch=int(data['epoch'])
        if p.shape[0]!=len(truth) or p.shape[1]<=10 or epoch!=10:raise ValueError('Require final10 recorded probabilities')
        if not np.array_equal(p.argmax(1),pred):raise ValueError('Recorded head predictions/probabilities differ')
        known_max=p[:,:10].max(1)
        unknown_max=p[:,10:].max(1)
        unknown_sum=p[:,10:].sum(1)
        split=(unknown_sum>known_max)&(unknown_max<=known_max)
        known=truth<10
        report['arms'][arm]=dict(K=p.shape[1]-10,
            collective_unknown_exceeds_known_but_no_individual_wins=int(split.sum()),
            split_fraction_all=float(split.mean()),split_fraction_true_unknown=float(split[~known].mean()),
            split_fraction_true_known=float(split[known].mean()),
            source_calibrated_readout_threshold_available=False,
            caveat='A mass competition event is not proof that a changed rule preserves known boundaries')
Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2))
