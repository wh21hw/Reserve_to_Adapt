"""C20 retraining without any supervision of simulated unknown source classes."""
import json
import os
from pathlib import Path
import subprocess

data=Path('/content/osda-officehome-pr2rw-v1')
output=Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
output.mkdir(parents=True,exist_ok=False)
rows=[line.rsplit(None,1) for line in (data/'product_0-24_train_all.txt').read_text().splitlines() if line.strip()]
hidden=list(range(10,15))
known=[row for row in rows if int(row[1]) not in hidden]
(output/'known-source.txt').write_text(''.join('%s %s\n'%tuple(row) for row in known))
# Extraction only: filenames from source domain, labels replaced by sentinel.
(output/'extract-all-source.txt').write_text(''.join(row[0]+' 0\n' for row in rows))
command=['/content/rta-py38/bin/python','-u','/content/train_source_prior_konly.py',
    '--code-root','/content/rta-legacy-l4-bridge-v1','--source',str(output/'known-source.txt'),
    '--target',str(output/'extract-all-source.txt'),'--data-root',str(data),
    '--weights','/content/osda-datasets/resnet50-19c8e357.pth',
    '--output',str(output/'source'),'--seed','1','--epochs','3']
(output/'launch.json').write_text(json.dumps(dict(hidden_ids=hidden,C=20,epochs=3,seed=1,
    known_samples=len(known),extract_source_samples=len(rows),real_target_used=False,
    hidden_supervision=False,selection='posthoc hardest block from geometry probe',command=command),indent=2))
with (output/'console.log').open('x') as stream:
    process=subprocess.Popen(command,cwd='/content',env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'),
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True); stream.write(line); stream.flush()
    if process.wait():
        raise RuntimeError('Source leave-class training failed; preserve evidence')
print('SOURCE_LEAVECLASS_COMPLETE',flush=True)
