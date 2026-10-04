"""Same supported K and budget; isolate repeated virtual prototype module."""
from pathlib import Path
import subprocess
import os

# Existing stable supported-capacity launcher, altered only experiment identity
# and virtual update entry. Count inference/seed/prior/loss/budget stay identical.
source = Path('/content/run_fusion_supported_capacity_colab.py').read_text()
source = source.replace('/content/imp-runs/fusion-supported-capacity-v1',
                        '/content/imp-runs/fusion-q20-virtual-control-v1')
source = source.replace('/content/train_fusion_imp_rta_entry.py','/content/train_fusion_q20_entry.py')
source = source.replace("only K counting; virtual directions remain raw IMP",
                        "supported K unchanged; only repeated virtual update becomes Q20 K-means")
# One focused build check before starting the stable ten-epoch training workflow.
checked = subprocess.run(['/content/rta-py38/bin/python','/content/train_fusion_q20_entry.py'],
    env=dict(os.environ,KONLY_SOURCE_PRIOR='build-only',FUSION_BUILD_ONLY='1',FUSION_EPOCHS='10',FUSION_DIAGNOSTICS='1'),
    cwd='/content',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(checked.stdout,flush=True)
checked.check_returncode()
exec(compile(source,'<q20-supported-launcher>','exec'))
