import subprocess
import sys
from pathlib import Path

workspace = Path('/content/osda-pipeline')
if not workspace.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'codex/colab-pipeline',
                    'https://github.com/wh21hw/Reserve_to_Adapt.git', str(workspace)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(workspace / 'requirements-colab.txt')], check=True)
import torch
import torchvision
import faiss
import sklearn
print('ENVIRONMENT_READY', dict(torch=torch.__version__, torchvision=torchvision.__version__,
                               gpu=torch.cuda.get_device_name(), sklearn=sklearn.__version__), flush=True)
