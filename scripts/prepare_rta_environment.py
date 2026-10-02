"""Prepare the paper-era Python environment without changing the active Colab kernel."""
import subprocess
from pathlib import Path

def run(args):
    print('+', ' '.join(args), flush=True)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, flush=True)
    result.check_returncode()

if not Path('/content/rta-py38/bin/python').exists():
    run(['uv', 'venv', '--python', '3.8', '/content/rta-py38'])
run(['uv', 'pip', 'install', '--python', '/content/rta-py38/bin/python',
     'torch==1.7.1+cu110', 'torchvision==0.8.2+cu110',
     '--find-links', 'https://download.pytorch.org/whl/torch_stable.html'])
run(['uv', 'pip', 'install', '--python', '/content/rta-py38/bin/python',
     'numpy==1.23.4', 'scipy==1.9.1', 'scikit-learn==1.1.2',
     'faiss-cpu==1.7.4', 'matplotlib==3.6.1', 'Pillow==9.2.0',
     'tqdm==4.64.1'])
run(['/content/rta-py38/bin/python', '-c',
     'import torch, torchvision, faiss; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name()); print(torch.ones(2,device="cuda")*2)'])
