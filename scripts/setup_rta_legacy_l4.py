"""Install the T4 reproduction's dependency versions in an isolated L4 venv.

No training, GPU operation, active-kernel changes or automatic retry.
"""
from pathlib import Path
import subprocess

python = '/content/rta-py38/bin/python'
commands = []
if not Path(python).exists():
    commands.append(['uv', 'venv', '--python', '3.8', '/content/rta-py38'])
commands += [
    ['uv', 'pip', 'install', '--python', python,
     'torch==1.7.1+cu110', 'torchvision==0.8.2+cu110',
     '--find-links', 'https://download.pytorch.org/whl/torch_stable.html'],
    ['uv', 'pip', 'install', '--python', python, 'numpy==1.23.4',
     'scipy==1.9.1', 'scikit-learn==1.1.2', 'faiss-cpu==1.7.4',
     'matplotlib==3.6.1', 'Pillow==9.2.0', 'tqdm==4.64.1'],
]
for command in commands:
    print('LEGACY_SETUP', ' '.join(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait():
        raise RuntimeError('Legacy dependency setup failed; preserve output')
print('LEGACY_ENV_INSTALLED; GPU compatibility not tested yet', flush=True)
