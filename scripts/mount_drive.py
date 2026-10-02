from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
for folder in ('datasets', 'pretrained', 'runs'):
    Path('/content/drive/MyDrive/OSDA', folder).mkdir(parents=True, exist_ok=True)
print('DRIVE_READY', flush=True)
