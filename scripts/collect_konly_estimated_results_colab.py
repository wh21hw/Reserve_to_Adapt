"""Collect only the finished estimated-K arm."""
import os
from pathlib import Path
os.environ['KONLY_COLLECT_ARM'] = 'estimated'
exec(compile(Path('/content/collect_konly_results_colab.py').read_text(),
             '/content/collect_konly_results_colab.py', 'exec'),
     dict(__name__='__main__', __file__='/content/collect_konly_results_colab.py'))
