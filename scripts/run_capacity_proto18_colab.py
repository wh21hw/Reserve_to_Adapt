from pathlib import Path
PILOT_ARM = 'proto18'
exec(compile(Path('/content/run_capacity_pilot_colab.py').read_text(), 'run_capacity_pilot_colab.py', 'exec'))
