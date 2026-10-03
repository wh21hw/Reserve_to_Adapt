"""Fresh independent on-arm audit; no training."""
import subprocess
import sys
subprocess.run([sys.executable, '-u', '/content/collect_matched_structure_pilot.py', '--arm', 'structure_on'], check=True)
