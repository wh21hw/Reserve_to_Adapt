"""Full budget uses the pinned repaired pilot, never resumes a short pilot."""
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, '/content')
from matched_full_budget import build, PILOT_SHA256

base = Path('/content/train_matched_structure_pilot_fixed_v2.py').read_bytes()
assert hashlib.sha256(base).hexdigest() == PILOT_SHA256
source = build(base.decode())
namespace = dict(__name__='__main__', __file__=__file__, PILOT_SHA256=PILOT_SHA256,
                 GENERATED_SHA256=hashlib.sha256(source.encode()).hexdigest())
exec(compile(source, __file__, 'exec'), namespace)
