"""One interface check, not evidence of unknown-class discovery performance."""
import subprocess
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_precision_capacity import estimate_source_cost_capacity

rng = np.random.RandomState(7)
source = np.vstack([rng.normal(0, .05, (40, 2)), rng.normal(2, .05, (40, 2))])
labels = np.repeat([0, 1], 40)
target = np.vstack([source[::4]+.02, rng.normal(4, .05, (8, 2))])
old = {}
exec(subprocess.check_output(['git', 'show', 'HEAD:source_precision_capacity.py'], text=True), old)
baseline, baseline_settings = old['estimate_source_cost_capacity'](source, labels, target, 16)
default, settings = estimate_source_cost_capacity(source, labels, target, 16)
balanced, alternative = estimate_source_cost_capacity(source, labels, target, 16,
    prior_mass_mode='domain_balanced')
np.testing.assert_array_equal(default['centers'], baseline['centers'])
np.testing.assert_array_equal(default['assignments'], baseline['assignments'])
assert default['history'] == baseline['history']
for key in ('lambda_radius', 'birth_cost', 'reference_samples', 'anchor_rows', 'birth_cost_rows'):
    assert settings[key] == baseline_settings[key] == alternative[key]
np.testing.assert_allclose(alternative['source_prior_precision'],
    np.asarray(settings['source_prior_precision'])/2)
assert settings['prior_to_target_mass_ratio'] == 2
assert alternative['prior_to_target_mass_ratio'] == 1
try:
    estimate_source_cost_capacity(source, labels, target, prior_mass_mode='invalid')
except ValueError:
    pass
else:
    raise AssertionError('Invalid mode accepted')
print('PRIOR_MASS_INTERFACE_OK: default unchanged; only empirical prior precision changed. Not a real-feature experiment.')
