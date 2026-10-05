"""Change only the post-warmup unknown pseudo-label CE coefficient, 1 -> 0."""
from pathlib import Path

original = Path('/content/train_visda_frozenbn_rta_entry.py')
entry = original.read_text()
anchor = "compiled = compile(source, str(root/'main.py'), 'exec')"
if entry.count(anchor) != 1:
    raise RuntimeError('Unexpected completed-control entry interface')
old_loss = 'loss = ce + 0.01 * virtual_ce + 0.3 * adv_loss + 1 * entropy + 1 * ce_ep'
new_loss = 'loss = ce + 0.01 * virtual_ce + 0.3 * adv_loss + 1 * entropy + 0 * ce_ep'
# Keep unknown selection and classifier forward: removing their computations
# would also change head BN statistics and no longer be a coefficient-only arm.
patch = 'replace_once(' + repr(old_loss) + ', ' + repr(new_loss) + ')\n'
patch += "print('UNKNOWN_CE_WEIGHT_ABLATION', dict(control=1, candidate=0, warmup_unchanged=True, forward_unchanged=True), flush=True)\n"
entry = entry.replace(anchor, patch + anchor)
exec(compile(entry, str(original), 'exec'), dict(__name__='__main__', __file__=str(original)))
