"""Read-only training diagnostics: no target labels, RNG, or gradient changes."""
import json
from pathlib import Path
import torch


class FusionDiagnostics:
    def __init__(self):
        self.rows = []

    @torch.no_grad()
    def add(self, logits, known_count, selected, known_weight):
        scores = logits.detach()
        known = scores[:, :known_count].max(1)[0]
        unknown, slots = scores[:, known_count:].max(1)
        probability = scores.softmax(1)
        mass = probability[:, known_count:].sum(1)
        known_max_probability = probability[:, :known_count].max(1)[0]
        mask = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
        mask[selected.reshape(-1).long()] = True
        self.rows.append(dict(margin=(unknown-known).cpu(), slots=slots.cpu(),
                              predicted=(unknown>known).cpu(), selected=mask.cpu(),
                              gate=known_weight.detach().cpu(), mass=mass.cpu(),
                              split=((mass>known_max_probability)&(unknown<=known)).cpu(),
                              K=scores.shape[1]-known_count))

    def save(self, folder, epoch):
        if not self.rows:
            return
        joined = {key:torch.cat([row[key] for row in self.rows])
                  for key in ('margin','slots','predicted','selected','gate','mass','split')}
        k = self.rows[0]['K']
        selected = joined['selected']
        report = dict(epoch=epoch, K=k, target_occurrences=len(selected),
                      unknown_selected_fraction=selected.float().mean().item(),
                      known_alignment_fraction=(joined['gate']>0).float().mean().item(),
                      unknown_prediction_fraction=joined['predicted'].float().mean().item(),
                      unknown_mass_mean=joined['mass'].mean().item(),
                      mass_exceeds_known_max_but_argmax_known_fraction=joined['split'].float().mean().item(),
                      unknown_minus_known_max_logit_mean=joined['margin'].mean().item(),
                      selected_unknown_slot_counts=torch.bincount(joined['slots'][selected],minlength=k).tolist(),
                      predicted_unknown_slot_counts=torch.bincount(joined['slots'][joined['predicted']],minlength=k).tolist(),
                      scope='training augmented target occurrences; not unique samples; no target labels')
        with (Path(folder)/'mechanism-history.jsonl').open('a') as stream:
            stream.write(json.dumps(report, allow_nan=False)+'\n')
        print('FUSION_DIAGNOSTICS', json.dumps(report), flush=True)
        self.rows.clear()
