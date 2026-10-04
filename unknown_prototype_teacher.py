"""Independent geometric pseudo-label teacher, fixed K and EMA centers.

Consumes only RTA-selected unknown features. Not a semantic discovery guarantee.
Uses the original warm-end K-means centers, not classifier weight coordinates.
"""
import json
from pathlib import Path
import torch


class UnknownPrototypeTeacher:
    def __init__(self, known_count, momentum=0.9):
        self.known_count = known_count
        self.momentum = momentum
        self.centers = None
        self.counts = None
        self.agreements = 0
        self.total = 0

    @torch.no_grad()
    def initialize(self, centers, folder):
        if centers.ndim != 2 or len(centers) < 1 or not torch.isfinite(centers).all():
            raise ValueError('Invalid teacher centers')
        self.centers = centers.detach().clone()
        self.counts = torch.zeros(len(centers),dtype=torch.long,device=centers.device)
        self.folder = folder
        print('UNKNOWN_TEACHER_INITIALIZED',len(centers),flush=True)

    @torch.no_grad()
    def assign(self, features, logits):
        if self.centers is None:
            # Warm-up loss multiplier is zero; original pseudo labels are retained.
            return logits[:,self.known_count:].argmax(1)+self.known_count
        if logits.shape[1]-self.known_count != len(self.centers):
            raise ValueError('Teacher K differs from classifier K')
        distances = (features.detach()[:,None]-self.centers[None]).square().sum(-1)
        index = distances.argmin(1)
        self.agreements += int((index == logits[:,self.known_count:].argmax(1)).sum())
        self.total += len(index)
        self.counts += torch.bincount(index,minlength=len(self.centers))
        for slot in index.unique():
            mean = features.detach()[index==slot].mean(0)
            self.centers[slot].mul_(self.momentum).add_(mean,alpha=1-self.momentum)
        return index+self.known_count

    def save(self, epoch):
        if self.centers is None:
            return
        report = dict(epoch=epoch,K=len(self.centers),momentum=self.momentum,
                      assignment_counts=self.counts.cpu().tolist(),
                      logit_teacher_agreement=self.agreements/max(self.total,1),
                      selected_occurrences=self.total,target_labels_used=False)
        with (Path(self.folder)/'teacher-history.jsonl').open('a') as stream:
            stream.write(json.dumps(report)+'\n')
        print('UNKNOWN_TEACHER',json.dumps(report),flush=True)
        self.counts.zero_()
        self.agreements = self.total = 0
