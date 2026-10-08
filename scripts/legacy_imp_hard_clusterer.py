"""Hard assignment ablation of the user's unchanged legacy clusterer."""
import torch
from IMPClusterer_user_snapshot import IMPClusterer as LegacyIMPClusterer


class IMPClusterer(LegacyIMPClusterer):
    def soft_label(self, features, centroids, sigma):
        # The inherited fit updates means from these responsibilities.
        # Keep its cluster birth threshold, fixed source anchors and empty-cluster removal.
        distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        assignments = distances.argmin(dim=1)
        return torch.nn.functional.one_hot(assignments, num_classes=centroids.size(0)).to(features.dtype)
