import torch
import numpy as np


class IMPClusterer:
    def __init__(self, alpha=None, num_cluster_steps=5):
        """Original comments: smaller alpha gives more clusters; Gibbs 5 steps.

        Historical comments retained here, not endorsed as correct theory.
        """
        self.alpha = alpha
        self.num_cluster_steps = num_cluster_steps
        self.source_centroids_fixed = None

    def compute_distance(self, protos, example):
        dist = torch.sum((protos - example)**2, dim=-1)
        return dist

    def estimate_lambda(self, features):
        n_samples = features.size(0)
        dim = features.size(1)
        if n_samples > 1:
            rho = features.var(dim=0).mean()
        else:
            rho = torch.tensor(1e-6, dtype=features.dtype, device=features.device)
        sigma = torch.sqrt(rho)
        alpha_val = self.alpha if self.alpha is not None else 0.05
        dim = features.size(1)
        alpha_tensor = torch.tensor(alpha_val, dtype=features.dtype, device=features.device)
        lam = -2 * sigma * torch.log(alpha_tensor) + dim * sigma * torch.log(1 + rho / sigma)
        return lam, sigma

    def soft_label(self, features, centroids, sigma):
        N, D = features.size()
        K = centroids.size(0)
        dists_matrix = torch.sum((features.unsqueeze(1) - centroids.unsqueeze(0))**2, dim=-1)
        guassian_prob = torch.exp(-dists_matrix / (2 * sigma**2))
        prob = guassian_prob / (torch.sum(guassian_prob, dim=-1, keepdim=True) + 1e-8)
        return prob

    def fit(self, features, max_clusters=100, source_centroids=None,
            use_source_centroids=False, fix_source_centroids=True):
        N, D = features.size()
        lam, sigma = self.estimate_lambda(features)
        if use_source_centroids and source_centroids is not None and source_centroids.size(0) > 0:
            self.source_centroids_fixed = source_centroids.clone().detach()
            self.source_centroids_fixed = self.source_centroids_fixed.to(features.device, dtype=features.dtype)
            centroids = self.source_centroids_fixed
        else:
            centroids = features[0].unsqueeze(0)
            self.source_centroids_fixed = None
        for step in range(self.num_cluster_steps):
            for i in range(N):
                ex = features[i]
                distances = self.compute_distance(centroids, ex)
                min_dist = torch.min(distances)
                if min_dist > lam:
                    new_proto = ex.unsqueeze(0)
                    centroids = torch.cat([centroids, new_proto])
            prob = self.soft_label(features, centroids, sigma)
            new_centroids = []
            current_k = centroids.size(0)
            for k in range(current_k):
                if fix_source_centroids and self.source_centroids_fixed is not None and k < self.source_centroids_fixed.size(0):
                    new_centroids.append(self.source_centroids_fixed[k])
                    continue
                cluster_prob = prob[:, k]
                if cluster_prob.sum() < 1e-8:
                    continue
                weighted_sum = torch.sum(features * cluster_prob.unsqueeze(1), dim=0)
                new_centroid = weighted_sum / cluster_prob.sum()
                new_centroids.append(new_centroid)
            if len(new_centroids) > 0:
                centroids = torch.stack(new_centroids, dim=0)
            else:
                centroids = features.mean(dim=0).unsqueeze(0)
        return centroids
