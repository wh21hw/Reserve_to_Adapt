import torch 
import numpy as np

class IMPClusterer:
    def __init__(self, alpha = None, num_cluster_steps = 5):
        """
        alpha concentraition parameter
        越小聚类越多
        None 根据数据自动计算
        gibbs采样通常5次收敛
        """
        self.alpha = alpha
        self.num_cluster_steps = num_cluster_steps
    
    def compute_distance(self, protos, example):
        """
        计算样本与聚类中心的距离
        protos: [K, D]
        example: [D] or [1,D]
        输出 distances: [K]
        """
        dist = torch.sum((protos - example)**2, dim = -1)
        return dist
    

    # 建议替换 estimate_lambda 为以下版本，以确保类型和设备兼容性

    def estimate_lambda(self, features):
        """
        计算特征方差
        feature维度为 (N, D)， N为样本数，D为特征维度
        """
        n_samples = features.size(0)
        dim = features.size(1)

        if n_samples > 1:
            # var = features.var(dim = 0).sum() # 仅用于参考，不用于计算 lam
            rho = features.var(dim = 0).mean() # 均方差
        else:
            # 极少样本数时的稳定值
            rho = torch.tensor(1e-6, dtype=features.dtype, device=features.device)
    
        # 用标准差作为初始估计
        sigma = torch.sqrt(rho)

        alpha_val = self.alpha if self.alpha is not None else 0.05
        dim = features.size(1)
    
        # 确保 alpha 值为张量，且在正确的设备和类型上
        alpha_tensor = torch.tensor(alpha_val, dtype=features.dtype, device=features.device)
    
        # 根据IMP论文经验公式计算 lambda (lam 是距离平方阈值)
        # lam = -2*sigma*log(alpha) + D*sigma*log(1 + rho/sigma)
        lam = -2 * sigma * torch.log(alpha_tensor) + dim * sigma * torch.log(1 + rho / sigma)
    
        return lam, sigma

    def soft_label(self, features, centroids, sigma):
        """
        输出样本的软分配概率
        """
        N, D = features.size()
        K = centroids.size(0)

        dists_matrix = torch.sum((features.unsqueeze(1) - centroids.unsqueeze(0))**2, dim = -1)
        guassian_prob = torch.exp(-dists_matrix / (2* sigma**2))
        prob = guassian_prob / (torch.sum(guassian_prob, dim = -1, keepdim = True) + 1e-8)
        return prob
    
    def fit(self, features, max_clusters = 100):
        """
        输入特征进行聚类
        输出 centroids：[K, D]聚类中心
        """
        
        N, D = features.size()
        lam, sigma = self.estimate_lambda(features)
        #第一个点为聚类中心
        centroids = features[0].unsqueeze(0)
        for step in range(self.num_cluster_steps):
            for i in range(N):
                ex = features[i]
                distances = self.compute_distance(centroids, ex)
                min_dist = torch.min(distances)

                if min_dist > lam:
                    new_proto = ex.unsqueeze(0)
                    centroids = torch.cat([centroids, new_proto])
            #重新分配点
            prob = self.soft_label(features, centroids, sigma) 
            #更新原型
            new_centroids = []
            current_k = centroids.size(0)
            for k in range(current_k):
                # 属于k的样本
                cluster_prob = prob[:, k]
                if cluster_prob.sum() <1e-8:
                    continue
                weighted_sum = torch.sum(features * cluster_prob.unsqueeze(1), dim = 0)
                new_centroid = weighted_sum / cluster_prob.sum()
                new_centroids.append(new_centroid)
            if len(new_centroids) > 0:
                centroids = torch.stack(new_centroids, dim =0)
            else:
                #保留至少一个
                centroids = features.mean(dim = 0).unsqueeze(0)
        return centroids
