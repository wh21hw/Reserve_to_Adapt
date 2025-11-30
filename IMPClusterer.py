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
    

    def estimate_lambda(self, features):
        """
        计算特征方差
        feature维度为 (N, D)， N为样本数，D为特征维度
        """
        n_samples = features.size(0)
        dim = features.size(1)
        if n_samples >1:
            var = features.var(dim = 0).sum()
            rho = features.var(dim = 0),mean()
        else:
            rho = torch.tensor(1.0).to(features.device)
        #目标域无监督 不能学习sigma
        #用标准差作为初始估计
        sigma = torch.sqrt(rho)

        alpha_val = self.alpha if self.alpha is not None else 0.05
        dim = features.size(1)
        #根据IMP论文经验公式计算lambda
        lam = -2 *sigma * np.log(alpha_val) + dim * sigma * np.log(1 + rho / sigma)
        return lam, sigma
    
    def fit(self, features, max_clusters = 100):
        """
        输入特征进行聚类
        输出 centroids：[K, D]聚类中心
        """
        if not torch.is_tensor(features):
            features = torch.from_numpy(features)
        
        if torch.cuda.is_available():
            features = features.cuda()
        
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
            dists_matrix = torch.sum((features.unsqueeze(1) - centroids.unsqueezq(0))**2, dim = -1)

            assignments = torch.argmin(dists_matrix, dim = 1)
            #更新原型
            new_centroids = []
            current_k = centroids.size(0)
            for k in range(current_k):
                # 属于k的样本
                mask = (assignments == k)
                if mask.sum() >0:
                    cluster_data = features[mask]
                    new_mean = cluster_data.mean(dim = 0)
                    new_centroids.append(new_mean.unsqueeze(0))
                else:
                    #删除空簇
                    pass
            if len(new_centroids) > 0:
                centroids = torch.stack(new_centroids, dim =0)
            else:
                #保留至少一个
                centroids = features.mean(dim = 0).unsqueeze(0)
        return centroids
